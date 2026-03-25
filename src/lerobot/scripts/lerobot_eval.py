#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if torch.is_tensor(value[0]):
            value = torch.stack(value, dim=0)
        else:
            value = np.stack(value, axis=0)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.array(value)


def _extract_overlay_state(policy: PreTrainedPolicy):
    if not hasattr(policy, "get_token_selection_state"):
        return None, None, None, None
    state = policy.get_token_selection_state()
    if state is None:
        return None, None, None, None

    overlay_grid = _to_numpy(state.get("last_overlay_grid"))
    if overlay_grid is None:
        return None, None, None, None

    if overlay_grid.ndim == 4:
        overlay_grid = overlay_grid[0]
    elif overlay_grid.ndim == 3:
        overlay_grid = overlay_grid
    elif overlay_grid.ndim == 2:
        overlay_grid = overlay_grid[None, ...]
    else:
        return None, None, None, None

    token_scores = _to_numpy(state.get("last_token_scores"))
    if token_scores is not None:
        if token_scores.ndim == 3:
            token_scores = token_scores[0]
        elif token_scores.ndim != 2:
            token_scores = None

    region_scores = _to_numpy(state.get("last_region_scores"))
    if region_scores is not None:
        if region_scores.ndim == 4:
            region_scores = region_scores[0]
        elif region_scores.ndim == 3:
            region_scores = region_scores[0]
        elif region_scores.ndim != 2:
            region_scores = None

    region_patch = max(1, int(getattr(policy.config, "region_patch_size", 1)))
    if region_patch > 1:
        grid_h, grid_w = overlay_grid.shape[1:3]
        if grid_h % region_patch == 0 and grid_w % region_patch == 0:
            region_h = grid_h // region_patch
            region_w = grid_w // region_patch
            labels = overlay_grid.reshape(-1, region_h, region_patch, region_w, region_patch)
            if token_scores is not None and token_scores.shape[1] == grid_h * grid_w:
                scores = token_scores.reshape(-1, region_h, region_patch, region_w, region_patch)
                label_scores = np.zeros((labels.shape[0], 4, region_h, region_w), dtype=np.float32)
                for label in range(4):
                    label_scores[:, label] = (scores * (labels == label)).sum(axis=(2, 4))
                overlay_grid = label_scores.argmax(axis=1).astype(np.uint8)
            else:
                label_counts = np.zeros((labels.shape[0], 4, region_h, region_w), dtype=np.int32)
                for label in range(4):
                    label_counts[:, label] = (labels == label).sum(axis=(2, 4))
                overlay_grid = label_counts.argmax(axis=1).astype(np.uint8)

            if region_scores is not None and region_scores.shape[1] == region_h * region_w:
                region_scores = region_scores.reshape(-1, region_h, region_w)
            elif region_scores is None and token_scores is not None and token_scores.shape[1] == grid_h * grid_w:
                region_scores = token_scores.reshape(
                    -1, region_h, region_patch, region_w, region_patch
                ).mean(axis=(2, 4))

    heatmap_grid_out = _to_numpy(state.get("last_heatmap_grid"))
    if heatmap_grid_out is not None:
        # Shape from torch.stack: [num_cameras, B, H, W]
        # We want [num_cameras, H, W] (take batch=0)
        if heatmap_grid_out.ndim == 4:
            heatmap_grid_out = heatmap_grid_out[:, 0]
        elif heatmap_grid_out.ndim == 2:
            heatmap_grid_out = heatmap_grid_out[None, ...]

    keep_grid_out = _to_numpy(state.get("last_keep_grid"))
    if keep_grid_out is not None:
        if keep_grid_out.ndim == 4:
            keep_grid_out = keep_grid_out[:, 0]  # [num_cameras, H, W]
        elif keep_grid_out.ndim == 2:
            keep_grid_out = keep_grid_out[None, ...]

    return overlay_grid.astype(np.uint8), region_scores, heatmap_grid_out, keep_grid_out


def _apply_heatmap_overlay(img, heatmap_grid, region_scores=None, alpha=0.5,
                          keep_grid=None, prune_darken=0.5, prune_stripe_gap=4):
    """Overlay a continuous heatmap (blue→red) on *img* based on *heatmap_grid*.

    Args:
        heatmap_grid: float [H_region, W_region] in [0, 1].
        keep_grid:    bool/uint8 [H_region, W_region] (optional).
                      True/1 = kept, False/0 = pruned.
                      When provided, pruned regions are darkened and overlaid
                      with diagonal stripe hatching so you can clearly see which
                      tokens were pruned while still reading the heatmap colour.
        prune_darken: brightness multiplier for pruned regions (0=black, 1=no change).
        prune_stripe_gap: pixel spacing of diagonal hatching lines.
    """
    if heatmap_grid is None:
        return img

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img

    img_np = img.astype(np.uint8, copy=False)
    height, width = img_np.shape[:2]

    # Upsample heatmap to image resolution (nearest for sharp region boundaries)
    hm_h, hm_w = heatmap_grid.shape
    hm_img = Image.fromarray(heatmap_grid.astype(np.float32), mode="F")
    hm_up = np.array(hm_img.resize((width, height), resample=Image.NEAREST))

    # Colormap: blue (0) → cyan → green → yellow → red (1)
    def _jet_colormap(v):
        r = np.clip(1.5 - np.abs(v - 0.75) * 4.0, 0.0, 1.0)
        g = np.clip(1.5 - np.abs(v - 0.5) * 4.0, 0.0, 1.0)
        b = np.clip(1.5 - np.abs(v - 0.25) * 4.0, 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    heat_rgb = _jet_colormap(hm_up)
    blended = (img_np.astype(np.float32) * (1.0 - alpha)
               + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)

    # ── Mark pruned regions: darken + diagonal stripe hatching ────────────
    if keep_grid is not None:
        kg_arr = keep_grid.astype(np.uint8) if keep_grid.dtype != np.uint8 else keep_grid
        kg_img = Image.fromarray(kg_arr, mode='L').resize((width, height), resample=Image.NEAREST)
        kg_up = np.array(kg_img)
        pruned_mask = (kg_up == 0)  # True where pruned (keep=0)

        # 1) Darken pruned pixels
        blended_f = blended.astype(np.float32)
        blended_f[pruned_mask] *= prune_darken
        blended = blended_f.astype(np.uint8)

        # 2) Diagonal stripe hatching (white, semi-transparent) on pruned regions
        #    Pattern: pixel (x, y) is on a stripe if (x + y) % gap < gap//3
        if prune_stripe_gap > 0:
            yy, xx = np.mgrid[:height, :width]
            stripe = ((xx + yy) % prune_stripe_gap) < max(1, prune_stripe_gap // 3)
            stripe_mask = pruned_mask & stripe
            # Blend stripe colour (white, 40% opacity) onto the darkened image
            blended_f2 = blended.astype(np.float32)
            blended_f2[stripe_mask] = blended_f2[stripe_mask] * 0.6 + 255.0 * 0.4
            blended = blended_f2.astype(np.uint8)

    # ── Optionally draw region score numbers ──────────────────────────────
    if region_scores is None:
        return blended
    if torch.is_tensor(region_scores):
        region_scores = region_scores.detach().cpu().numpy()
    region_scores = region_scores.astype(np.float32)
    total = float(region_scores.sum())
    if total > 0:
        region_scores = region_scores / total

    region_h, region_w = heatmap_grid.shape
    cell_w = width / max(region_w, 1)
    cell_h = height / max(region_h, 1)
    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    score_font_size = max(7, min(12, int(min(cell_w, cell_h) * 0.38)))
    def _load_font(size):
        for font_name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()
    score_font = _load_font(score_font_size)
    for r in range(region_h):
        for c in range(region_w):
            x = int((c + 0.03) * cell_w)
            y = int((r + 0.03) * cell_h)
            score = region_scores[r, c]
            text = f"{score:.3f}"
            draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=score_font)
            draw.text((x, y), text, fill=(255, 255, 255), font=score_font)
    return np.array(pil_img)


def _apply_overlay(img, overlay_grid, region_scores=None, alpha=0.35, show_ids=False):
    if overlay_grid is None:
        return img

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img

    img_np = img.astype(np.uint8, copy=False)
    height, width = img_np.shape[:2]

    overlay_img = Image.fromarray(overlay_grid).resize((width, height), resample=Image.NEAREST)
    overlay_labels = np.array(overlay_img)

    overlay_colors = np.zeros_like(img_np)
    overlay_colors[overlay_labels == 0] = (0, 255, 0)       # green: not important
    overlay_colors[overlay_labels == 1] = (255, 255, 0)     # yellow: gate-protected (important + cosine-static)
    overlay_colors[overlay_labels == 2] = (255, 0, 0)       # red: important foreground
    overlay_colors[overlay_labels == 3] = (0, 0, 255)       # blue: important but pruned as bg (gate didn't protect)

    blended = (img_np.astype(np.float32) * (1.0 - alpha) + overlay_colors.astype(np.float32) * alpha).astype(
        np.uint8
    )
    # label=4 (pruned bg): transparent — restore original image
    bg_mask = overlay_labels == 4
    blended[bg_mask] = img_np[bg_mask]

    show_scores = region_scores is not None
    if not show_scores and not show_ids:
        return blended

    if torch.is_tensor(region_scores):
        region_scores = region_scores.detach().cpu().numpy()

    region_h, region_w = overlay_grid.shape
    cell_w = width / max(region_w, 1)
    cell_h = height / max(region_h, 1)

    if show_scores:
        region_scores = region_scores.astype(np.float32)
        total = float(region_scores.sum())
        if total > 0:
            region_scores = region_scores / total

    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    id_font_size = max(10, min(16, int(min(cell_w, cell_h) * 0.55)))
    score_font_size = max(7, min(12, int(min(cell_w, cell_h) * 0.38)))

    def _load_font(size):
        for font_name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    id_font = _load_font(id_font_size)
    score_font = _load_font(score_font_size)

    for r in range(region_h):
        for c in range(region_w):
            x = int((c + 0.03) * cell_w)
            y = int((r + 0.03) * cell_h)
            if show_ids:
                region_id = r * region_w + c
                id_text = str(region_id)
                draw.text((x + 1, y + 1), id_text, fill=(0, 0, 0), font=id_font)
                draw.text((x, y), id_text, fill=(255, 255, 255), font=id_font)
                y = y + int(id_font_size * 0.85)
            if show_scores:
                score = region_scores[r, c]
                text = f"{score:.3f}"
                draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=score_font)
                draw.text((x, y), text, fill=(255, 255, 255), font=score_font)

    return np.array(pil_img)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.no_grad():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        # Build postfix: start with success rate, then append token-selection
        # stats (entropy, avg eval interval) if the policy exposes them.
        _postfix = {"sr": f"{running_success_rate.item() * 100:.1f}%"}
        _model = getattr(policy, "model", None)
        _ts = getattr(_model, "_token_selection_state", None) if _model is not None else None
        if _ts is not None and getattr(_ts, "last_stats", None):
            _postfix.update(_ts.last_stats)
        progbar.set_postfix(_postfix)
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    overlay_enabled = bool(getattr(getattr(policy, "config", None), "token_selection_enabled", False))
    heatmap_debug = bool(getattr(getattr(policy, "config", None), "score_debug_heatmap", False))
    overlay_heatmap = getattr(getattr(policy, "config", None), "overlay_mode", "label") == "heatmap"
    show_scores = bool(getattr(getattr(policy, "config", None), "overlay_show_scores", False))
    show_ids = bool(getattr(getattr(policy, "config", None), "overlay_show_ids", False))
    last_overlay_grid = None
    last_region_scores = None

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        nonlocal last_overlay_grid, last_region_scores
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        overlay_grid = None
        region_scores = None
        heatmap_grid = None
        keep_grid = None
        if overlay_enabled:
            overlay_grid, region_scores, heatmap_grid, keep_grid = _extract_overlay_state(policy)
            if overlay_grid is not None:
                last_overlay_grid = overlay_grid
                last_region_scores = region_scores
            else:
                overlay_grid = last_overlay_grid
                region_scores = last_region_scores
        if isinstance(env, gym.vector.SyncVectorEnv):
            frames = [env.envs[i].render() for i in range(n_to_render_now)]  # noqa: B023
            if heatmap_grid is not None and (heatmap_debug or overlay_heatmap):
                rendered = []
                for idx, frame in enumerate(frames):
                    hm = heatmap_grid[idx] if idx < heatmap_grid.shape[0] else heatmap_grid[0]
                    scores = None
                    if show_scores and region_scores is not None:
                        scores = region_scores[idx] if idx < region_scores.shape[0] else region_scores[0]
                    kg = None
                    if keep_grid is not None and not heatmap_debug:
                        kg = keep_grid[idx] if idx < keep_grid.shape[0] else keep_grid[0]
                    rendered.append(_apply_heatmap_overlay(frame, hm, scores, keep_grid=kg))
                frames = rendered
            elif overlay_grid is not None:
                rendered = []
                for idx, frame in enumerate(frames):
                    grid = overlay_grid[idx] if idx < overlay_grid.shape[0] else overlay_grid[0]
                    scores = None
                    if show_scores and region_scores is not None:
                        scores = region_scores[idx] if idx < region_scores.shape[0] else region_scores[0]
                    rendered.append(_apply_overlay(frame, grid, scores, show_ids=show_ids))
                frames = rendered
            ep_frames.append(np.stack(frames))
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            frames = env.call("render")[:n_to_render_now]
            if heatmap_grid is not None and (heatmap_debug or overlay_heatmap):
                rendered = []
                for idx, frame in enumerate(frames):
                    hm = heatmap_grid[idx] if idx < heatmap_grid.shape[0] else heatmap_grid[0]
                    scores = None
                    if show_scores and region_scores is not None:
                        scores = region_scores[idx] if idx < region_scores.shape[0] else region_scores[0]
                    kg = None
                    if keep_grid is not None and not heatmap_debug:
                        kg = keep_grid[idx] if idx < keep_grid.shape[0] else keep_grid[0]
                    rendered.append(_apply_heatmap_overlay(frame, hm, scores, keep_grid=kg))
                frames = rendered
            elif overlay_grid is not None:
                rendered = []
                for idx, frame in enumerate(frames):
                    grid = overlay_grid[idx] if idx < overlay_grid.shape[0] else overlay_grid[0]
                    scores = None
                    if show_scores and region_scores is not None:
                        scores = region_scores[idx] if idx < region_scores.shape[0] else region_scores[0]
                    rendered.append(_apply_overlay(frame, grid, scores, show_ids=show_ids))
                frames = rendered
            ep_frames.append(np.stack(frames))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Flush any pending action L1 norm plot for the last episode.
    _flush_model = getattr(policy, "model", None)
    if _flush_model is not None:
        _flush_state = getattr(_flush_model, "_token_selection_state", None)
        _flush_cfg = getattr(_flush_model, "config", None)
        if (_flush_state is not None
                and _flush_cfg is not None
                and getattr(_flush_cfg, "interp_plot_actions_l1", False)
                and _flush_state.actions_l1_history):
            if not hasattr(_flush_model, '_plot_episode_idx'):
                _flush_model._plot_episode_idx = 0
            _flush_model._save_actions_l1_plot(_flush_state, _flush_model._plot_episode_idx)
            _flush_model._plot_episode_idx += 1

    # Collect final token selection stats if available.
    _token_sel_stats = {}
    _model = getattr(policy, "model", None)
    _ts_final = getattr(_model, "_token_selection_state", None) if _model is not None else None
    if _ts_final is not None and getattr(_ts_final, "last_stats", None):
        _token_sel_stats = dict(_ts_final.last_stats)
    if _ts_final is not None:
        _token_sel_stats["total_frames"] = _ts_final.frame_idx
        _token_sel_stats["total_eval_frames"] = _ts_final.eval_frame_count
        if _ts_final.frame_idx > 0 and _ts_final.eval_frame_count > 0:
            _token_sel_stats["avg_eval_interval"] = round(_ts_final.frame_idx / _ts_final.eval_frame_count, 2)
            _token_sel_stats["eval_rate"] = round(_ts_final.eval_frame_count / _ts_final.frame_idx, 4)
        # Per-episode average pruning ratio
        if _ts_final.total_possible > 0:
            _avg_prune = (1.0 - _ts_final.total_kept / _ts_final.total_possible) * 100.0
            _token_sel_stats["avg_prune_ratio"] = round(_avg_prune, 2)
            _token_sel_stats["total_kept"] = _ts_final.total_kept
            _token_sel_stats["total_possible"] = _ts_final.total_possible

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if _token_sel_stats:
        info["token_selection_stats"] = _token_sel_stats

    _sched_stats = {}
    if hasattr(policy, "get_model_scheduling_stats"):
        _maybe_stats = policy.get_model_scheduling_stats()
        if _maybe_stats:
            _sched_stats = dict(_maybe_stats)
    if _sched_stats:
        info["model_scheduling_stats"] = _sched_stats

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info("CLI args: %s", " ".join(sys.argv[1:]) or "<none>")
    logging.info("Resolved config:\n%s", pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # ── Full determinism ──
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_episodes_rendered,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict, total=False):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    token_selection_stats: dict
    model_scheduling_stats: dict


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    result = TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )
    if "token_selection_stats" in task_result:
        result["token_selection_stats"] = task_result["token_selection_stats"]
    if "model_scheduling_stats" in task_result:
        result["model_scheduling_stats"] = task_result["model_scheduling_stats"]
    return result


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # Collect per-group token selection stats from per_task_infos
    group_token_sel: dict[str, list[dict]] = defaultdict(list)
    group_sched: dict[str, list[dict]] = defaultdict(list)
    for ti in per_task_infos:
        ts = ti["metrics"].get("token_selection_stats")
        if ts:
            group_token_sel[ti["task_group"]].append(ts)
        sched = ti["metrics"].get("model_scheduling_stats")
        if sched:
            group_sched[ti["task_group"]].append(sched)

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        group_agg = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }
        # Aggregate token selection stats across tasks in this group
        ts_list = group_token_sel.get(group, [])
        if ts_list:
            _total_frames = sum(t.get("total_frames", 0) for t in ts_list)
            _total_evals = sum(t.get("total_eval_frames", 0) for t in ts_list)
            group_ts = {
                "total_frames": _total_frames,
                "total_eval_frames": _total_evals,
            }
            if _total_frames > 0 and _total_evals > 0:
                group_ts["avg_eval_interval"] = round(_total_frames / _total_evals, 2)
                group_ts["eval_rate"] = round(_total_evals / _total_frames, 4)
            _entropies = [t["final_entropy"] for t in ts_list if "final_entropy" in t]
            if _entropies:
                group_ts["avg_final_entropy"] = round(float(np.mean(_entropies)), 4)
            # Aggregate pruning ratio across tasks in this group
            _g_total_kept = sum(t.get("total_kept", 0) for t in ts_list)
            _g_total_possible = sum(t.get("total_possible", 0) for t in ts_list)
            if _g_total_possible > 0:
                group_ts["avg_prune_ratio"] = round((1.0 - _g_total_kept / _g_total_possible) * 100.0, 2)
                group_ts["total_kept"] = _g_total_kept
                group_ts["total_possible"] = _g_total_possible
            group_agg["token_selection_stats"] = group_ts
            _sr = group_agg.get("pc_success")
            _sr_str = f"{_sr:.1f}%" if _sr is not None and _sr == _sr else "N/A"
            logging.info(
                f"[{group}] Token selection: frames={_total_frames}, "
                f"eval_frames={_total_evals}, "
                f"avg_interval={group_ts.get('avg_eval_interval', 'N/A')}, "
                f"eval_rate={group_ts.get('eval_rate', 'N/A')}, "
                f"avg_entropy={group_ts.get('avg_final_entropy', 'N/A')}, "
                f"avg_prune={group_ts.get('avg_prune_ratio', 'N/A')}%, "
                f"success_rate={_sr_str}"
            )
        sched_list = group_sched.get(group, [])
        if sched_list:
            group_sched_agg = {
                "vla_calls": sum(s.get("vla_calls", 0) for s in sched_list),
                "vla_actions": sum(s.get("vla_actions", 0) for s in sched_list),
                "lightweight_actions": sum(s.get("lightweight_actions", 0) for s in sched_list),
                "generator_fallbacks": sum(s.get("generator_fallbacks", 0) for s in sched_list),
                "discarded_vla_actions": sum(s.get("discarded_vla_actions", 0) for s in sched_list),
                "scheduler_switches": sum(s.get("scheduler_switches", 0) for s in sched_list),
            }
            _total_actions = group_sched_agg["vla_actions"] + group_sched_agg["lightweight_actions"]
            if _total_actions > 0:
                group_sched_agg["lightweight_action_rate"] = round(
                    group_sched_agg["lightweight_actions"] / _total_actions, 4
                )
                group_sched_agg["vla_action_rate"] = round(
                    group_sched_agg["vla_actions"] / _total_actions, 4
                )
            group_agg["model_scheduling_stats"] = group_sched_agg
        groups_aggregated[group] = group_agg

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
