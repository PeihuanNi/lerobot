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
import math
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
from transformers.utils import logging as hf_logging

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
from PIL import Image, ImageDraw

from lerobot.datasets.image_writer import write_image
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)


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
        with torch.inference_mode():
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
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
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
    render_reuse_mask: bool = False,
    render_reuse_camera: str | None = None,
    save_attn_maps: bool = False,
    attn_save_interval: int = 1,
    attn_camera: str | None = None,
    render_attn_heatmap: bool = False,
    attn_heatmap_layer: int | str = -1,
    attn_heatmap_alpha: float = 0.5,
    attn_dir: Path | None = None,
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
        render_reuse_mask: Whether to overlay reuse/update masks on rendered videos.
        render_reuse_camera: Camera key to use when overlaying reuse/update masks.
        save_attn_maps: Whether to save per-layer attention maps to .pth files.
        attn_save_interval: Save attention maps every N forwards (action chunk inferences).
        attn_camera: Camera key to use when saving attention maps.
        render_attn_heatmap: Whether to save attention heatmaps as standalone images (no video overlay).
        attn_heatmap_layer: Which layer index to visualize (0-based, -1 for last, "mean" for average, or "all").
        attn_heatmap_alpha: Intensity scale for the saved heatmaps.
        attn_dir: Where to save attention map .pth files.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")
    needs_attn = save_attn_maps or render_attn_heatmap
    if needs_attn and max_episodes_rendered <= 0:
        raise ValueError("Attention map saving/visualization requires max_episodes_rendered > 0.")
    if needs_attn and not attn_dir:
        raise ValueError("If attention maps are enabled, attn_dir must be provided.")
    attn_save_interval = max(1, attn_save_interval)
    attn_heatmap_alpha = float(max(0.0, min(attn_heatmap_alpha, 1.0)))

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

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
    missing_attn_logged = False
    attn_camera_used: str | None = None
    reuse_camera_used: str | None = None
    last_forward_counter: int | None = None

    def _select_reuse_mask() -> tuple[np.ndarray | None, str | None]:
        model = getattr(policy, "model", None)
        if model is None:
            return None, None
        mask_dict = getattr(model, "_last_update_masks", None)
        if not mask_dict:
            return None, None
        if render_reuse_camera and render_reuse_camera in mask_dict:
            return mask_dict[render_reuse_camera], render_reuse_camera
        # fallback to first available
        first_key = next(iter(mask_dict))
        return mask_dict[first_key], first_key

    def _expand_mask_to_frame(mask_grid: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        mask_grid = mask_grid.astype(bool)
        frame_h, frame_w = frame_shape
        grid_h, grid_w = mask_grid.shape
        if grid_h == 0 or grid_w == 0:
            return np.zeros((frame_h, frame_w), dtype=bool)
        scale_h = int(math.ceil(frame_h / grid_h))
        scale_w = int(math.ceil(frame_w / grid_w))
        mask = np.repeat(np.repeat(mask_grid, scale_h, axis=0), scale_w, axis=1)
        return mask[:frame_h, :frame_w]

    def _overlay_reuse_mask(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
        # frames: (t, h, w, c), masks: (t, gh, gw)
        update_color = np.array([255, 0, 0], dtype=np.float32)
        reuse_color = np.array([0, 0, 255], dtype=np.float32)
        update_alpha = 0.1
        reuse_alpha = 0.1

        out = frames.astype(np.float32).copy()
        for idx, (frame, mask_grid) in enumerate(zip(out, masks, strict=False)):
            update_mask = _expand_mask_to_frame(mask_grid, frame.shape[:2])
            reuse_mask = ~update_mask
            if update_mask.any():
                frame[update_mask] = frame[update_mask] * (1.0 - update_alpha) + update_color * update_alpha
            if reuse_mask.any():
                frame[reuse_mask] = frame[reuse_mask] * (1.0 - reuse_alpha) + reuse_color * reuse_alpha
            out[idx] = frame
        return out.astype(np.uint8)

    def _select_attn_maps() -> tuple[list | None, str | None]:
        model = getattr(policy, "model", None)
        if model is None:
            return None, None
        attn_dict = getattr(model, "_last_attn_maps", None)
        if not attn_dict:
            return None, None
        if attn_camera and attn_camera in attn_dict:
            return attn_dict[attn_camera], attn_camera
        first_key = next(iter(attn_dict))
        return attn_dict[first_key], first_key

    def _select_reuse_analysis() -> tuple[dict | None, str | None]:
        model = getattr(policy, "model", None)
        if model is None:
            return None, None
        analysis_dict = getattr(model, "_last_reuse_analysis", None)
        if not analysis_dict:
            return None, None
        if attn_camera and attn_camera in analysis_dict:
            return analysis_dict[attn_camera], attn_camera
        first_key = next(iter(analysis_dict))
        return analysis_dict[first_key], first_key

    def _slice_reuse_analysis(analysis: dict, idx: int) -> dict:
        out: dict[str, Any] = {"metrics": analysis.get("metrics", {})}
        if "patch_diff_grid" in analysis and analysis["patch_diff_grid"] is not None:
            out["patch_diff_grid"] = analysis["patch_diff_grid"][idx]
        if "layer_token_diff_grids" in analysis:
            out["layer_token_diff_grids"] = [
                item[idx] if item is not None else None for item in analysis["layer_token_diff_grids"]
            ]
        if "layer_channel_diffs" in analysis:
            out["layer_channel_diffs"] = [
                item[idx] if item is not None else None for item in analysis["layer_channel_diffs"]
            ]
        if "attn_diff_maps" in analysis:
            out["attn_diff_maps"] = [
                item[idx] if item is not None else None for item in analysis["attn_diff_maps"]
            ]
        if "update_mask_grid" in analysis and analysis["update_mask_grid"] is not None:
            out["update_mask_grid"] = analysis["update_mask_grid"][idx]
        return out

    def _select_update_mask_for_attn() -> tuple[np.ndarray | None, str | None]:
        model = getattr(policy, "model", None)
        if model is None:
            return None, None
        mask_dict = getattr(model, "_last_update_masks", None)
        if not mask_dict:
            return None, None
        if attn_camera and attn_camera in mask_dict:
            return mask_dict[attn_camera], attn_camera
        first_key = next(iter(mask_dict))
        return mask_dict[first_key], first_key

    def _infer_update_modes(update_mask: np.ndarray | None, batch_size: int) -> list[str]:
        if update_mask is None:
            return ["full"] * batch_size
        mask_np = update_mask
        if torch.is_tensor(mask_np):
            mask_np = mask_np.detach().cpu().numpy()
        mask_np = np.asarray(mask_np, dtype=bool)
        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]
        modes = []
        for idx in range(min(batch_size, mask_np.shape[0])):
            grid = mask_np[idx]
            total = grid.size
            updated = int(grid.sum())
            mode = "full" if updated >= total else "partial"
            modes.append(mode)
        if len(modes) < batch_size:
            modes.extend([modes[-1]] * (batch_size - len(modes)))
        return modes

    def _get_forward_counter() -> int | None:
        model = getattr(policy, "model", None)
        if model is None:
            return None
        return getattr(model, "_frame_counter", None)

    def _expand_map_to_frame(map_grid: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        frame_h, frame_w = frame_shape
        grid_h, grid_w = map_grid.shape
        if grid_h == 0 or grid_w == 0:
            return np.zeros((frame_h, frame_w), dtype=np.float32)
        scale_h = int(math.ceil(frame_h / grid_h))
        scale_w = int(math.ceil(frame_w / grid_w))
        expanded = np.repeat(np.repeat(map_grid, scale_h, axis=0), scale_w, axis=1)
        return expanded[:frame_h, :frame_w]

    def _render_heatmap_image(
        layer_map: np.ndarray,
        frame_shape: tuple[int, int],
        intensity_scale: float,
        tick_count: int = 5,
    ) -> np.ndarray:
        grid_h, grid_w = layer_map.shape
        frame_h, frame_w = frame_shape
        expanded = _expand_map_to_frame(layer_map, frame_shape)
        intensity = np.clip(expanded * intensity_scale, 0.0, 1.0)
        heat = np.full((frame_h, frame_w, 3), 255, dtype=np.uint8)
        inv = (255.0 * (1.0 - intensity)).astype(np.uint8)
        heat[..., 1] = inv
        heat[..., 2] = inv

        margin_left = 60
        margin_right = 90
        margin_top = 20
        margin_bottom = 45
        bar_pad = 10
        bar_width = 20
        canvas_w = margin_left + frame_w + bar_pad + bar_width + margin_right
        canvas_h = margin_top + frame_h + margin_bottom

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        canvas.paste(Image.fromarray(heat), (margin_left, margin_top))
        draw = ImageDraw.Draw(canvas)

        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + frame_w - 1
        y1 = margin_top + frame_h - 1
        draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=1)
        draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=1)

        ticks = max(2, tick_count)
        for i in range(ticks):
            tx = x0 + int(i * (frame_w - 1) / (ticks - 1))
            draw.line([(tx, y1), (tx, y1 + 4)], fill=(0, 0, 0), width=1)
            label_x = str(int(round(i * (grid_w - 1) / (ticks - 1))))
            draw.text((tx - 6, y1 + 8), label_x, fill=(0, 0, 0))

            ty = y0 + int(i * (frame_h - 1) / (ticks - 1))
            draw.line([(x0 - 4, ty), (x0, ty)], fill=(0, 0, 0), width=1)
            label_y = str(int(round(i * (grid_h - 1) / (ticks - 1))))
            draw.text((x0 - 32, ty - 6), label_y, fill=(0, 0, 0))

        draw.text((x0 + frame_w // 2 - 25, y1 + 25), "X (patch)", fill=(0, 0, 0))
        draw.text((5, y0 - 2), "Y (patch)", fill=(0, 0, 0))

        bar_x0 = x1 + bar_pad
        bar_x1 = bar_x0 + bar_width - 1
        bar = np.full((frame_h, bar_width, 3), 255, dtype=np.uint8)
        for j in range(frame_h):
            val = 1.0 - (j / max(frame_h - 1, 1))
            bar_intensity = np.clip(val * intensity_scale, 0.0, 1.0)
            inv = int(255.0 * (1.0 - bar_intensity))
            bar[j, :, 1] = inv
            bar[j, :, 2] = inv
        canvas.paste(Image.fromarray(bar), (bar_x0, margin_top))
        draw.rectangle([bar_x0, y0, bar_x1, y1], outline=(0, 0, 0), width=1)
        draw.text((bar_x0 - 2, y0 - 14), "attn", fill=(0, 0, 0))
        scale_steps = 10
        for i in range(scale_steps + 1):
            val = 1.0 - (i / scale_steps)
            ty = y0 + int(i * (frame_h - 1) / scale_steps)
            draw.line([(bar_x1, ty), (bar_x1 + 4, ty)], fill=(0, 0, 0), width=1)
            draw.text((bar_x1 + 6, ty - 6), f"{val:.1f}", fill=(0, 0, 0))

        return np.asarray(canvas)

    def _render_matrix_heatmap(
        matrix: np.ndarray,
        intensity_scale: float,
        tick_count: int = 5,
    ) -> np.ndarray:
        rows, cols = matrix.shape
        intensity = np.clip(matrix * intensity_scale, 0.0, 1.0)
        heat = np.full((rows, cols, 3), 255, dtype=np.uint8)
        inv = (255.0 * (1.0 - intensity)).astype(np.uint8)
        heat[..., 1] = inv
        heat[..., 2] = inv

        margin_left = 60
        margin_right = 90
        margin_top = 20
        margin_bottom = 45
        bar_pad = 10
        bar_width = 20
        canvas_w = margin_left + cols + bar_pad + bar_width + margin_right
        canvas_h = margin_top + rows + margin_bottom

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        canvas.paste(Image.fromarray(heat), (margin_left, margin_top))
        draw = ImageDraw.Draw(canvas)

        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + cols - 1
        y1 = margin_top + rows - 1
        draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=1)
        draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=1)

        ticks = max(2, tick_count)
        for i in range(ticks):
            tx = x0 + int(i * (cols - 1) / (ticks - 1))
            draw.line([(tx, y1), (tx, y1 + 4)], fill=(0, 0, 0), width=1)
            label_x = str(int(round(i * (cols - 1) / (ticks - 1))))
            draw.text((tx - 8, y1 + 8), label_x, fill=(0, 0, 0))

            ty = y0 + int(i * (rows - 1) / (ticks - 1))
            draw.line([(x0 - 4, ty), (x0, ty)], fill=(0, 0, 0), width=1)
            label_y = str(int(round(i * (rows - 1) / (ticks - 1))))
            draw.text((x0 - 36, ty - 6), label_y, fill=(0, 0, 0))

        draw.text((x0 + cols // 2 - 25, y1 + 25), "K (token)", fill=(0, 0, 0))
        draw.text((5, y0 - 2), "Q (token)", fill=(0, 0, 0))

        bar_x0 = x1 + bar_pad
        bar_x1 = bar_x0 + bar_width - 1
        bar = np.full((rows, bar_width, 3), 255, dtype=np.uint8)
        for j in range(rows):
            val = 1.0 - (j / max(rows - 1, 1))
            bar_intensity = np.clip(val * intensity_scale, 0.0, 1.0)
            inv = int(255.0 * (1.0 - bar_intensity))
            bar[j, :, 1] = inv
            bar[j, :, 2] = inv
        canvas.paste(Image.fromarray(bar), (bar_x0, margin_top))
        draw.rectangle([bar_x0, y0, bar_x1, y1], outline=(0, 0, 0), width=1)
        draw.text((bar_x0 - 2, y0 - 14), "attn", fill=(0, 0, 0))
        scale_steps = 10
        for i in range(scale_steps + 1):
            val = 1.0 - (i / scale_steps)
            ty = y0 + int(i * (rows - 1) / scale_steps)
            draw.line([(bar_x1, ty), (bar_x1 + 4, ty)], fill=(0, 0, 0), width=1)
            draw.text((bar_x1 + 6, ty - 6), f"{val:.1f}", fill=(0, 0, 0))

        return np.asarray(canvas)

    def _normalize_map(map_array: np.ndarray) -> np.ndarray:
        map_array = np.nan_to_num(map_array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        min_val = float(map_array.min())
        max_val = float(map_array.max())
        if max_val > min_val:
            return (map_array - min_val) / (max_val - min_val)
        return np.zeros_like(map_array, dtype=np.float32)

    def _pick_attn_layer(attn_map: np.ndarray) -> np.ndarray:
        if attn_map.ndim != 3:
            raise ValueError("Expected attn_map with shape (layers, h, w).")
        if isinstance(attn_heatmap_layer, str) and attn_heatmap_layer == "mean":
            return attn_map.mean(axis=0)
        layer_idx = int(attn_heatmap_layer)
        if layer_idx < 0:
            layer_idx = attn_map.shape[0] + layer_idx
        layer_idx = max(0, min(layer_idx, attn_map.shape[0] - 1))
        return attn_map[layer_idx]

    def _collect_attn_episode(
        attn_seq: list[np.ndarray | None],
        done_index: int,
        forward_seq: list[int | None] | None = None,
        mode_seq: list[str | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str] | None] | tuple[None, None, None, None]:
        maps = []
        frame_indices = []
        forward_indices: list[int] = []
        modes: list[str] = []
        for frame_idx, attn_map in enumerate(attn_seq[: done_index + 1]):
            if attn_map is None:
                continue
            maps.append(attn_map)
            frame_indices.append(frame_idx)
            if forward_seq is not None:
                forward_idx = forward_seq[frame_idx] if frame_idx < len(forward_seq) else None
                forward_indices.append(-1 if forward_idx is None else int(forward_idx))
            if mode_seq is not None:
                mode_val = mode_seq[frame_idx] if frame_idx < len(mode_seq) else None
                modes.append("unknown" if mode_val is None else str(mode_val))
        if not maps:
            return None, None, None, None
        forward_arr = np.asarray(forward_indices, dtype=np.int32) if forward_seq is not None else None
        mode_list = modes if mode_seq is not None else None
        return np.stack(maps, axis=0), np.asarray(frame_indices, dtype=np.int32), forward_arr, mode_list

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        nonlocal missing_attn_logged, attn_camera_used, reuse_camera_used, last_forward_counter
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))
        if render_reuse_mask:
            mask, _ = _select_reuse_mask()
            if mask is None:
                ep_masks.append(None)
            else:
                mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                ep_masks.append(mask_np[:n_to_render_now])
        if needs_attn:
            frame_idx = len(ep_frames) - 1
            forward_counter = _get_forward_counter()
            if forward_counter is None:
                forward_counter = frame_idx
            forward_advanced = (
                last_forward_counter is None or forward_counter != last_forward_counter
            )
            if forward_advanced:
                last_forward_counter = forward_counter
            should_capture = forward_advanced and (forward_counter % attn_save_interval == 0)
            attn_snapshot = None
            attn_missing_reason = None
            attn_cam_key = None
            if should_capture:
                attn_maps, cam_key = _select_attn_maps()
                attn_cam_key = cam_key
                if attn_maps is None:
                    attn_missing_reason = "no_attn_maps"
                    if forward_counter > 0 and not missing_attn_logged:
                        logging.info(
                            "Attention map capture requested, but no attn maps were produced. "
                            "Enable policy.record_attn and set policy.attn_reduce."
                        )
                        missing_attn_logged = True
                else:
                    if cam_key is not None and attn_camera_used is None:
                        attn_camera_used = cam_key
                    layer_maps = []
                    for layer_idx, layer_map in enumerate(attn_maps):
                        if layer_map is None:
                            attn_missing_reason = f"layer_{layer_idx}_none"
                            layer_maps = None
                            break
                        if torch.is_tensor(layer_map):
                            layer_map = layer_map.detach().to(dtype=torch.float32).cpu().numpy()
                        else:
                            layer_map = np.asarray(layer_map, dtype=np.float32)
                        layer_maps.append(layer_map)
                    if layer_maps:
                        try:
                            stacked = np.stack(layer_maps, axis=1)
                        except ValueError as exc:
                            attn_missing_reason = f"stack_error:{exc}"
                            stacked = None
                        else:
                            attn_snapshot = stacked[:n_to_render_now]
                    elif layer_maps == []:
                        attn_missing_reason = "empty_layer_maps"

            mode_list = None
            if should_capture:
                update_mask, _ = _select_update_mask_for_attn()
                mode_list = _infer_update_modes(update_mask, n_to_render_now)
            analysis_snapshot = None
            if should_capture:
                reuse_analysis, reuse_cam_key = _select_reuse_analysis()
                if reuse_analysis is not None:
                    if reuse_cam_key is not None and reuse_camera_used is None:
                        reuse_camera_used = reuse_cam_key
                    analysis_snapshot = [
                        _slice_reuse_analysis(reuse_analysis, idx) for idx in range(n_to_render_now)
                    ]
            if should_capture and attn_snapshot is not None:
                ep_attn_samples.append(attn_snapshot)
                ep_attn_forward_indices.append(forward_counter)
                ep_attn_modes.append(mode_list)
            elif should_capture and attn_snapshot is None and forward_counter > 0:
                logging.info(
                    "Attention snapshot missing: forward=%s reason=%s record_attn=%s attn_reduce=%s camera=%s",
                    forward_counter,
                    attn_missing_reason or "unknown",
                    getattr(policy.config, "record_attn", None),
                    getattr(policy.config, "attn_reduce", None),
                    attn_cam_key or attn_camera,
                )
            else:
                ep_attn_samples.append(None)
                ep_attn_forward_indices.append(None)
                ep_attn_modes.append(None)
            if should_capture and analysis_snapshot is not None:
                ep_reuse_analysis_samples.append(analysis_snapshot)
                ep_reuse_forward_indices.append(forward_counter)
                ep_reuse_modes.append(mode_list)
            else:
                ep_reuse_analysis_samples.append(None)
                ep_reuse_forward_indices.append(None)
                ep_reuse_modes.append(None)

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
            if render_reuse_mask:
                ep_masks: list[np.ndarray | None] = []
            ep_attn_samples = None
            ep_attn_forward_indices = None
            ep_attn_modes = None
            ep_reuse_analysis_samples = None
            ep_reuse_forward_indices = None
            ep_reuse_modes = None
            if needs_attn:
                ep_attn_samples = []
                ep_attn_forward_indices = []
                ep_attn_modes = []
                ep_reuse_analysis_samples = []
                ep_reuse_forward_indices = []
                ep_reuse_modes = []

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
            batch_stacked_masks = None
            if render_reuse_mask and ep_masks and all(mask is not None for mask in ep_masks):
                batch_stacked_masks = np.stack(ep_masks, axis=1)
            batch_attn_samples = ep_attn_samples if ep_attn_samples else None
            batch_attn_forward_indices = ep_attn_forward_indices if ep_attn_forward_indices else None
            batch_attn_modes = ep_attn_modes if ep_attn_modes else None
            for ep_idx, (stacked_frames, done_index) in enumerate(
                zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False)
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                render_frames = stacked_frames[: done_index + 1]  # + 1 to capture the last observation
                if batch_stacked_masks is not None:
                    mask_seq = batch_stacked_masks[ep_idx][: done_index + 1]
                    render_frames = _overlay_reuse_mask(render_frames, mask_seq)
                attn_stack = None
                frame_indices = None
                forward_indices = None
                mode_list = None
                if needs_attn and attn_dir is not None and batch_attn_samples is not None:
                    attn_save_seq = [
                        step[ep_idx] if step is not None else None
                        for step in batch_attn_samples[: done_index + 1]
                    ]
                    forward_seq = (
                        [
                            step if step is not None else None
                            for step in batch_attn_forward_indices[: done_index + 1]
                        ]
                        if batch_attn_forward_indices is not None
                        else None
                    )
                    mode_seq = (
                        [
                            step[ep_idx] if step is not None else None
                            for step in batch_attn_modes[: done_index + 1]
                        ]
                        if batch_attn_modes is not None
                        else None
                    )
                    attn_stack, frame_indices, forward_indices, mode_list = _collect_attn_episode(
                        attn_save_seq,
                        done_index,
                        forward_seq=forward_seq,
                        mode_seq=mode_seq,
                    )
                analysis_seq = None
                analysis_forward_seq = None
                analysis_mode_seq = None
                if needs_attn and attn_dir is not None and batch_attn_samples is not None:
                    analysis_seq = [
                        step[ep_idx] if step is not None else None
                        for step in ep_reuse_analysis_samples[: done_index + 1]
                    ]
                    analysis_forward_seq = (
                        [
                            step if step is not None else None
                            for step in ep_reuse_forward_indices[: done_index + 1]
                        ]
                        if ep_reuse_forward_indices is not None
                        else None
                    )
                    analysis_mode_seq = (
                        [
                            step[ep_idx] if step is not None else None
                            for step in ep_reuse_modes[: done_index + 1]
                        ]
                        if ep_reuse_modes is not None
                        else None
                    )
                if save_attn_maps and attn_dir is not None and attn_stack is not None:
                    attn_dir.mkdir(parents=True, exist_ok=True)
                    attn_path = attn_dir / f"attn_episode_{n_episodes_rendered}.pth"
                    attn_meta = {
                        "attn_maps": torch.from_numpy(attn_stack),
                        "frame_indices": torch.from_numpy(frame_indices),
                        "forward_indices": torch.from_numpy(forward_indices) if forward_indices is not None else None,
                        "update_modes": mode_list,
                        "camera": attn_camera_used or attn_camera,
                        "attn_reduce": getattr(policy.config, "attn_reduce", None),
                        "episode_index": n_episodes_rendered,
                    }
                    torch.save(attn_meta, attn_path)
                if render_attn_heatmap and attn_dir is not None and attn_stack is not None:
                    cam_tag = attn_camera_used or attn_camera or "camera"
                    base_heatmap_dir = attn_dir / "heatmaps" / str(cam_tag) / f"episode_{n_episodes_rendered}"
                    base_heatmap_dir.mkdir(parents=True, exist_ok=True)
                    save_all_layers = isinstance(attn_heatmap_layer, str) and attn_heatmap_layer == "all"
                    for idx, (attn_map, frame_idx) in enumerate(zip(attn_stack, frame_indices, strict=False)):
                        if attn_map.ndim == 2:
                            layer_items = [(0, attn_map)]
                        elif attn_map.ndim == 3:
                            if save_all_layers:
                                layer_items = list(enumerate(attn_map))
                            else:
                                if isinstance(attn_heatmap_layer, str) and attn_heatmap_layer == "mean":
                                    layer_items = [("mean", attn_map.mean(axis=0))]
                                else:
                                    layer_idx = int(attn_heatmap_layer)
                                    if layer_idx < 0:
                                        layer_idx = attn_map.shape[0] + layer_idx
                                    layer_idx = max(0, min(layer_idx, attn_map.shape[0] - 1))
                                    layer_items = [(layer_idx, attn_map[layer_idx])]
                        else:
                            continue

                        forward_idx = int(forward_indices[idx]) if forward_indices is not None else idx
                        if forward_idx < 0:
                            forward_idx = idx
                        mode_tag = "unknown"
                        if mode_list is not None and idx < len(mode_list):
                            mode_tag = str(mode_list[idx])
                        forward_dir = base_heatmap_dir / f"forward_{forward_idx:04d}"
                        forward_dir.mkdir(parents=True, exist_ok=True)
                        if save_attn_maps and attn_dir is not None:
                            forward_pth = forward_dir / f"attn_forward_{forward_idx:04d}_mode_{mode_tag}.pth"
                            if not forward_pth.exists():
                                attn_forward_meta = {
                                    "attn_maps": torch.from_numpy(attn_map),
                                    "frame_index": int(frame_idx),
                                    "forward_index": int(forward_idx),
                                    "update_mode": mode_tag,
                                    "camera": attn_camera_used or attn_camera,
                                    "attn_reduce": getattr(policy.config, "attn_reduce", None),
                                    "episode_index": n_episodes_rendered,
                                }
                                torch.save(attn_forward_meta, forward_pth)
                        for layer_id, layer_map in layer_items:
                            layer_map = np.nan_to_num(
                                layer_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                            )
                            min_val = float(layer_map.min())
                            max_val = float(layer_map.max())
                            if max_val > min_val:
                                layer_map = (layer_map - min_val) / (max_val - min_val)
                            else:
                                layer_map = np.zeros_like(layer_map, dtype=np.float32)
                            layer_dir = forward_dir / f"layer_{layer_id}"
                            layer_dir.mkdir(parents=True, exist_ok=True)
                            heatmap_path = layer_dir / f"heatmap_frame_{int(frame_idx):04d}_mode_{mode_tag}.png"
                            if layer_map.ndim == 2 and layer_map.shape[0] >= 128 and layer_map.shape[1] >= 128:
                                heatmap_img = _render_matrix_heatmap(
                                    layer_map,
                                    attn_heatmap_alpha,
                                )
                            else:
                                heatmap_img = _render_heatmap_image(
                                    layer_map,
                                    render_frames[frame_idx].shape[:2],
                                    attn_heatmap_alpha,
                                )
                            write_image(heatmap_img, heatmap_path)
                if attn_dir is not None and analysis_seq is not None:
                    analysis_cam_tag = reuse_camera_used or attn_camera_used or attn_camera or "camera"
                    analysis_base_dir = (
                        attn_dir / "reuse_analysis" / str(analysis_cam_tag) / f"episode_{n_episodes_rendered}"
                    )
                    analysis_base_dir.mkdir(parents=True, exist_ok=True)
                    for idx, analysis in enumerate(analysis_seq):
                        if analysis is None:
                            continue
                        forward_idx = (
                            int(analysis_forward_seq[idx])
                            if analysis_forward_seq is not None and analysis_forward_seq[idx] is not None
                            else idx
                        )
                        mode_tag = "unknown"
                        if analysis_mode_seq is not None and idx < len(analysis_mode_seq):
                            mode_val = analysis_mode_seq[idx]
                            if mode_val is not None:
                                mode_tag = str(mode_val)
                        forward_dir = analysis_base_dir / f"forward_{forward_idx:04d}"
                        forward_dir.mkdir(parents=True, exist_ok=True)
                        metrics_path = forward_dir / f"reuse_metrics_mode_{mode_tag}.json"
                        metrics = analysis.get("metrics", {})
                        with open(metrics_path, "w", encoding="utf-8") as metrics_fp:
                            json.dump(metrics, metrics_fp, ensure_ascii=False, indent=2)

                        patch_diff_grid = analysis.get("patch_diff_grid")
                        if patch_diff_grid is not None:
                            patch_map = _normalize_map(np.asarray(patch_diff_grid))
                            patch_path = forward_dir / f"patch_diff_mode_{mode_tag}.png"
                            heatmap_img = _render_heatmap_image(
                                patch_map,
                                patch_map.shape,
                                attn_heatmap_alpha,
                            )
                            write_image(heatmap_img, patch_path)

                        layer_token_grids = analysis.get("layer_token_diff_grids", [])
                        layer_channel_diffs = analysis.get("layer_channel_diffs", [])
                        attn_diff_maps = analysis.get("attn_diff_maps", [])
                        for layer_idx in range(max(len(layer_token_grids), len(layer_channel_diffs), len(attn_diff_maps))):
                            layer_dir = forward_dir / f"layer_{layer_idx}"
                            layer_dir.mkdir(parents=True, exist_ok=True)
                            if layer_idx < len(layer_token_grids):
                                token_grid = layer_token_grids[layer_idx]
                                if token_grid is not None:
                                    token_map = _normalize_map(np.asarray(token_grid))
                                    token_path = layer_dir / f"token_diff_mode_{mode_tag}.png"
                                    heatmap_img = _render_heatmap_image(
                                        token_map,
                                        token_map.shape,
                                        attn_heatmap_alpha,
                                    )
                                    write_image(heatmap_img, token_path)
                            if layer_idx < len(layer_channel_diffs):
                                channel_diff = layer_channel_diffs[layer_idx]
                                if channel_diff is not None:
                                    channel_map = _normalize_map(np.asarray(channel_diff)[None, :])
                                    channel_path = layer_dir / f"channel_diff_mode_{mode_tag}.png"
                                    heatmap_img = _render_matrix_heatmap(
                                        channel_map,
                                        attn_heatmap_alpha,
                                    )
                                    write_image(heatmap_img, channel_path)
                            if layer_idx < len(attn_diff_maps):
                                attn_diff = attn_diff_maps[layer_idx]
                                if attn_diff is not None:
                                    attn_map = _normalize_map(np.asarray(attn_diff))
                                    attn_path = layer_dir / f"attn_diff_mode_{mode_tag}.png"
                                    heatmap_img = _render_matrix_heatmap(
                                        attn_map,
                                        attn_heatmap_alpha,
                                    )
                                    write_image(heatmap_img, attn_path)
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        render_frames,
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
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
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
    logging.info("Effective policy config: %s", policy.config)
    logging.info("Effective eval config: %s", asdict(cfg.eval))

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
            render_reuse_mask=cfg.eval.render_reuse_mask,
            render_reuse_camera=cfg.eval.render_reuse_camera,
            save_attn_maps=cfg.eval.save_attn_maps,
            attn_save_interval=cfg.eval.attn_save_interval,
            attn_camera=cfg.eval.attn_camera,
            render_attn_heatmap=cfg.eval.render_attn_heatmap,
            attn_heatmap_layer=cfg.eval.attn_heatmap_layer,
            attn_heatmap_alpha=cfg.eval.attn_heatmap_alpha,
            attn_dir=Path(cfg.output_dir) / "attn",
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
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


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
    render_reuse_mask: bool,
    render_reuse_camera: str | None,
    save_attn_maps: bool,
    attn_save_interval: int,
    attn_camera: str | None,
    render_attn_heatmap: bool,
    attn_heatmap_layer: int | str,
    attn_heatmap_alpha: float,
    attn_dir: Path | None,
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
        render_reuse_mask=render_reuse_mask,
        render_reuse_camera=render_reuse_camera,
        save_attn_maps=save_attn_maps,
        attn_save_interval=attn_save_interval,
        attn_camera=attn_camera,
        render_attn_heatmap=render_attn_heatmap,
        attn_heatmap_layer=attn_heatmap_layer,
        attn_heatmap_alpha=attn_heatmap_alpha,
        attn_dir=attn_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


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
    render_reuse_mask: bool,
    render_reuse_camera: str | None,
    save_attn_maps: bool,
    attn_save_interval: int,
    attn_camera: str | None,
    render_attn_heatmap: bool,
    attn_heatmap_layer: int | str,
    attn_heatmap_alpha: float,
    attn_dir: Path | None,
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
    task_attn_dir = None
    if attn_dir is not None:
        task_attn_dir = attn_dir / f"{task_group}_{task_id}"
        task_attn_dir.mkdir(parents=True, exist_ok=True)

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
        render_reuse_mask=render_reuse_mask,
        render_reuse_camera=render_reuse_camera,
        save_attn_maps=save_attn_maps,
        attn_save_interval=attn_save_interval,
        attn_camera=attn_camera,
        render_attn_heatmap=render_attn_heatmap,
        attn_heatmap_layer=attn_heatmap_layer,
        attn_heatmap_alpha=attn_heatmap_alpha,
        attn_dir=task_attn_dir,
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
    render_reuse_mask: bool = False,
    render_reuse_camera: str | None = None,
    save_attn_maps: bool = False,
    attn_save_interval: int = 1,
    attn_camera: str | None = None,
    render_attn_heatmap: bool = False,
    attn_heatmap_layer: int | str = -1,
    attn_heatmap_alpha: float = 0.5,
    attn_dir: Path | None = None,
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
        render_reuse_mask=render_reuse_mask,
        render_reuse_camera=render_reuse_camera,
        save_attn_maps=save_attn_maps,
        attn_save_interval=attn_save_interval,
        attn_camera=attn_camera,
        render_attn_heatmap=render_attn_heatmap,
        attn_heatmap_layer=attn_heatmap_layer,
        attn_heatmap_alpha=attn_heatmap_alpha,
        attn_dir=attn_dir,
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

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

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
    # Route transformers logs to the root logger (console + file), without double handlers.
    hf_logging.enable_propagation()
    hf_logging.disable_default_handler()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
