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
    timing_calls_seen = -1
    infer_time_sum = 0.0
    infer_time_count = 0

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
        use_grad = False
        if hasattr(policy, "requires_grad_for_action"):
            use_grad = bool(policy.requires_grad_for_action())
        if use_grad:
            action = policy.select_action(observation)
        else:
            with torch.inference_mode():
                action = policy.select_action(observation)
        timing = None
        if hasattr(policy, "get_last_timing"):
            timing = policy.get_last_timing()
        if timing and "call_idx" in timing and "total_s" in timing:
            call_idx = int(timing["call_idx"])
            if call_idx != timing_calls_seen:
                timing_calls_seen = call_idx
                infer_time_sum += float(timing["total_s"])
                infer_time_count += 1
        action = postprocessor(action)

        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.detach().to("cpu").numpy()
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
        postfix = {"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"}
        if infer_time_count > 0:
            avg_infer_ms = (infer_time_sum / infer_time_count) * 1000.0
            postfix["avg_infer_ms"] = f"{avg_infer_ms:.1f}"
        progbar.set_postfix(postfix)
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


def _mask_to_numpy(mask: torch.Tensor | np.ndarray | None, batch_idx: int) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask_arr = mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2:
        if batch_idx >= mask_arr.shape[0]:
            return None
        return mask_arr[batch_idx].astype(bool)
    if mask_arr.ndim == 1:
        return mask_arr.astype(bool)
    return None


def _array_to_numpy(arr: torch.Tensor | np.ndarray | None, batch_idx: int) -> np.ndarray | None:
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy()
    else:
        arr_np = np.asarray(arr)
    if arr_np.ndim == 2:
        if batch_idx >= arr_np.shape[0]:
            return None
        return arr_np[batch_idx]
    if arr_np.ndim == 1:
        return arr_np
    return None


def _overlay_tokens_on_frame(
    frame: np.ndarray,
    overlay: dict[str, object],
    batch_idx: int,
    alpha: float,
    show_scores: bool,
    score_precision: int,
    score_normalize: str,
    score_scale: float,
) -> np.ndarray:
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return frame
    grid_h = int(overlay.get("grid_h", 0))
    grid_w = int(overlay.get("grid_w", 0))
    if grid_h <= 0 or grid_w <= 0:
        return frame

    has_cls = bool(overlay.get("has_cls", False))
    background = _mask_to_numpy(overlay.get("background_mask"), batch_idx)
    important = _mask_to_numpy(overlay.get("important_mask"), batch_idx)
    keep_mask = _mask_to_numpy(overlay.get("keep_mask"), batch_idx)
    clipped_mask = _mask_to_numpy(overlay.get("clipped_mask"), batch_idx)
    token_mask = _mask_to_numpy(overlay.get("token_mask"), batch_idx)
    region_scores = _array_to_numpy(overlay.get("region_scores"), batch_idx)
    region_patch_size = int(overlay.get("region_patch_size", 1))

    num_tokens = grid_h * grid_w
    if token_mask is None:
        token_mask = np.ones(num_tokens + (1 if has_cls else 0), dtype=bool)

    if has_cls:
        if token_mask.size > 0:
            token_mask = token_mask[1:]
        if background is not None and background.size > 0:
            background = background[1:]
        if important is not None and important.size > 0:
            important = important[1:]
        if keep_mask is not None and keep_mask.size > 0:
            keep_mask = keep_mask[1:]
        if clipped_mask is not None and clipped_mask.size > 0:
            clipped_mask = clipped_mask[1:]

    if token_mask.size != num_tokens:
        return frame
    if background is not None and background.size != num_tokens:
        return frame
    if important is not None and important.size != num_tokens:
        return frame
    if keep_mask is not None and keep_mask.size != num_tokens:
        return frame
    if clipped_mask is not None and clipped_mask.size != num_tokens:
        return frame

    if background is None:
        background = np.zeros(num_tokens, dtype=bool)
    if important is None:
        important = np.zeros(num_tokens, dtype=bool)

    valid = token_mask.astype(bool)
    important = important.astype(bool) & valid
    if keep_mask is not None:
        kept = keep_mask.astype(bool) & valid
        clipped = clipped_mask.astype(bool) & valid if clipped_mask is not None else np.zeros_like(valid)
        prunable = valid & (~kept) & (~clipped)
        other = kept & (~important)
    else:
        background = background.astype(bool) & valid
        prunable = background & (~important)
        other = valid & (~important) & (~background)

    important_grid = important.reshape(grid_h, grid_w)
    prunable_grid = prunable.reshape(grid_h, grid_w)
    other_grid = other.reshape(grid_h, grid_w)
    clipped_grid = clipped.reshape(grid_h, grid_w) if keep_mask is not None else None

    height, width = frame.shape[:2]
    ys = np.linspace(0, height, grid_h + 1, dtype=int)
    xs = np.linspace(0, width, grid_w + 1, dtype=int)
    overlay_img = np.zeros((height, width, 3), dtype=np.float32)
    overlay_mask = np.zeros((height, width), dtype=np.float32)

    for i in range(grid_h):
        y0, y1 = ys[i], ys[i + 1]
        for j in range(grid_w):
            if not (
                important_grid[i, j]
                or prunable_grid[i, j]
                or other_grid[i, j]
                or (clipped_grid is not None and clipped_grid[i, j])
            ):
                continue
            x0, x1 = xs[j], xs[j + 1]
            if clipped_grid is not None and clipped_grid[i, j]:
                color = (0.0, 0.0, 255.0)
            elif important_grid[i, j]:
                color = (255.0, 0.0, 0.0)
            elif prunable_grid[i, j]:
                color = (0.0, 255.0, 0.0)
            else:
                color = (255.0, 255.0, 0.0)
            overlay_img[y0:y1, x0:x1] = color
            overlay_mask[y0:y1, x0:x1] = 1.0

    if alpha <= 0.0 or not overlay_mask.any():
        return frame

    frame_f = frame.astype(np.float32)
    alpha_mask = (overlay_mask * alpha)[:, :, None]
    blended = frame_f * (1.0 - alpha_mask) + overlay_img * alpha_mask
    blended = blended.astype(frame.dtype)

    if show_scores and region_scores is not None and region_patch_size > 0:
        scores = region_scores.astype(np.float32, copy=True)
        if score_normalize == "max":
            denom = float(np.max(scores))
            if denom > 0:
                scores = scores / denom
        elif score_normalize == "sum":
            denom = float(np.sum(scores))
            if denom > 0:
                scores = scores / denom
        try:
            import cv2  # type: ignore
        except Exception:
            return blended

        num_regions_h = (grid_h + region_patch_size - 1) // region_patch_size
        num_regions_w = (grid_w + region_patch_size - 1) // region_patch_size
        if scores.size != num_regions_h * num_regions_w:
            return blended

        valid_grid = valid.reshape(grid_h, grid_w)
        height, width = blended.shape[:2]
        ys = np.linspace(0, height, grid_h + 1, dtype=int)
        xs = np.linspace(0, width, grid_w + 1, dtype=int)
        region_px = min(height / grid_h, width / grid_w) * region_patch_size
        font_scale = max(0.2, min(0.8, region_px / 110.0)) * score_scale
        thickness = 1

        for rh in range(num_regions_h):
            r0 = rh * region_patch_size
            r1 = min((rh + 1) * region_patch_size, grid_h)
            for rw in range(num_regions_w):
                c0 = rw * region_patch_size
                c1 = min((rw + 1) * region_patch_size, grid_w)
                if not valid_grid[r0:r1, c0:c1].any():
                    continue
                region_idx = rh * num_regions_w + rw
                score_val = float(scores[region_idx])
                text = f"{score_val:.{score_precision}f}"
                y0, y1 = ys[r0], ys[r1]
                x0, x1 = xs[c0], xs[c1]
                cx = int((x0 + x1) / 2)
                cy = int((y0 + y1) / 2)
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = max(0, min(width - text_size[0], cx - text_size[0] // 2))
                ty = max(text_size[1], min(height - 1, cy + text_size[1] // 2))
                cv2.putText(blended, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                cv2.putText(blended, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return blended


def _apply_overlay_to_frames(
    frames: list[np.ndarray],
    policy: PreTrainedPolicy,
    image_index: int,
    alpha: float,
    show_scores: bool,
    score_precision: int,
    score_normalize: str,
    score_scale: float,
) -> list[np.ndarray]:
    getter = getattr(policy, "get_last_token_overlay", None)
    if getter is None:
        return frames
    overlay = getter(image_index=image_index)
    if overlay is None:
        return frames
    return [
        _overlay_tokens_on_frame(
            frame,
            overlay,
            batch_idx,
            alpha,
            show_scores,
            score_precision,
            score_normalize,
            score_scale,
        )
        for batch_idx, frame in enumerate(frames)
    ]


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
    overlay_token_masks: bool = False,
    overlay_alpha: float = 0.35,
    overlay_image_index: int = 0,
    overlay_show_scores: bool = False,
    overlay_score_precision: int = 2,
    overlay_score_normalize: str = "none",
    overlay_score_scale: float = 1.0,
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
        overlay_token_masks: Whether to overlay token masks on rendered videos.
        overlay_alpha: Alpha value for overlay blending.
        overlay_image_index: Which image index to visualize for overlay.
        overlay_show_scores: Whether to overlay region scores on rendered videos.
        overlay_score_precision: Number of decimals to show for region scores.
        overlay_score_normalize: Normalization mode for region scores ("none", "max", "sum").
        overlay_score_scale: Scale factor for overlay score text size.
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
    # Callback for rendering.

    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            frames = [env.envs[i].render() for i in range(n_to_render_now)]  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            frames = env.call("render")[:n_to_render_now]
        else:
            frames = []

        if overlay_token_masks and frames:
            frames = _apply_overlay_to_frames(
                frames,
                policy,
                overlay_image_index,
                overlay_alpha,
                overlay_show_scores,
                overlay_score_precision,
                overlay_score_normalize,
                overlay_score_scale,
            )

        if frames:
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
            for ep_idx, (stacked_frames, done_index) in enumerate(
                zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False)
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                render_frames = stacked_frames[: done_index + 1]  # + 1 to capture the last observation
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

        postfix = {"sr": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        if hasattr(policy, "get_timing_summary"):
            summary = policy.get_timing_summary()
        else:
            summary = None
        if summary and isinstance(summary, dict):
            avg = summary.get("avg_s", {})
            avg_eval = summary.get("avg_eval_s", {})
            avg_noeval = summary.get("avg_noeval_s", {})

            total_ms = float(avg.get("total_s", 0.0)) * 1000.0 if isinstance(avg, dict) else 0.0
            vision_ms = float(avg.get("vision_encode_s", 0.0)) * 1000.0 if isinstance(avg, dict) else 0.0
            diff_ms = float(avg.get("diffusion_denoise_s", 0.0)) * 1000.0 if isinstance(avg, dict) else 0.0
            bg_ms = float(avg.get("background_s", 0.0)) * 1000.0 if isinstance(avg, dict) else 0.0
            region_ms = (
                float(avg_eval.get("region_eval_s", 0.0)) * 1000.0 if isinstance(avg_eval, dict) else 0.0
            )
            mask_ms = (
                float(avg_noeval.get("prune_mask_s", 0.0)) * 1000.0 if isinstance(avg_noeval, dict) else 0.0
            )
            prune_ms = (
                float(avg_noeval.get("prune_pack_s", 0.0)) * 1000.0 if isinstance(avg_noeval, dict) else 0.0
            )
            eval_ms = float(avg_eval.get("total_s", 0.0)) * 1000.0 if isinstance(avg_eval, dict) else 0.0
            noeval_ms = float(avg_noeval.get("total_s", 0.0)) * 1000.0 if isinstance(avg_noeval, dict) else 0.0
            llm_pruned_ms = float(summary.get("avg_llm_pruned_s", 0.0)) * 1000.0
            llm_unpruned_ms = float(summary.get("avg_llm_unpruned_s", 0.0)) * 1000.0
            pruned_tokens = float(summary.get("avg_pruned_tokens", 0.0))
            pruned_ratio = float(summary.get("avg_pruned_ratio", 0.0)) * 100.0

            postfix.update(
                {
                    "avg_t": f"{total_ms:.1f}",
                    "eval_t": f"{eval_ms:.1f}",
                    "noeval_t": f"{noeval_ms:.1f}",
                    "vis_t": f"{vision_ms:.1f}",
                    "diff_t": f"{diff_ms:.1f}",
                    "bg_t": f"{bg_ms:.1f}",
                    "reg_t": f"{region_ms:.1f}",
                    "mask_t": f"{mask_ms:.1f}",
                    "prn_t": f"{prune_ms:.1f}",
                    "p_llm": f"{llm_pruned_ms:.1f}",
                    "u_llm": f"{llm_unpruned_ms:.1f}",
                    "prn_tok": f"{pruned_tokens:.1f}",
                    "prn_ratio": f"{pruned_ratio:.1f}%",
                }
            )

            token_counts = summary.get("token_avg_counts", summary.get("token_counts", {}))
            if isinstance(token_counts, dict):
                total_tokens = float(token_counts.get("total", 0.0))
                bg_tokens = float(token_counts.get("prunable_bg", 0.0))
                region_tokens = float(token_counts.get("important", 0.0))
                if total_tokens > 0:
                    bg_ratio = bg_tokens / total_tokens * 100.0
                    region_ratio = region_tokens / total_tokens * 100.0
                    postfix["bg_ratio"] = f"{bg_tokens:.1f}/{bg_ratio:.1f}%"
                    postfix["reg_ratio"] = f"{region_tokens:.1f}/{region_ratio:.1f}%"
                else:
                    postfix["bg_ratio"] = "0/0.0%"
                    postfix["reg_ratio"] = "0/0.0%"
        progbar.set_postfix(postfix)

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
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=log_dir / "eval.log")

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

    use_grad = False
    if hasattr(policy, "requires_grad_for_action"):
        use_grad = bool(policy.requires_grad_for_action())
    grad_ctx = nullcontext() if use_grad else torch.no_grad()
    amp_ctx = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    with grad_ctx, amp_ctx:
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
            overlay_token_masks=cfg.eval.overlay_token_masks,
            overlay_alpha=cfg.eval.overlay_alpha,
            overlay_image_index=cfg.eval.overlay_image_index,
            overlay_show_scores=cfg.eval.overlay_show_scores,
            overlay_score_precision=cfg.eval.overlay_score_precision,
            overlay_score_normalize=cfg.eval.overlay_score_normalize,
            overlay_score_scale=cfg.eval.overlay_score_scale,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save compact info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(build_eval_summary(info), f, indent=2)

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
    return_episode_data: bool,
    start_seed: int | None,
    overlay_token_masks: bool,
    overlay_alpha: float,
    overlay_image_index: int,
    overlay_show_scores: bool,
    overlay_score_precision: int,
    overlay_score_normalize: str,
    overlay_score_scale: float,
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
        overlay_token_masks=overlay_token_masks,
        overlay_alpha=overlay_alpha,
        overlay_image_index=overlay_image_index,
        overlay_show_scores=overlay_show_scores,
        overlay_score_precision=overlay_score_precision,
        overlay_score_normalize=overlay_score_normalize,
        overlay_score_scale=overlay_score_scale,
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
    return_episode_data: bool,
    start_seed: int | None,
    overlay_token_masks: bool,
    overlay_alpha: float,
    overlay_image_index: int,
    overlay_show_scores: bool,
    overlay_score_precision: int,
    overlay_score_normalize: str,
    overlay_score_scale: float,
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
    if hasattr(policy, "reset_timing_stats"):
        policy.reset_timing_stats()
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
        overlay_token_masks=overlay_token_masks,
        overlay_alpha=overlay_alpha,
        overlay_image_index=overlay_image_index,
        overlay_show_scores=overlay_show_scores,
        overlay_score_precision=overlay_score_precision,
        overlay_score_normalize=overlay_score_normalize,
        overlay_score_scale=overlay_score_scale,
    )
    _log_timing_summary(policy, task_group=task_group, task_id=task_id)
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
    overlay_token_masks: bool = False,
    overlay_alpha: float = 0.35,
    overlay_image_index: int = 0,
    overlay_show_scores: bool = False,
    overlay_score_precision: int = 2,
    overlay_score_normalize: str = "none",
    overlay_score_scale: float = 1.0,
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
            overlay_token_masks=overlay_token_masks,
            overlay_alpha=overlay_alpha,
            overlay_image_index=overlay_image_index,
            overlay_show_scores=overlay_show_scores,
            overlay_score_precision=overlay_score_precision,
            overlay_score_normalize=overlay_score_normalize,
            overlay_score_scale=overlay_score_scale,
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


def _success_rate(successes: list[bool]) -> float:
    if not successes:
        return float("nan")
    arr = np.array(successes, dtype=float)
    return float(np.nanmean(arr) * 100)


def build_eval_summary(info: dict) -> dict:
    per_task: list[dict[str, object]] = []
    group_successes: dict[str, list[bool]] = defaultdict(list)
    all_successes: list[bool] = []

    for item in info.get("per_task", []):
        group = item.get("task_group")
        task_id = item.get("task_id")
        metrics = item.get("metrics", {})
        successes = metrics.get("successes", [])
        if successes is None:
            successes_list: list[bool] = []
        elif isinstance(successes, list):
            successes_list = successes
        else:
            successes_list = [bool(successes)]

        per_task.append(
            {
                "task_group": group,
                "task_id": task_id,
                "pc_success": _success_rate(successes_list),
                "n_episodes": len(successes_list),
            }
        )
        if group is not None:
            group_successes[str(group)].extend(successes_list)
        all_successes.extend(successes_list)

    per_group = {
        group: {"pc_success": _success_rate(succ), "n_episodes": len(succ)}
        for group, succ in group_successes.items()
    }

    overall = {
        "pc_success": _success_rate(all_successes),
        "n_episodes": len(all_successes),
    }
    if "overall" in info:
        overall["eval_s"] = info["overall"].get("eval_s")
        overall["eval_ep_s"] = info["overall"].get("eval_ep_s")

    return {"per_task": per_task, "per_group": per_group, "overall": overall}


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _log_timing_summary(policy: PreTrainedPolicy, task_group: str, task_id: int) -> None:
    if not hasattr(policy, "get_timing_summary"):
        return
    summary = policy.get_timing_summary()
    if not summary:
        return
    num_calls = int(summary.get("num_calls", 0))
    num_eval_calls = int(summary.get("num_eval_calls", 0))
    num_noeval_calls = int(summary.get("num_noeval_calls", 0))
    avg = summary.get("avg_s", {})
    if num_calls <= 0 or not avg:
        return

    total_s = float(avg.get("total_s", 0.0))
    if total_s <= 0:
        return

    def _ms(value: float) -> float:
        return value * 1000.0

    avg_eval = summary.get("avg_eval_s", {})
    avg_noeval = summary.get("avg_noeval_s", {})
    token_counts = summary.get("token_avg_counts", summary.get("token_counts", {}))
    total_tokens = float(token_counts.get("total", 0.0)) if isinstance(token_counts, dict) else 0.0
    bg_tokens = float(token_counts.get("prunable_bg", 0.0)) if isinstance(token_counts, dict) else 0.0
    region_tokens = float(token_counts.get("important", 0.0)) if isinstance(token_counts, dict) else 0.0
    bg_ratio = (bg_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    region_ratio = (region_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0

    logging.info(
        "Timing metrics task_group=%s task_id=%d calls=%d avg_time=%.2fms eval_time=%.2fms "
        "noeval_time=%.2fms vision_time=%.2fms diff_time=%.2fms bg_time=%.2fms "
        "region_time=%.2fms mask_time=%.2fms prune_time=%.2fms "
        "prune_llm=%.2fms unprune_llm=%.2fms pruned_tokens=%.1f pruned_ratio=%.1f%% "
        "bg_ratio=%.1f/%.1f%% region_ratio=%.1f/%.1f%%",
        task_group,
        task_id,
        num_calls,
        _ms(float(avg.get("total_s", 0.0))),
        _ms(float(avg_eval.get("total_s", 0.0))) if isinstance(avg_eval, dict) else 0.0,
        _ms(float(avg_noeval.get("total_s", 0.0))) if isinstance(avg_noeval, dict) else 0.0,
        _ms(float(avg.get("vision_encode_s", 0.0))),
        _ms(float(avg.get("diffusion_denoise_s", 0.0))),
        _ms(float(avg.get("background_s", 0.0))),
        _ms(float(avg_eval.get("region_eval_s", 0.0))) if isinstance(avg_eval, dict) else 0.0,
        _ms(float(avg_noeval.get("prune_mask_s", 0.0))) if isinstance(avg_noeval, dict) else 0.0,
        _ms(float(avg_noeval.get("prune_pack_s", 0.0))) if isinstance(avg_noeval, dict) else 0.0,
        _ms(float(summary.get("avg_llm_pruned_s", 0.0))),
        _ms(float(summary.get("avg_llm_unpruned_s", 0.0))),
        float(summary.get("avg_pruned_tokens", 0.0)),
        float(summary.get("avg_pruned_ratio", 0.0)) * 100.0,
        bg_tokens,
        bg_ratio,
        region_tokens,
        region_ratio,
    )
    if isinstance(avg_eval, dict) and "total_s" in avg_eval:
        logging.info(
            "Timing split task_group=%s task_id=%d eval_calls=%d eval_total=%.2fms",
            task_group,
            task_id,
            num_eval_calls,
            _ms(float(avg_eval.get("total_s", 0.0))),
        )
    if isinstance(avg_noeval, dict) and "total_s" in avg_noeval:
        logging.info(
            "Timing split task_group=%s task_id=%d noeval_calls=%d noeval_total=%.2fms",
            task_group,
            task_id,
            num_noeval_calls,
            _ms(float(avg_noeval.get("total_s", 0.0))),
        )
    llm_pruned_s = summary.get("avg_llm_pruned_s")
    llm_unpruned_s = summary.get("avg_llm_unpruned_s")
    llm_pruned_calls = int(summary.get("num_llm_pruned_calls", 0))
    llm_unpruned_calls = int(summary.get("num_llm_unpruned_calls", 0))
    if isinstance(llm_pruned_s, (int, float)):
        logging.info(
            "Timing LLM split task_group=%s task_id=%d pruned_calls=%d pruned_llm=%.2fms",
            task_group,
            task_id,
            llm_pruned_calls,
            _ms(float(llm_pruned_s)),
        )
    if isinstance(llm_unpruned_s, (int, float)):
        logging.info(
            "Timing LLM split task_group=%s task_id=%d unpruned_calls=%d unpruned_llm=%.2fms",
            task_group,
            task_id,
            llm_unpruned_calls,
            _ms(float(llm_unpruned_s)),
        )

    pct = {
        "vision": float(avg.get("vision_encode_s", 0.0)) / total_s,
        "prefix": float(avg.get("prefix_build_s", 0.0)) / total_s,
        "diffusion": float(avg.get("diffusion_total_s", 0.0)) / total_s,
        "background": float(avg.get("background_s", 0.0)) / total_s,
        "region_eval": float(avg.get("region_eval_s", 0.0)) / total_s,
        "prune": float(avg.get("prune_s", 0.0)) / total_s,
        "token_sel": float(avg.get("token_selection_s", 0.0)) / total_s,
    }
    logging.info(
        "Timing pct task_group=%s task_id=%d vision=%s prefix=%s diffusion=%s background=%s "
        "region_eval=%s prune=%s token_sel=%s",
        task_group,
        task_id,
        _format_pct(pct["vision"]),
        _format_pct(pct["prefix"]),
        _format_pct(pct["diffusion"]),
        _format_pct(pct["background"]),
        _format_pct(pct["region_eval"]),
        _format_pct(pct["prune"]),
        _format_pct(pct["token_sel"]),
    )

    token_counts = summary.get("token_counts", {})
    token_ratios = summary.get("token_ratios", {})
    total_tokens = int(token_counts.get("total", 0)) if isinstance(token_counts, dict) else 0
    if total_tokens > 0 and isinstance(token_counts, dict) and isinstance(token_ratios, dict):
        logging.info(
            "Token ratio task_group=%s task_id=%d total_tokens=%d important=%s prunable_bg=%s other=%s",
            task_group,
            task_id,
            total_tokens,
            _format_pct(float(token_ratios.get("important", 0.0))),
            _format_pct(float(token_ratios.get("prunable_bg", 0.0))),
            _format_pct(float(token_ratios.get("other", 0.0))),
        )


def main():
    init_logging()
    # Route transformers logs to the root logger (console + file), without double handlers.
    hf_logging.enable_propagation()
    hf_logging.disable_default_handler()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
