#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn


def extract_translation_speed(action: Tensor) -> Tensor:
    """Return the max absolute xyz displacement used by SP-VLA scheduling."""
    if action.ndim == 1:
        xyz = action[: min(3, action.shape[0])]
        if xyz.numel() == 0:
            return action.new_zeros(())
        return xyz.abs().amax()

    xyz = action[..., : min(3, action.shape[-1])]
    if xyz.numel() == 0:
        return action.new_zeros(action.shape[:-1])
    return xyz.abs().amax(dim=-1)


def extract_z_translation(action: Tensor) -> Tensor:
    """Return the absolute z translation used by the reference SP-VLA code."""
    if action.ndim == 1:
        if action.shape[0] < 3:
            return action.new_zeros(())
        return action[2].abs()

    if action.shape[-1] < 3:
        return action.new_zeros(action.shape[:-1])
    return action[..., 2].abs()


def fit_ridge_action(
    action_history: deque[Tensor],
    reg_lambda: float,
) -> Tensor | None:
    """Fit the SP-VLA ridge generator on a short action segment."""
    if len(action_history) < 2:
        return None

    history = torch.stack(list(action_history), dim=0).float()
    last_action = history[-1].clone()
    action_dim = history.shape[-1]

    fit_dims = action_dim if action_dim <= 6 else 6
    fit_dims = max(fit_dims, 0)
    if fit_dims == 0:
        return last_action

    timesteps = torch.arange(history.shape[0], device=history.device, dtype=torch.float32)
    design = torch.stack([timesteps, torch.ones_like(timesteps)], dim=1)
    xtx = design.T @ design
    rhs = design.T @ history[:, :fit_dims]
    eye = torch.eye(xtx.shape[0], device=history.device, dtype=history.dtype)
    beta = torch.linalg.solve(xtx + reg_lambda * eye, rhs)
    x_t = torch.tensor(
        [float(history.shape[0]), 1.0],
        device=history.device,
        dtype=history.dtype,
    )
    predicted = x_t @ beta
    last_action[:fit_dims] = predicted
    if action_dim > fit_dims:
        last_action[fit_dims:] = history[-1, fit_dims:]
    return last_action


def fit_reference_ridge_action(
    action_history: deque[Tensor],
    reg_lambda: float,
    clip_min: float = -0.9,
    clip_max: float = 0.9,
) -> Tensor | None:
    """Mirror the ridge-based step-skipping generator in the reference SP-VLA repo."""
    if len(action_history) < 2:
        return None

    history = torch.stack(list(action_history), dim=0).float()
    last_action = history[-1].clone()
    fit_dims = min(3, history.shape[-1])
    if fit_dims == 0:
        return last_action

    x = torch.arange(history.shape[0], device=history.device, dtype=history.dtype).unsqueeze(-1)
    design = torch.cat([x, torch.ones_like(x)], dim=1)
    y = history[:, :fit_dims]
    xtx = design.T @ design
    rhs = design.T @ y
    beta = torch.linalg.solve(
        xtx + reg_lambda * torch.eye(xtx.shape[0], device=history.device, dtype=history.dtype),
        rhs,
    )

    next_step = torch.tensor(
        [float(history.shape[0]), 1.0],
        device=history.device,
        dtype=history.dtype,
    )
    next_action = next_step @ beta
    next_action = next_action.clamp(min=clip_min, max=clip_max)

    if history.shape[0] >= 2:
        delta_max = history[:, :fit_dims].diff(dim=0).abs().amax(dim=0)
        delta_pred = next_action - y[-1]
        next_action = y[-1] + delta_pred.clamp(min=-delta_max, max=delta_max)

    last_action[:fit_dims] = next_action
    if history.shape[-1] > fit_dims:
        last_action[fit_dims:] = history[-1, fit_dims:]
    return last_action


def passes_action_validity_check(
    action: Tensor,
    action_history: deque[Tensor],
    max_scale: float,
    floor: float = 1e-4,
) -> bool:
    """Reject non-finite or implausibly large ridge predictions."""
    if not torch.isfinite(action).all():
        return False
    if len(action_history) == 0:
        return True

    history = torch.stack(list(action_history), dim=0).float()
    hist_limit = history.abs().amax(dim=0).clamp_min(floor) * max_scale
    return bool((action.float().abs() <= hist_limit).all().item())


@dataclass
class SPVLASchedulerState:
    buffer_size: int
    queue_size: int
    action_histories: list[deque[Tensor]] = field(default_factory=list)
    source_histories: list[deque[bool]] = field(default_factory=list)
    vla_action_queues: list[deque[Tensor]] = field(default_factory=list)
    last_used_lightweight: list[bool | None] = field(default_factory=list)
    num_vla_calls: int = 0
    num_vla_actions: int = 0
    num_lightweight_actions: int = 0
    num_generator_fallbacks: int = 0
    num_discarded_vla_actions: int = 0
    num_scheduler_switches: int = 0

    def reset(self) -> None:
        self.action_histories = []
        self.source_histories = []
        self.vla_action_queues = []
        self.last_used_lightweight = []
        self.num_vla_calls = 0
        self.num_vla_actions = 0
        self.num_lightweight_actions = 0
        self.num_generator_fallbacks = 0
        self.num_discarded_vla_actions = 0
        self.num_scheduler_switches = 0

    def ensure_batch(self, batch_size: int) -> None:
        if len(self.action_histories) == batch_size:
            return

        self.action_histories = [deque(maxlen=self.buffer_size) for _ in range(batch_size)]
        self.source_histories = [deque(maxlen=self.buffer_size) for _ in range(batch_size)]
        self.vla_action_queues = [deque(maxlen=self.queue_size) for _ in range(batch_size)]
        self.last_used_lightweight = [None for _ in range(batch_size)]

    def get_last_actions(self, device: torch.device) -> Tensor | None:
        if not self.action_histories:
            return None
        if any(len(history) == 0 for history in self.action_histories):
            return None
        return torch.stack([history[-1] for history in self.action_histories], dim=0).to(device)

    def get_motion_speeds(self, device: torch.device, batch_size: int) -> Tensor:
        self.ensure_batch(batch_size)
        speeds = []
        for history in self.action_histories:
            if history:
                speeds.append(extract_translation_speed(history[-1]).reshape(()))
            else:
                speeds.append(torch.zeros((), device=device))
        return torch.stack([speed.to(device=device, dtype=torch.float32) for speed in speeds], dim=0)

    def get_z_translations(self, device: torch.device, batch_size: int) -> Tensor:
        self.ensure_batch(batch_size)
        z_values = []
        for history in self.action_histories:
            if history:
                z_values.append(extract_z_translation(history[-1]).reshape(()))
            else:
                z_values.append(torch.zeros((), device=device))
        return torch.stack([z.to(device=device, dtype=torch.float32) for z in z_values], dim=0)

    def can_use_lightweight(
        self,
        env_idx: int,
        velocity_min: float,
        velocity_max: float,
        ratio_threshold: float,
        warmup_vla_steps: int,
    ) -> bool:
        action_history = self.action_histories[env_idx]
        source_history = self.source_histories[env_idx]
        if not action_history or len(source_history) < max(1, warmup_vla_steps):
            return False

        prev_action = action_history[-1]
        xyz = prev_action[: min(3, prev_action.shape[0])].abs()
        if xyz.numel() == 0:
            return False
        within_speed_window = bool(
            (xyz >= velocity_min).all().item() and (xyz <= velocity_max).all().item()
        )
        if not within_speed_window:
            return False

        vla_ratio = sum(int(flag) for flag in source_history) / len(source_history)
        return vla_ratio > ratio_threshold

    def can_use_reference_skip(
        self,
        env_idx: int,
        z_xy_rate_skip: float,
        z_max_skip: float,
        step_skip: int,
        min_generated_ratio: float,
    ) -> bool:
        action_history = self.action_histories[env_idx]
        source_history = self.source_histories[env_idx]
        if len(action_history) < max(1, step_skip):
            return False
        if len(source_history) == 0:
            return False

        generated_ratio = sum(int(flag) for flag in source_history) / len(source_history)
        if generated_ratio < min_generated_ratio:
            return False

        for action in list(action_history)[-step_skip:]:
            if action.shape[0] < 3:
                return False
            max_xy = max(abs(float(action[0].item())), abs(float(action[1].item())), 1e-4)
            max_z = abs(float(action[2].item()))
            if not (max_z / max_xy < z_xy_rate_skip and max_z < z_max_skip):
                return False
        return True

    def replace_vla_queue(self, env_idx: int, action_chunk: Tensor) -> None:
        queue = self.vla_action_queues[env_idx]
        queue.clear()
        for action in action_chunk[: self.queue_size]:
            queue.append(action.detach())
        self.num_vla_calls += 1

    def pop_vla_action(self, env_idx: int) -> Tensor | None:
        queue = self.vla_action_queues[env_idx]
        if not queue:
            return None
        return queue.popleft()

    def record_action(self, env_idx: int, action: Tensor, from_vla: bool) -> None:
        self.action_histories[env_idx].append(action.detach())
        self.source_histories[env_idx].append(bool(from_vla))
        used_lightweight = not from_vla
        prev_mode = self.last_used_lightweight[env_idx]
        if prev_mode is not None and prev_mode != used_lightweight:
            self.num_scheduler_switches += 1
        self.last_used_lightweight[env_idx] = used_lightweight

        if from_vla:
            self.num_vla_actions += 1
        else:
            self.num_lightweight_actions += 1

    def summary(self) -> dict[str, float | int]:
        total_actions = self.num_vla_actions + self.num_lightweight_actions
        summary = {
            "vla_calls": self.num_vla_calls,
            "vla_actions": self.num_vla_actions,
            "lightweight_actions": self.num_lightweight_actions,
            "generator_fallbacks": self.num_generator_fallbacks,
            "discarded_vla_actions": self.num_discarded_vla_actions,
            "scheduler_switches": self.num_scheduler_switches,
        }
        if total_actions > 0:
            summary["lightweight_action_rate"] = round(self.num_lightweight_actions / total_actions, 4)
            summary["vla_action_rate"] = round(self.num_vla_actions / total_actions, 4)
        return summary


def compute_reference_prune_threshold(
    z_trans: Tensor,
    z_thre_prune: float,
    z_min_prune: float,
) -> Tensor:
    """Mirror the threshold formula used by the reference SP-VLA code."""
    z_trans = z_trans.float()
    threshold = torch.ones_like(z_trans)
    high_motion = z_trans >= z_thre_prune
    if high_motion.any():
        clipped_z = z_trans.clamp(max=1.0)
        threshold_high = 1.0 - (clipped_z - z_thre_prune) * (z_min_prune / (1 - z_thre_prune))
        threshold = torch.where(high_motion, threshold_high, threshold)
    return threshold.clamp(0.0, 1.0)


def _find_last_vision_attention_module(root: nn.Module) -> nn.Module | None:
    candidate = None
    for module in root.modules():
        if hasattr(module, "q_norm") and hasattr(module, "k_norm"):
            candidate = module
        elif hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            candidate = module
    return candidate


def _reshape_attention_tensor(output: Tensor, attn_module: nn.Module) -> Tensor | None:
    if output.ndim == 4:
        num_heads = getattr(attn_module, "num_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_module, "num_attention_heads", None)
        if num_heads is not None and output.shape[1] == num_heads:
            return output
        if num_heads is not None and output.shape[2] == num_heads:
            return output.permute(0, 2, 1, 3)
        return None

    if output.ndim != 3:
        return None

    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn_module, "num_attention_heads", None)
    head_dim = getattr(attn_module, "head_dim", None)
    if num_heads is None and head_dim is None:
        return None
    if num_heads is None and head_dim is not None and output.shape[-1] % head_dim == 0:
        num_heads = output.shape[-1] // head_dim
    if head_dim is None and num_heads is not None and output.shape[-1] % num_heads == 0:
        head_dim = output.shape[-1] // num_heads
    if num_heads is None or head_dim is None or output.shape[-1] != num_heads * head_dim:
        return None
    return output.view(output.shape[0], output.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)


def capture_last_vision_attention_scores(
    vision_root: nn.Module,
    forward_fn: Callable[[], Tensor],
) -> tuple[Tensor, Tensor | None]:
    """Capture last-layer vision attention scores during the normal image-embedding forward pass."""
    attn_module = _find_last_vision_attention_module(vision_root)
    if attn_module is None:
        return forward_fn(), None

    q_source = getattr(attn_module, "q_norm", None) or getattr(attn_module, "q_proj", None)
    k_source = getattr(attn_module, "k_norm", None) or getattr(attn_module, "k_proj", None)
    if q_source is None or k_source is None:
        return forward_fn(), None

    captured: dict[str, Tensor] = {}

    def _save_q(_, __, output):
        captured["q"] = output.detach()

    def _save_k(_, __, output):
        captured["k"] = output.detach()

    handles = [
        q_source.register_forward_hook(_save_q),
        k_source.register_forward_hook(_save_k),
    ]
    try:
        embeddings = forward_fn()
    finally:
        for handle in handles:
            handle.remove()

    q = captured.get("q")
    k = captured.get("k")
    if q is None or k is None:
        return embeddings, None

    q = _reshape_attention_tensor(q, attn_module)
    k = _reshape_attention_tensor(k, attn_module)
    if q is None or k is None:
        return embeddings, None

    head_dim = q.shape[-1]
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(head_dim))
    attn = torch.softmax(attn, dim=-1)
    scores = attn.mean(dim=1).mean(dim=1)
    return embeddings, scores
