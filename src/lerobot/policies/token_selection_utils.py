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
import numpy as np
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@dataclass
class PatchGridMeta:
    num_patches: int
    num_extra_tokens: int
    patches_per_side: int


@dataclass
class TokenSelectionState:
    frame_idx: int = 0
    last_score_token: list[Tensor] | None = None
    last_score_region: list[Tensor] | None = None
    last_important_region: list[Tensor] | None = None
    last_overlay_labels: list[Tensor] | None = None
    last_overlay_grid: list[Tensor] | None = None
    # Global token pool: binary mask tracking permanently discarded tokens across the trajectory
    global_active_region_masks: list[Tensor] | None = None
    # Restore the global token pool on the next eval frame.
    pending_global_pool_restore: bool = False
    # Heatmap debug: continuous float score grid per camera
    last_heatmap_grid: list[Tensor] | None = None
    # Binary keep mask grid per camera (True = kept, False = pruned)
    last_keep_grid: list[Tensor] | None = None
    # Heatmap overlay marker grid per camera. Its semantics depend on
    # `last_heatmap_mask_mode` (e.g. pruned vs reused).
    last_heatmap_mask_grid: list[Tensor] | None = None
    last_heatmap_mask_mode: str | None = None
    # Number of frames that ran the full scoring pipeline.
    eval_frame_count: int = 0
    # Accumulated pruning counters for per-episode average pruning ratio
    total_kept: int = 0
    total_possible: int = 0
    # Accumulated reuse counters for VLA-Cache reporting.
    total_reused: int = 0
    total_reusable: int = 0
    # Stats for tqdm postfix (updated every frame when token_selection_enabled)
    last_stats: dict = None  # type: ignore[assignment]
    # Action L1 norm history: list of (frame_idx, l1_norm) per inference frame
    actions_l1_history: list = None  # type: ignore[assignment]
    # Last frame's action L1 norm (used by L1 dynamic prune ratio)
    last_actions_l1: float | None = None
    # EMA-smoothed L1 norm (used by "ema" dynamic prune mode)
    ema_l1: float | None = None
    # EMA-tracked L1 mean and variance (used by adaptive EMA threshold)
    ema_l1_mean: float | None = None
    ema_l1_var: float = 0.0
    # Acceleration-based pruning: chunk-internal velocity jerk
    last_accel: float | None = None
    ema_accel: float | None = None
    last_accel_xyz: float | None = None
    ema_accel_xyz: float | None = None
    ema_accel_xyz_var: float = 0.0
    last_accel_rot: float | None = None
    ema_accel_rot: float | None = None
    ema_accel_rot_var: float = 0.0
    # VLA-Cache: previous-frame per-camera input images.
    last_vla_cache_images: list[Tensor] | None = None
    # VLA-Cache: previous-frame aggregated vision-token relevance scores per camera.
    last_vla_cache_attention_scores: list[Tensor] | None = None
    # VLA-Cache: previous-frame expert attention maps used to derive layer reuse ratios.
    last_vla_cache_attentions: list[Tensor] | None = None
    # VLA-Cache: prefix outputs after each prefix layer from the previous frame.
    last_prefix_layer_outputs: list[Tensor] | None = None
    # VLA-Cache: previous-frame prefix KV cache stored as legacy tuples.
    last_prefix_past_key_values: list[tuple[Tensor, Tensor]] | None = None
    # VLA-Cache: cached language prompt tokens and masks; cache reuse is disabled if these change.
    last_lang_tokens: Tensor | None = None
    last_lang_masks: Tensor | None = None
    # VLA-Cache: last-frame reused visual token mask per camera.
    last_vla_cache_reuse_masks: list[Tensor] | None = None
    # CUDA latency samples (milliseconds) measured around model-side VLA-Cache inference only.
    cuda_latency_ms: list[float] | None = None
    cuda_latency_with_reuse_ms: list[float] | None = None
    cuda_latency_without_reuse_ms: list[float] | None = None

    def __post_init__(self):
        if self.last_stats is None:
            self.last_stats = {}
        if self.actions_l1_history is None:
            self.actions_l1_history = []
        if self.cuda_latency_ms is None:
            self.cuda_latency_ms = []
        if self.cuda_latency_with_reuse_ms is None:
            self.cuda_latency_with_reuse_ms = []
        if self.cuda_latency_without_reuse_ms is None:
            self.cuda_latency_without_reuse_ms = []

    def reset(self) -> None:
        self.frame_idx = 0
        self.last_score_token = None
        self.last_score_region = None
        self.last_important_region = None
        self.last_overlay_labels = None
        self.last_overlay_grid = None
        self.global_active_region_masks = None
        self.pending_global_pool_restore = False
        self.last_heatmap_grid = None
        self.last_keep_grid = None
        self.last_heatmap_mask_grid = None
        self.last_heatmap_mask_mode = None
        self.eval_frame_count = 0
        self.total_kept = 0
        self.total_possible = 0
        self.total_reused = 0
        self.total_reusable = 0
        self.last_stats = {}
        self.actions_l1_history = []
        self.last_actions_l1 = None
        self.ema_l1 = None
        self.ema_l1_mean = None
        self.ema_l1_var = 0.0
        self.last_accel = None
        self.ema_accel = None
        self.last_accel_xyz = None
        self.ema_accel_xyz = None
        self.ema_accel_xyz_var = 0.0
        self.last_accel_rot = None
        self.ema_accel_rot = None
        self.ema_accel_rot_var = 0.0
        self.last_vla_cache_images = None
        self.last_vla_cache_attention_scores = None
        self.last_vla_cache_attentions = None
        self.last_prefix_layer_outputs = None
        self.last_prefix_past_key_values = None
        self.last_lang_tokens = None
        self.last_lang_masks = None
        self.last_vla_cache_reuse_masks = None
        self.cuda_latency_ms = []
        self.cuda_latency_with_reuse_ms = []
        self.cuda_latency_without_reuse_ms = []


def infer_patch_grid(num_tokens: int) -> PatchGridMeta:
    patches_per_side = int(round(math.sqrt(num_tokens)))
    if patches_per_side * patches_per_side == num_tokens:
        return PatchGridMeta(num_patches=num_tokens, num_extra_tokens=0, patches_per_side=patches_per_side)
    patches_per_side = int(round(math.sqrt(max(num_tokens - 1, 0))))
    if patches_per_side * patches_per_side == num_tokens - 1:
        return PatchGridMeta(num_patches=num_tokens - 1, num_extra_tokens=1, patches_per_side=patches_per_side)
    raise ValueError(f"Unable to infer square patch grid from token count {num_tokens}.")


def split_patch_tokens(img_emb: Tensor) -> tuple[Tensor, PatchGridMeta]:
    meta = infer_patch_grid(img_emb.shape[1])
    if meta.num_extra_tokens > 0:
        return img_emb[:, meta.num_extra_tokens :, :], meta
    return img_emb, meta


def compute_region_scores(
    token_scores: Tensor, patches_per_side: int, region_patch_size: int
) -> Tensor:
    if patches_per_side % region_patch_size != 0:
        raise ValueError(
            f"region_patch_size ({region_patch_size}) must divide patches_per_side ({patches_per_side})."
        )
    batch_size, num_patches = token_scores.shape
    if num_patches != patches_per_side * patches_per_side:
        raise ValueError("Token scores do not match the inferred grid.")
    region_h = patches_per_side // region_patch_size
    region_w = patches_per_side // region_patch_size
    reshaped = token_scores.view(batch_size, region_h, region_patch_size, region_w, region_patch_size)
    region_scores = reshaped.mean(dim=(2, 4))
    return region_scores.view(batch_size, region_h * region_w)


def expand_region_mask(
    region_mask: Tensor, patches_per_side: int, region_patch_size: int
) -> Tensor:
    if patches_per_side % region_patch_size != 0:
        raise ValueError(
            f"region_patch_size ({region_patch_size}) must divide patches_per_side ({patches_per_side})."
        )
    batch_size, num_regions = region_mask.shape
    region_h = patches_per_side // region_patch_size
    region_w = patches_per_side // region_patch_size
    if num_regions != region_h * region_w:
        raise ValueError("Region mask does not match the inferred grid.")
    region_mask_2d = region_mask.view(batch_size, region_h, region_w)
    expanded = region_mask_2d.repeat_interleave(region_patch_size, dim=1).repeat_interleave(region_patch_size, dim=2)
    return expanded.view(batch_size, patches_per_side * patches_per_side)


def compute_relative_deviation(raw_value: float | None, ema_value: float | None) -> float | None:
    if raw_value is None or ema_value is None:
        return None
    denom = max(abs(ema_value), 1e-8)
    return (raw_value - ema_value) / denom


def sigmoid_ratio(value: float) -> float:
    x = max(min(value, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def ratio_to_keep_tokens(
    ratio: float,
    min_kept_tokens: int,
    max_kept_tokens: int,
    direction: str,
) -> int:
    ratio = max(min(ratio, 1.0), 0.0)
    span = max_kept_tokens - min_kept_tokens
    if direction == "reverse":
        eff = min_kept_tokens + ratio * span
    else:
        eff = max_kept_tokens - ratio * span
    return int(round(min(max(eff, min_kept_tokens), max_kept_tokens)))


def update_ema_mean_and_var(
    value: float,
    mean: float | None,
    var: float,
    alpha: float,
) -> tuple[float, float]:
    if mean is None:
        return value, 0.0
    new_mean = alpha * mean + (1.0 - alpha) * value
    new_var = alpha * var + (1.0 - alpha) * (value - mean) * (value - new_mean)
    return new_mean, max(new_var, 0.0)


def _to_channels_first(images: Tensor) -> Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected 4D image tensor, got shape {tuple(images.shape)}")
    if images.shape[1] in {1, 3, 4}:
        return images
    if images.shape[-1] in {1, 3, 4}:
        return images.permute(0, 3, 1, 2)
    raise ValueError(f"Unable to infer channels dimension from shape {tuple(images.shape)}")


def compute_patch_cosine_similarity(
    current_images: Tensor,
    previous_images: Tensor,
    patches_per_side: int,
) -> Tensor:
    current = _to_channels_first(current_images).float()
    previous = _to_channels_first(previous_images).float()
    if current.shape != previous.shape:
        raise ValueError(
            "Current and previous images must share the same shape for VLA-Cache similarity computation."
        )

    # The reference VLA-Cache implementation computes patch similarity on raw
    # RGB images. PI policies preprocess images to [-1, 1] for PaliGemma, so
    # map them back to [0, 1] before patchifying to keep the original
    # similarity threshold semantics (e.g. 0.996) meaningful.
    if current.min().item() < 0.0 or previous.min().item() < 0.0:
        current = (current + 1.0) * 0.5
        previous = (previous + 1.0) * 0.5

    _, _, height, width = current.shape
    if height % patches_per_side != 0 or width % patches_per_side != 0:
        raise ValueError(
            f"Image size {(height, width)} is not divisible by patches_per_side={patches_per_side}."
        )

    patch_h = height // patches_per_side
    patch_w = width // patches_per_side

    def patchify(images: Tensor) -> Tensor:
        patches = images.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(images.shape[0], patches_per_side * patches_per_side, -1)

    current_patches = F.normalize(patchify(current), dim=-1)
    previous_patches = F.normalize(patchify(previous), dim=-1)
    return (current_patches * previous_patches).sum(dim=-1)


def select_vla_cache_reuse_tokens(
    similarity_scores: Tensor,
    valid_mask: Tensor,
    reuse_tokens: int,
    similarity_threshold: float,
    protected_scores: Tensor | None = None,
    protect_top_tokens: int = 0,
) -> tuple[Tensor, Tensor]:
    if similarity_scores.dim() != 2:
        raise ValueError("similarity_scores must have shape [batch, num_patches].")
    if valid_mask.dim() == 1:
        valid_mask = valid_mask[:, None].expand_as(similarity_scores)
    elif valid_mask.shape != similarity_scores.shape:
        raise ValueError("valid_mask must broadcast to similarity_scores.")

    scores = similarity_scores.clone()
    scores = torch.where(valid_mask, scores, torch.full_like(scores, -float("inf")))

    if protected_scores is not None and protect_top_tokens > 0:
        protect_top_tokens = min(protect_top_tokens, protected_scores.shape[-1])
        if protect_top_tokens > 0:
            protected_mask = torch.zeros_like(scores, dtype=torch.bool)
            protected_valid_scores = torch.where(
                valid_mask,
                protected_scores,
                torch.full_like(protected_scores, -float("inf")),
            )
            top_idx = torch.topk(
                protected_valid_scores,
                k=protect_top_tokens,
                dim=-1,
            ).indices
            protected_mask.scatter_(1, top_idx, True)
            scores = torch.where(protected_mask, torch.full_like(scores, -float("inf")), scores)

    reuse_mask = torch.zeros_like(scores, dtype=torch.bool)
    selected_scores = torch.full_like(scores, -float("inf"))
    if reuse_tokens <= 0:
        return reuse_mask, selected_scores

    top_k = min(reuse_tokens, scores.shape[-1])
    top_values, top_idx = torch.topk(scores, k=top_k, dim=-1)
    keep = torch.isfinite(top_values) & (top_values >= similarity_threshold)
    if keep.any():
        reuse_mask.scatter_(1, top_idx, keep)
        selected_scores.scatter_(1, top_idx, torch.where(keep, top_values, torch.full_like(top_values, -float("inf"))))

    return reuse_mask, selected_scores


@torch.no_grad()
def compute_vla_cache_layer_schedule(
    attentions: list[Tensor],
    apply_weighted_growth: bool = True,
    growth_factor: float = 0.55,
) -> Tensor | None:
    if not attentions:
        return None

    device = attentions[0].device
    entropies = []
    for attn in attentions:
        if attn.dim() != 4:
            raise ValueError("Attention tensors must have shape [batch, heads, query, key].")
        attn_mean = attn.float().mean(dim=1)
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        attn_mean = torch.nan_to_num(attn_mean, nan=0.0)
        token_entropy = -(attn_mean * (attn_mean + 1e-10).log()).sum(dim=-1)
        entropies.append(token_entropy.mean())

    entropies_t = torch.stack(entropies)
    if torch.allclose(entropies_t.max(), entropies_t.min()):
        reuse = torch.ones_like(entropies_t)
    else:
        norm_entropy = (entropies_t - entropies_t.min()) / (entropies_t.max() - entropies_t.min() + 1e-10)
        reuse = 1.0 - norm_entropy

    if apply_weighted_growth and reuse.numel() > 1:
        reuse_list = reuse.tolist()
        for idx in range(1, len(reuse_list)):
            delta = reuse_list[idx] - reuse_list[idx - 1]
            if delta > 0:
                reuse_list[idx] = reuse_list[idx - 1] + delta * growth_factor
        reuse = torch.tensor(reuse_list, dtype=torch.float32, device=device)

    return reuse.clamp(0.0, 1.0)


def build_layer_reuse_masks(
    selected_scores: Tensor,
    always_reuse_mask: Tensor,
    layer_schedule: Tensor | None,
) -> list[Tensor]:
    if selected_scores.dim() != 2 or always_reuse_mask.shape != selected_scores.shape:
        raise ValueError("selected_scores and always_reuse_mask must share shape [batch, num_tokens].")

    if layer_schedule is None:
        return [always_reuse_mask | torch.isfinite(selected_scores)]

    eligible_mask = torch.isfinite(selected_scores)
    selected_count = eligible_mask.sum(dim=-1)
    masks: list[Tensor] = []
    for ratio in layer_schedule.tolist():
        layer_mask = always_reuse_mask.clone()
        if ratio <= 0.0:
            masks.append(layer_mask)
            continue
        max_k = int(selected_scores.shape[-1])
        if max_k <= 0:
            masks.append(layer_mask)
            continue
        want_k = torch.round(selected_count.float() * float(ratio)).to(dtype=torch.int64)
        want_k = torch.clamp(want_k, min=0, max=max_k)
        if int(want_k.max().item()) == 0:
            masks.append(layer_mask)
            continue
        top_idx = torch.topk(
            torch.where(eligible_mask, selected_scores, torch.full_like(selected_scores, -float("inf"))),
            k=int(want_k.max().item()),
            dim=-1,
        ).indices
        top_rank = torch.arange(top_idx.shape[1], device=selected_scores.device)[None, :]
        keep = top_rank < want_k[:, None]
        layer_selected = torch.zeros_like(layer_mask)
        layer_selected.scatter_(1, top_idx, keep)
        masks.append(layer_mask | layer_selected)
    return masks


def apply_min_max_constraints(
    keep_pre: Tensor, score_region: Tensor, min_regions: int, max_regions: int
) -> tuple[Tensor, Tensor]:
    batch_size, num_regions = keep_pre.shape
    min_regions = max(min_regions, 0)
    max_regions = min(max_regions, num_regions) if max_regions > 0 else num_regions
    keep_mask = keep_pre.clone()
    clipped_mask = torch.zeros_like(keep_pre, dtype=torch.bool)
    order = torch.argsort(score_region, dim=-1, descending=True)
    for batch_idx in range(batch_size):
        keep_count = int(keep_mask[batch_idx].sum().item())
        if keep_count < min_regions:
            needed = min_regions - keep_count
            add_indices = [idx for idx in order[batch_idx].tolist() if not keep_mask[batch_idx, idx]][:needed]
            if add_indices:
                keep_mask[batch_idx, add_indices] = True
        keep_count = int(keep_mask[batch_idx].sum().item())
        if keep_count > max_regions:
            top = order[batch_idx, :max_regions]
            new_keep = torch.zeros_like(keep_mask[batch_idx])
            new_keep[top] = True
            clipped = keep_mask[batch_idx] & ~new_keep
            keep_mask[batch_idx] = new_keep
            clipped_mask[batch_idx] = clipped
    return keep_mask, clipped_mask


def discard_top_scoring_regions(
    score_region: Tensor, 
    keep_region: Tensor, 
    discard_ratio: float,
    max_discard: Tensor | None = None,
    mode: str = "top",
) -> Tensor:
    """Select regions from the previously *kept* regions to discard.
    
    Args:
        score_region: [B, R] raw scores from the previous frame.
        keep_region: [B, R] boolean mask of regions kept in the previous frame.
        discard_ratio: Fraction (0.0 to 1.0) of previous kept regions to discard.
        max_discard: [B] integer tensor specifying max allowed discards per batch item 
                     to avoid depleting the global token pool.
        mode: "top"    = discard highest-scoring (old center, trajectory remnants);
              "bottom" = discard lowest-scoring (least relevant background);
              "middle" = discard mid-scoring (ambiguous/uncertain tokens).
                     
    Returns:
        Boolean mask [B, R] with True for regions scheduled to be permanently discarded.
    """
    batch_size = score_region.shape[0]
    discard_mask = torch.zeros_like(keep_region, dtype=torch.bool)
    if discard_ratio <= 0.0:
        return discard_mask
        
    for batch_idx in range(batch_size):
        kept_indices = keep_region[batch_idx].nonzero(as_tuple=True)[0]
        if len(kept_indices) == 0:
            continue
            
        num_to_discard = int(math.ceil(len(kept_indices) * discard_ratio))
        if max_discard is not None:
            num_to_discard = min(num_to_discard, max_discard[batch_idx].item())
            
        if num_to_discard <= 0:
            continue
            
        kept_scores = score_region[batch_idx, kept_indices]
        
        if mode == "middle":
            # Sort ascending, pick the middle slice
            sorted_idx = torch.argsort(kept_scores, descending=False)
            n = len(sorted_idx)
            mid_start = (n - num_to_discard) // 2
            mid_end = mid_start + num_to_discard
            selected_idx = sorted_idx[mid_start:mid_end]
        else:
            # "top" → sort descending (highest first); "bottom" → sort ascending
            _descending = (mode == "top")
            sorted_idx = torch.argsort(kept_scores, descending=_descending)[:num_to_discard]
            selected_idx = sorted_idx
        
        absolute_discard_indices = kept_indices[selected_idx]
        discard_mask[batch_idx, absolute_discard_indices] = True
        
    return discard_mask

def build_overlay_labels(
    keep: Tensor,
    important: Tensor,
    clipped: Tensor,
) -> Tensor:
    if keep.dim() == 1:
        keep = keep.unsqueeze(0)
    if important.dim() == 1:
        important = important.unsqueeze(0)
    if clipped.dim() == 1:
        clipped = clipped.unsqueeze(0)

    overlay = torch.zeros_like(keep, dtype=torch.uint8)  # default = 0 (green)
    overlay[keep & ~important] = 1
    overlay[keep & important] = 2
    overlay[clipped] = 3
    return overlay


def build_overlay(
    image: Tensor,
    region_h: int,
    region_w: int,
    region_patch_size: int,
    important: Tensor,
    keep: Tensor,
    clipped: Tensor,
    show_scores: bool,
    show_ids: bool,
    scores: Tensor | None = None,
) -> Tensor:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        if isinstance(image, torch.Tensor):
            return image
        return torch.from_numpy(image)

    height, width = image.shape[:2]
    overlay = Image.fromarray(image)
    draw = ImageDraw.Draw(overlay)

    region_height = height // region_h
    region_width = width // region_w
    font = None
    if show_scores or show_ids:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for idx in range(region_h * region_w):
        row = idx // region_w
        col = idx % region_w
        y0 = row * region_height
        x0 = col * region_width
        y1 = y0 + region_height
        x1 = x0 + region_width
        if clipped[idx]:
            color = (0, 0, 255)
        elif important[idx]:
            color = (255, 0, 0)
        elif keep[idx]:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if show_scores or show_ids:
            label_parts = []
            if show_ids:
                label_parts.append(str(idx))
            if show_scores and scores is not None:
                label_parts.append(f"{float(scores[idx]):.3f}")
            if label_parts:
                draw.text((x0 + 2, y0 + 2), " ".join(label_parts), fill=(255, 255, 255), font=font)
    return torch.from_numpy(np.array(overlay))


def to_tensor_list(values: Iterable[Tensor]) -> list[Tensor]:
    return [value for value in values]


def prune_image_embeddings(
    image_embs: list[Tensor],
    keep_token_masks: list[Tensor],
    img_masks: list[Tensor],
    metas: list[PatchGridMeta],
) -> tuple[list[Tensor], list[Tensor]]:
    pruned_embs: list[Tensor] = []
    pruned_pad_masks: list[Tensor] = []
    for emb, keep_mask, img_mask, meta in zip(image_embs, keep_token_masks, img_masks, metas, strict=True):
        batch_size, _, hidden = emb.shape
        per_batch_tokens = []
        lengths = []
        for batch_idx in range(batch_size):
            if not bool(img_mask[batch_idx].item()):
                per_batch_tokens.append(emb.new_zeros((0, hidden)))
                lengths.append(0)
                continue
            extra = (
                emb[batch_idx, : meta.num_extra_tokens, :]
                if meta.num_extra_tokens > 0
                else emb.new_zeros((0, hidden))
            )
            patches = emb[batch_idx, meta.num_extra_tokens :, :]
            keep_idx = torch.nonzero(keep_mask[batch_idx], as_tuple=False).squeeze(-1)
            kept = patches.index_select(0, keep_idx) if keep_idx.numel() > 0 else patches[:0]
            tokens = torch.cat([extra, kept], dim=0)
            per_batch_tokens.append(tokens)
            lengths.append(tokens.shape[0])
        max_len = max(lengths) if lengths else 0
        padded = emb.new_zeros((batch_size, max_len, hidden))
        pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=emb.device)
        for batch_idx, length in enumerate(lengths):
            if length > 0:
                padded[batch_idx, :length] = per_batch_tokens[batch_idx]
                pad_mask[batch_idx, :length] = True
        pruned_embs.append(padded)
        pruned_pad_masks.append(pad_mask)
    return pruned_embs, pruned_pad_masks
