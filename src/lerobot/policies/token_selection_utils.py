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
from torch import Tensor, nn

from lerobot.utils.profiling import profile_range


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
    # Number of frames that ran the full scoring pipeline.
    eval_frame_count: int = 0
    # Accumulated pruning counters for per-episode average pruning ratio
    total_kept: int = 0
    total_possible: int = 0
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
    # Velocity-based pruning: chunk-integrated action magnitude
    last_velocity: float | None = None
    ema_velocity: float | None = None
    last_velocity_xyz: float | None = None
    ema_velocity_xyz: float | None = None
    ema_velocity_xyz_var: float = 0.0
    last_velocity_rot: float | None = None
    ema_velocity_rot: float | None = None
    ema_velocity_rot_var: float = 0.0

    def __post_init__(self):
        if self.last_stats is None:
            self.last_stats = {}
        if self.actions_l1_history is None:
            self.actions_l1_history = []

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
        self.eval_frame_count = 0
        self.total_kept = 0
        self.total_possible = 0
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
        self.last_velocity = None
        self.ema_velocity = None
        self.last_velocity_xyz = None
        self.ema_velocity_xyz = None
        self.ema_velocity_xyz_var = 0.0
        self.last_velocity_rot = None
        self.ema_velocity_rot = None
        self.ema_velocity_rot_var = 0.0


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
        finite_order = order[batch_idx][torch.isfinite(score_region[batch_idx, order[batch_idx]])]
        keep_count = int(keep_mask[batch_idx].sum().item())
        if keep_count < min_regions:
            needed = min_regions - keep_count
            add_indices = [idx for idx in finite_order.tolist() if not keep_mask[batch_idx, idx]][:needed]
            if add_indices:
                keep_mask[batch_idx, add_indices] = True
        keep_count = int(keep_mask[batch_idx].sum().item())
        if keep_count > max_regions:
            top = finite_order[:max_regions]
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
              "middle" = discard mid-scoring (ambiguous/uncertain tokens);
              "random" = randomly discard from previous kept regions.
                     
    Returns:
        Boolean mask [B, R] with True for regions scheduled to be permanently discarded.
    """
    batch_size = score_region.shape[0]
    discard_mask = torch.zeros_like(keep_region, dtype=torch.bool)
    if discard_ratio <= 0.0:
        return discard_mask
    valid_modes = {"top", "bottom", "middle", "random"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid discard mode: {mode}")
        
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
        elif mode == "random":
            selected_idx = torch.randperm(len(kept_indices), device=kept_indices.device)[:num_to_discard]
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


def compact_prefix_inputs(
    prefix_embs: Tensor,
    prefix_pad_masks: Tensor,
    prefix_att_masks: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Drop prefix columns that are padding for the whole batch."""
    if prefix_pad_masks.ndim != 2:
        raise ValueError(f"prefix_pad_masks must have shape [B, L], got {tuple(prefix_pad_masks.shape)}")
    if prefix_embs.ndim != 3:
        raise ValueError(f"prefix_embs must have shape [B, L, D], got {tuple(prefix_embs.shape)}")
    if prefix_att_masks.ndim != 2:
        raise ValueError(f"prefix_att_masks must have shape [B, L], got {tuple(prefix_att_masks.shape)}")
    if prefix_embs.shape[:2] != prefix_pad_masks.shape or prefix_att_masks.shape != prefix_pad_masks.shape:
        raise ValueError(
            "prefix_embs, prefix_pad_masks and prefix_att_masks have incompatible sequence shapes: "
            f"{tuple(prefix_embs.shape)}, {tuple(prefix_pad_masks.shape)}, {tuple(prefix_att_masks.shape)}"
        )

    keep_idx = torch.nonzero(prefix_pad_masks.any(dim=0), as_tuple=False).squeeze(-1)
    return (
        prefix_embs.index_select(1, keep_idx),
        prefix_pad_masks.index_select(1, keep_idx),
        prefix_att_masks.index_select(1, keep_idx),
    )


def embed_images_for_token_selection(
    embed_image,
    images: list[Tensor],
    *,
    parallel_streams: bool = True,
) -> list[Tensor]:
    """Embed actual token-selection cameras without touching baseline inference."""
    if len(images) <= 1:
        return [embed_image(img) for img in images]

    first_shape = images[0].shape
    if parallel_streams and all(img.shape == first_shape for img in images):
        batch_sizes = [img.shape[0] for img in images]
        image_batch = torch.cat(images, dim=0)
        image_embs = embed_image(image_batch)
        return list(image_embs.split(batch_sizes, dim=0))

    if parallel_streams and images[0].is_cuda:
        device = images[0].device
        current_stream = torch.cuda.current_stream(device)
        streams = [torch.cuda.Stream(device=device) for _ in images]
        outputs: list[Tensor | None] = [None] * len(images)
        for stream in streams:
            stream.wait_stream(current_stream)
        for idx, (stream, img) in enumerate(zip(streams, images, strict=True)):
            with torch.cuda.stream(stream):
                outputs[idx] = embed_image(img)
        for stream in streams:
            current_stream.wait_stream(stream)
        embedded = []
        for out in outputs:
            if out is None:
                raise RuntimeError("Missing image embedding output.")
            out.record_stream(current_stream)
            embedded.append(out)
        return embedded

    return [embed_image(img) for img in images]


def embed_pruned_images_for_token_selection(
    embed_image_pruned,
    images: list[Tensor],
    keep_token_masks: list[Tensor],
    img_masks: list[Tensor],
    *,
    parallel_streams: bool = True,
) -> tuple[list[Tensor], list[Tensor]]:
    """Embed pruned token-selection cameras without touching baseline inference."""
    if len(images) <= 1:
        image_embs = []
        image_pad_masks = []
        for img, keep_mask, img_mask in zip(images, keep_token_masks, img_masks, strict=True):
            pruned_embs, pruned_pad_mask = embed_image_pruned(img, keep_mask, img_mask)
            image_embs.append(pruned_embs)
            image_pad_masks.append(pruned_pad_mask)
        return image_embs, image_pad_masks

    first_image_shape = images[0].shape
    first_keep_shape = keep_token_masks[0].shape
    first_mask_shape = img_masks[0].shape
    can_batch = (
        parallel_streams
        and all(img.shape == first_image_shape for img in images)
        and all(keep_mask.shape == first_keep_shape for keep_mask in keep_token_masks)
        and all(img_mask.shape == first_mask_shape for img_mask in img_masks)
    )
    if can_batch:
        batch_sizes = [img.shape[0] for img in images]
        image_batch = torch.cat(images, dim=0)
        keep_mask_batch = torch.cat(keep_token_masks, dim=0)
        img_mask_batch = torch.cat(img_masks, dim=0)
        pruned_embs, pruned_pad_mask = embed_image_pruned(image_batch, keep_mask_batch, img_mask_batch)
        return (
            list(pruned_embs.split(batch_sizes, dim=0)),
            list(pruned_pad_mask.split(batch_sizes, dim=0)),
        )

    if not parallel_streams or not images[0].is_cuda:
        image_embs = []
        image_pad_masks = []
        for img, keep_mask, img_mask in zip(images, keep_token_masks, img_masks, strict=True):
            pruned_embs, pruned_pad_mask = embed_image_pruned(img, keep_mask, img_mask)
            image_embs.append(pruned_embs)
            image_pad_masks.append(pruned_pad_mask)
        return image_embs, image_pad_masks

    device = images[0].device
    current_stream = torch.cuda.current_stream(device)
    streams = [torch.cuda.Stream(device=device) for _ in images]
    outputs: list[tuple[Tensor, Tensor] | None] = [None] * len(images)
    for stream in streams:
        stream.wait_stream(current_stream)
    for idx, (stream, img, keep_mask, img_mask) in enumerate(
        zip(streams, images, keep_token_masks, img_masks, strict=True)
    ):
        with torch.cuda.stream(stream):
            outputs[idx] = embed_image_pruned(img, keep_mask, img_mask)
    for stream in streams:
        current_stream.wait_stream(stream)

    image_embs = []
    image_pad_masks = []
    for output in outputs:
        if output is None:
            raise RuntimeError("Missing pruned image embedding output.")
        output[0].record_stream(current_stream)
        output[1].record_stream(current_stream)
        image_embs.append(output[0])
        image_pad_masks.append(output[1])
    return image_embs, image_pad_masks


def embed_pruned_vision_tokens(
    pixel_values: Tensor,
    keep_masks: Tensor,
    img_masks: Tensor,
    patch_embedding: nn.Module,
    position_embedding: nn.Module,
    encoder: nn.Module,
    post_layernorm: nn.Module,
    projector: nn.Module,
) -> tuple[Tensor, Tensor]:
    if keep_masks.dim() != 2:
        raise ValueError(f"keep_masks must have shape [B, N], got {tuple(keep_masks.shape)}")

    patch_dtype = patch_embedding.weight.dtype
    pos_dtype = position_embedding.weight.dtype
    with profile_range("policy.token_selection.pruned_vision_model.patch_embed_dense"):
        patch_embeds = patch_embedding(pixel_values.to(dtype=patch_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    batch_size, num_patches, _ = patch_embeds.shape
    if keep_masks.shape != (batch_size, num_patches):
        raise ValueError(
            f"keep_masks shape {tuple(keep_masks.shape)} does not match patch sequence {(batch_size, num_patches)}"
        )

    with profile_range("policy.token_selection.pruned_vision_model.add_pos_embedding"):
        position_ids = torch.arange(num_patches, device=patch_embeds.device)
        patch_embeds = patch_embeds.to(dtype=pos_dtype)
        patch_embeds = patch_embeds + position_embedding(position_ids).unsqueeze(0).to(dtype=pos_dtype)

    encoder_dtype = None
    encoder_layers = getattr(encoder, "layers", None)
    if encoder_layers is not None and len(encoder_layers) > 0:
        encoder_dtype = encoder_layers[0].layer_norm1.weight.dtype
    elif hasattr(encoder, "layer_norm1"):
        encoder_dtype = encoder.layer_norm1.weight.dtype

    projected_dim = getattr(getattr(projector, "linear", None), "out_features", patch_embeds.shape[-1])
    projector_dtype = None
    projector_param = next(projector.parameters(), None)
    if projector_param is not None:
        projector_dtype = projector_param.dtype
    post_layernorm_dtype = getattr(post_layernorm, "weight", None)
    if post_layernorm_dtype is not None:
        post_layernorm_dtype = post_layernorm_dtype.dtype
    projected_samples: list[Tensor | None] = [None] * batch_size
    lengths: list[int] = [0] * batch_size
    grouped_keep_indices: dict[int, list[tuple[int, Tensor]]] = {}

    for batch_idx in range(batch_size):
        sample_keep_mask = keep_masks[batch_idx] & img_masks[batch_idx]
        keep_idx = torch.nonzero(sample_keep_mask, as_tuple=False).squeeze(-1)
        length = int(keep_idx.shape[0])
        lengths[batch_idx] = length
        if length == 0:
            projected_samples[batch_idx] = patch_embeds.new_zeros((0, projected_dim))
            continue
        grouped_keep_indices.setdefault(length, []).append((batch_idx, keep_idx))

    for length, group in grouped_keep_indices.items():
        with profile_range("policy.token_selection.pruned_vision_model.noncontig_gather"):
            batch_indices = torch.tensor(
                [batch_idx for batch_idx, _ in group],
                dtype=torch.long,
                device=patch_embeds.device,
            )
            keep_indices = torch.stack([keep_idx for _, keep_idx in group], dim=0)
            hidden_states = patch_embeds[batch_indices[:, None], keep_indices]
        if encoder_dtype is not None:
            hidden_states = hidden_states.to(dtype=encoder_dtype)
        with profile_range("policy.token_selection.pruned_vision_model.pruned_vit_forward"):
            encoder_outputs = encoder(inputs_embeds=hidden_states)
        if isinstance(encoder_outputs, tuple):
            hidden_states = encoder_outputs[0]
        else:
            hidden_states = encoder_outputs.last_hidden_state
        if post_layernorm_dtype is not None:
            hidden_states = hidden_states.to(dtype=post_layernorm_dtype)
        with profile_range("policy.token_selection.pruned_vision_model.post_layernorm"):
            hidden_states = post_layernorm(hidden_states)
        if projector_dtype is not None:
            hidden_states = hidden_states.to(dtype=projector_dtype)
        with profile_range("policy.token_selection.pruned_vision_model.projector"):
            hidden_states = projector(hidden_states)
        for group_idx, (batch_idx, _) in enumerate(group):
            projected_samples[batch_idx] = hidden_states[group_idx]

    max_len = max(lengths) if lengths else 0
    out_dtype = next(
        (
            sample.dtype
            for sample in projected_samples
            if sample is not None and sample.numel() > 0
        ),
        patch_embeds.dtype,
    )
    with profile_range("policy.token_selection.pruned_vision_model.pad_output"):
        padded = torch.zeros((batch_size, max_len, projected_dim), dtype=out_dtype, device=patch_embeds.device)
        pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=patch_embeds.device)
        for batch_idx, length in enumerate(lengths):
            if length > 0:
                projected_sample = projected_samples[batch_idx]
                if projected_sample is None:
                    raise RuntimeError("Missing pruned vision output for non-empty token selection.")
                padded[batch_idx, :length] = projected_sample
                pad_mask[batch_idx, :length] = True

    return padded, pad_mask
