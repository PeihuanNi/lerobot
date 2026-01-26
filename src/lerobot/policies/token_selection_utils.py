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
    last_region_embs: list[Tensor] | None = None
    last_image_embs: list[Tensor] | None = None

    def reset(self) -> None:
        self.frame_idx = 0
        self.last_score_token = None
        self.last_score_region = None
        self.last_important_region = None
        self.last_region_embs = None
        self.last_image_embs = None


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


def compute_region_embeddings(
    patch_emb: Tensor, patches_per_side: int, region_patch_size: int
) -> Tensor:
    if patches_per_side % region_patch_size != 0:
        raise ValueError(
            f"region_patch_size ({region_patch_size}) must divide patches_per_side ({patches_per_side})."
        )
    batch_size, num_patches, hidden = patch_emb.shape
    if num_patches != patches_per_side * patches_per_side:
        raise ValueError("Patch embeddings do not match the inferred grid.")
    region_h = patches_per_side // region_patch_size
    region_w = patches_per_side // region_patch_size
    reshaped = patch_emb.view(
        batch_size,
        region_h,
        region_patch_size,
        region_w,
        region_patch_size,
        hidden,
    )
    region_emb = reshaped.mean(dim=(2, 4))
    return region_emb.view(batch_size, region_h * region_w, hidden)


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


def select_regions_by_mass(score_region: Tensor, mass: float) -> Tensor:
    batch_size, num_regions = score_region.shape
    total = score_region.sum(dim=-1, keepdim=True)
    if mass <= 0:
        keep_count = torch.ones(batch_size, dtype=torch.long, device=score_region.device)
        order = torch.argsort(score_region, dim=-1, descending=True)
    else:
        normalized = torch.where(
            total > 0,
            score_region / total,
            torch.full_like(score_region, 1.0 / num_regions),
        )
        order = torch.argsort(normalized, dim=-1, descending=True)
        sorted_scores = torch.gather(normalized, 1, order)
        cumulative = torch.cumsum(sorted_scores, dim=-1)
        keep_count = (cumulative < mass).sum(dim=-1) + 1
    keep_count = torch.clamp(keep_count, min=1, max=num_regions)
    mask = torch.zeros_like(score_region, dtype=torch.bool)
    for batch_idx in range(batch_size):
        selected = order[batch_idx, : keep_count[batch_idx]]
        mask[batch_idx, selected] = True
    return mask


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


def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    a_f = a.float()
    b_f = b.float()
    a_norm = torch.linalg.vector_norm(a_f, dim=-1)
    b_norm = torch.linalg.vector_norm(b_f, dim=-1)
    denom = (a_norm * b_norm).clamp_min(eps)
    return (a_f * b_f).sum(dim=-1) / denom


def compute_temporal_mask(
    region_emb: Tensor, prev_region_emb: Tensor | None, threshold: float, valid_mask: Tensor
) -> Tensor:
    if prev_region_emb is None:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    cos = cosine_similarity(region_emb, prev_region_emb)
    return (cos >= threshold) & valid_mask


def compute_spatial_mask(
    region_emb: Tensor, patches_per_side: int, region_patch_size: int, radius: int, threshold: float
) -> Tensor:
    batch_size, num_regions, hidden = region_emb.shape
    region_h = patches_per_side // region_patch_size
    region_w = patches_per_side // region_patch_size
    if num_regions != region_h * region_w:
        raise ValueError("Region embeddings do not match the inferred grid.")
    region_emb_2d = region_emb.view(batch_size, region_h, region_w, hidden)
    spatial_mask = torch.zeros(batch_size, region_h, region_w, dtype=torch.bool, device=region_emb.device)
    for row in range(region_h):
        r0 = max(0, row - radius)
        r1 = min(region_h, row + radius + 1)
        for col in range(region_w):
            c0 = max(0, col - radius)
            c1 = min(region_w, col + radius + 1)
            anchor = region_emb_2d[:, row, col, :].unsqueeze(1)
            neighbors = region_emb_2d[:, r0:r1, c0:c1, :].reshape(batch_size, -1, hidden)
            cos = cosine_similarity(anchor, neighbors)
            mean_cos = cos.mean(dim=-1)
            spatial_mask[:, row, col] = mean_cos >= threshold
    return spatial_mask.view(batch_size, region_h * region_w)


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
        elif important[idx] and keep[idx]:
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


def apply_partial_update(
    image_embs: list[Tensor],
    last_image_embs: list[Tensor] | None,
    important_token_masks: list[Tensor] | None,
    metas: list[PatchGridMeta],
) -> list[Tensor]:
    if last_image_embs is None or important_token_masks is None:
        return image_embs
    updated = []
    for emb, prev, mask, meta in zip(image_embs, last_image_embs, important_token_masks, metas, strict=True):
        if prev is None or prev.shape != emb.shape:
            updated.append(emb)
            continue
        if meta.num_extra_tokens > 0:
            extra = emb[:, : meta.num_extra_tokens, :]
            cur_patch = emb[:, meta.num_extra_tokens :, :]
            prev_patch = prev[:, meta.num_extra_tokens :, :]
        else:
            extra = None
            cur_patch = emb
            prev_patch = prev
        if mask.shape[-1] != cur_patch.shape[1]:
            updated.append(emb)
            continue
        combined_patch = torch.where(mask.unsqueeze(-1), cur_patch, prev_patch)
        if extra is not None:
            updated.append(torch.cat([extra, combined_patch], dim=1))
        else:
            updated.append(combined_patch)
    return updated


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
