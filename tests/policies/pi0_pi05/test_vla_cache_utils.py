#!/usr/bin/env python

import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.token_selection_utils import (
    build_layer_reuse_masks,
    compute_patch_cosine_similarity,
    compute_vla_cache_layer_schedule,
    select_vla_cache_reuse_tokens,
)


def test_pi0_config_accepts_vla_cache():
    config = PI0Config(
        token_selection_enabled=True,
        grad_score_method="vla_cache",
        vla_cache_reuse_tokens=64,
    )
    assert config.grad_score_method == "vla_cache"
    assert config.vla_cache_reuse_tokens == 64


def test_pi05_config_accepts_vla_cache():
    config = PI05Config(
        token_selection_enabled=True,
        grad_score_method="vla_cache",
        vla_cache_reuse_tokens=96,
    )
    assert config.grad_score_method == "vla_cache"
    assert config.vla_cache_reuse_tokens == 96


def test_compute_patch_cosine_similarity_identical_images():
    image = torch.arange(0, 3 * 4 * 4, dtype=torch.float32).view(1, 3, 4, 4)
    similarity = compute_patch_cosine_similarity(image, image.clone(), patches_per_side=2)
    assert similarity.shape == (1, 4)
    assert torch.allclose(similarity, torch.ones_like(similarity), atol=1e-5)


def test_select_vla_cache_reuse_tokens_respects_protected_scores():
    similarity = torch.tensor([[0.99, 0.98, 0.97, 0.96]], dtype=torch.float32)
    valid_mask = torch.ones_like(similarity, dtype=torch.bool)
    protected_scores = torch.tensor([[0.1, 0.2, 10.0, 0.3]], dtype=torch.float32)

    reuse_mask, selected_scores = select_vla_cache_reuse_tokens(
        similarity_scores=similarity,
        valid_mask=valid_mask,
        reuse_tokens=2,
        similarity_threshold=0.95,
        protected_scores=protected_scores,
        protect_top_tokens=1,
    )

    assert reuse_mask.tolist() == [[True, True, False, False]]
    assert torch.isfinite(selected_scores[0, 0])
    assert torch.isfinite(selected_scores[0, 1])
    assert not torch.isfinite(selected_scores[0, 2])


def test_build_layer_reuse_masks_follows_schedule():
    selected_scores = torch.tensor([[0.9, 0.8, float("-inf"), float("-inf")]], dtype=torch.float32)
    always_reuse_mask = torch.tensor([[False, False, True, False]])
    layer_schedule = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    masks = build_layer_reuse_masks(selected_scores, always_reuse_mask, layer_schedule)
    assert len(masks) == 3
    assert masks[0].tolist() == [[False, False, True, False]]
    assert masks[1].sum().item() == 2
    assert masks[2].tolist() == [[True, True, True, False]]


def test_compute_vla_cache_layer_schedule_returns_unit_interval():
    attentions = [
        torch.full((1, 2, 3, 3), 1 / 3, dtype=torch.float32),
        torch.tensor([[[[1.0, 0.0, 0.0]] * 3] * 2], dtype=torch.float32),
    ]
    schedule = compute_vla_cache_layer_schedule(attentions)
    assert schedule is not None
    assert schedule.shape == (2,)
    assert torch.all(schedule >= 0.0)
    assert torch.all(schedule <= 1.0)
