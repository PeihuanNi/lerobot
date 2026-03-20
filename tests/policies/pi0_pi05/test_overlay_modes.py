#!/usr/bin/env python

import numpy as np

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_eval import _apply_heatmap_overlay


def test_pi0_config_accepts_heatmap_kept_only():
    config = PI0Config(token_selection_enabled=True, overlay_mode="heatmap_kept_only")
    assert config.overlay_mode == "heatmap_kept_only"


def test_pi05_config_accepts_heatmap_kept_only():
    config = PI05Config(token_selection_enabled=True, overlay_mode="heatmap_kept_only")
    assert config.overlay_mode == "heatmap_kept_only"


def test_pi05_config_accepts_heatmap_threshold():
    config = PI05Config(token_selection_enabled=True, overlay_heatmap_threshold=0.3)
    assert config.overlay_heatmap_threshold == 0.3


def test_apply_heatmap_overlay_keep_only_hides_non_kept_regions():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    heatmap = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    keep_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    rendered = _apply_heatmap_overlay(image, heatmap, mask_grid=keep_mask, mask_mode="keep_only")

    assert rendered.shape == image.shape
    assert rendered[0, 0].tolist() != image[0, 0].tolist()
    assert rendered[3, 3].tolist() == image[3, 3].tolist()


def test_apply_heatmap_overlay_threshold_keeps_only_higher_scores():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    heatmap = np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32)

    rendered = _apply_heatmap_overlay(image, heatmap, heatmap_threshold=0.5)

    assert rendered[0, 0].tolist() == image[0, 0].tolist()
    assert rendered[0, 3].tolist() != image[0, 3].tolist()
