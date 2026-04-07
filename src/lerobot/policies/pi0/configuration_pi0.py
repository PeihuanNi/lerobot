#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES

DEFAULT_IMAGE_SIZE = 224

ACE_CONFIG_FIELD_ALIASES = {
    "interp_denoise_step": "ace_denoise_step",
    "interp_use_residual": "ace_use_residual",
    "interp_action_start": "ace_action_start",
    "interp_action_end": "ace_action_end",
    "interp_variant": "ace_variant",
    "interp_objective": "ace_objective",
    "interp_plot_actions_l1": "ace_plot_actions_l1",
}


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # Token selection / pruning (optional, inference-time)
    token_selection_enabled: bool = False
    # ═══════════════════════════════════════════════════════════════════════
    # 1. Scoring Method
    #    Determines how vision-token importance is computed.
    #    Each method outputs per-token scores; downstream modules are unaffected.
    # ═══════════════════════════════════════════════════════════════════════
    grad_score_method: str = "ace"
    # Supported: ace | grad_only | attn_only
    # Aliases: action_vision -> attn_only
    # Legacy values are still accepted during config loading for compatibility.
    score_attn_source: str = "action_vision"
    # "action_vision" = diffusion expert action queries -> vision keys
    # "vision_text"   = prefix language queries -> vision keys

    # -- ACE params --
    ace_denoise_step: int = -1  # -1 = avg all denoise steps; >=0 = specific step
    ace_use_residual: bool = False  # True = full Chefer residual propagation across all layers
    ace_action_start: int = 5       # first action step for objective (0-indexed)
    ace_action_end: int = 9        # last action step (exclusive); -1 = chunk_size (all)
    ace_variant: str = "original"   # "original" = ReLU | "abs_heads" = mean_h(|g*A|) | "direct" = signed mean_h(g*A) | "dimension_independent" = per-dim backprop
    ace_objective: str = "vector_field_L2"  # "action_sample_L1" | "action_sample_L2" | "vector_field_L2"
    ace_plot_actions_l1: bool = False  # save per-episode action L1 norm curve

    # -- Shared attention params --
    attn_num_layers: int = 1       # avg attention over last N Expert layers (1=last only, 18=all)
    attn_num_denoise_steps: int = 1  # avg attention over last N denoise steps (1=last only)
    attn_score_beta: float = 1.0
    grad_head_beta: float = 1.0      # legacy unused field kept for config compatibility
    grad_head_norm: str = "sum"      # legacy unused field kept for config compatibility
    grad_action_agg: str = "sum"     # sum | max — action-step & head aggregation

    # -- legacy full_grad / partial_grad params --
    partial_grad_phi: str = "l2"           # legacy unused field kept for config compatibility
    partial_grad_pos_weight: float = 1.0   # legacy unused field kept for config compatibility
    partial_grad_grip_weight: float = 2.0  # legacy unused field kept for config compatibility

    # -- General scoring params --
    grad_denoise_steps: int = 1           # legacy unused field kept for config compatibility
    grad_tau: float = 0.1                 # legacy unused field kept for config compatibility
    grad_alpha: float = 1.0               # legacy unused field kept for config compatibility
    grad_beta: float = 1.0                # legacy unused field kept for config compatibility
    grad_region_ema: float = 0.0  # EMA smoothing for region scores (0=none)
    grad_keep_prev: bool = False

    # Legacy background-filtering fields kept for config compatibility.
    static_bg_enabled: bool = False
    static_bg_threshold: float = 0.95
    static_bg_camera_idx: int = 0
    static_bg_score_gate: float = 0.3

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Pruning Ratio
    #    Controls *how many* tokens to keep after scoring.
    #
    #    Runtime now hardcodes fixed-count TopK. The fields below are kept only
    #    so older config.json files still parse cleanly.
    # ═══════════════════════════════════════════════════════════════════════
    pruning_mode: str = "fixed_count"  # legacy unused field kept for config compatibility

    # Legacy mass-based pruning fields kept for config compatibility.
    grad_region_mass: float = 0.7
    pruning_method: str = "topk_ratio"
    mass_low: float = 0.5
    mass_high: float = 0.95

    # Legacy entropy-dynamic fields kept for config compatibility.
    keep_ratio_low: float = 0.5
    keep_ratio_high: float = 0.9

    # -- Shared: min/max guardrails (applied in ALL pruning modes) --
    min_kept_tokens: int = 0
    max_kept_tokens: int = 1_000_000

    # -- Global Active Token Pool --
    # Ratio of the *previously kept* tokens to permanently discard in the next eval frame.
    # Discards the highest-scoring tokens from the previous frame.
    discard_prev_kept_ratio: float = 0.0
    # "top" = discard highest-scoring; "bottom" = discard lowest-scoring;
    # "middle" = discard mid-scoring; "random" = randomly discard previous kept regions
    discard_mode: str = "top"
    # If enabled, a gripper-close action schedules the global discard pool to be
    # fully restored on the next eval frame.
    reset_discard_pool_on_gripper_close: bool = False
    gripper_close_threshold: float = 0.0

    # -- Dynamic Prune Mode --
    #    Controls how kept_tokens adapts at runtime:
    #      "none"          — fixed prune_ratio for both min and max
    #      "l1_threshold"  — binary switch: L1 > threshold → min_kept, else → max_kept
    #      "ema"           — continuous: sigmoid((L - L_hat) / L_hat) → interpolate [max, min] (or reverse)
    #      "accel"         — chunk-internal xyz/rot acceleration, each using half of the [min, max] span
    #      "velocity"      — chunk-internal xyz/rot action magnitude, each using half of the [min, max] span
    dynamic_prune_mode: str = "none"  # "none" | "l1_threshold" | "ema" | "accel" | "velocity"
    prune_ratio: int = 64  # fixed kept-token count (when dynamic_prune_mode="none")

    # -- l1_threshold params --
    l1_dynamic_prune_threshold: float = 8.0

    # -- ema / accel / velocity params --
    l1_ema_alpha: float = 0.7        # EMA coefficient: higher = more smoothing / slower response
    ema_sigmoid_gain: float = 4.0    # legacy unused field kept for config compatibility
    accel_sigmoid_gain: float = 4.0  # legacy unused field kept for config compatibility
    l1_ema_threshold: float = 8.0    # legacy field kept for compatibility
    l1_ema_temperature: float = 2.0  # legacy field kept for compatibility
    l1_ema_adaptive: bool = False    # legacy field kept for compatibility
    l1_ema_adaptive_k: float = 1.0   # legacy field kept for compatibility
    # Shared direction for ema / accel / velocity:
    # "normal"  = high deviation / motion → fewer tokens (toward MIN)
    # "reverse" = high deviation / motion → more tokens (toward MAX)
    ema_direction: str = "normal"

    region_patch_size: int = 1
    token_prune_enabled: bool = False
    token_temporal_threshold: float = 0.9  # legacy unused field kept for config compatibility
    token_spatial_threshold: float = 0.9   # legacy unused field kept for config compatibility
    token_spatial_radius: int = 1          # legacy unused field kept for config compatibility
    mask_dilation_radius: int = 0          # legacy unused field kept for config compatibility
    vision_partial_update_enabled: bool = False  # legacy unused field kept for config compatibility

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Eval Interval
    # ═══════════════════════════════════════════════════════════════════════
    region_eval_interval: int = 1
    dynamic_eval_enabled: bool = False    # legacy unused field kept for config compatibility
    eval_entropy_threshold: float = 3.5   # legacy unused field kept for config compatibility
    max_eval_interval: int = 5            # legacy unused field kept for config compatibility

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Overlay / Visualization
    #    overlay_mode selects rendering style:
    #      "heatmap" — continuous jet colormap; pruned regions darkened+hatched
    #      "label"   — discrete 5-color overlay (green/yellow/red/blue/transparent)
    #    score_debug_heatmap overrides everything: forces every-frame scoring,
    #    disables pruning, shows pure heatmap without prune marks.
    # ═══════════════════════════════════════════════════════════════════════
    overlay_mode: str = "heatmap"       # "heatmap" | "label"
    overlay_show_scores: bool = False
    overlay_show_ids: bool = False
    score_debug_heatmap: bool = False   # debug: no pruning, every frame scored

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Logging / Debug
    # ═══════════════════════════════════════════════════════════════════════
    rollout_dir: str | None = None
    local_log_dir: str | None = None
    run_id_note: str = ""
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None
    seed: int | None = None

    # Action decoding options
    use_l1_regression: bool = False
    use_diffusion: bool = True

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    @classmethod
    def _migrate_pretrained_config_dict(cls, config: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(config)
        if migrated.get("grad_score_method") == "transformer_interpretability":
            migrated["grad_score_method"] = "ace"
        for old_key, new_key in ACE_CONFIG_FIELD_ALIASES.items():
            if old_key in migrated and new_key not in migrated:
                migrated[new_key] = migrated[old_key]
            migrated.pop(old_key, None)
        return migrated

    @classmethod
    def _translate_cli_overrides(cls, cli_overrides: list[str]) -> list[str]:
        translated = []
        for arg in cli_overrides:
            new_arg = arg
            for old_key, new_key in ACE_CONFIG_FIELD_ALIASES.items():
                new_arg = new_arg.replace(f"--{old_key}=", f"--{new_key}=")
                new_arg = new_arg.replace(f".{old_key}=", f".{new_key}=")
            if "grad_score_method" in new_arg and "transformer_interpretability" in new_arg:
                new_arg = new_arg.replace("transformer_interpretability", "ace")
            translated.append(new_arg)
        return translated

    def __post_init__(self):
        super().__post_init__()

        grad_method_aliases = {
            "transformer_interpretability": "ace",
            "transformer-interpretability": "ace",
            "action-vision": "attn_only",
            "action_vision": "attn_only",
            "attn-only": "attn_only",
            "grad-only": "grad_only",
        }
        ace_variant_aliases = {
            "relu": "original",
            "ReLU": "original",
            "abs": "abs_heads",
            "raw": "direct",
            "none": "direct",
        }
        self.grad_score_method = grad_method_aliases.get(self.grad_score_method, self.grad_score_method)
        self.ace_variant = ace_variant_aliases.get(self.ace_variant, self.ace_variant)

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.token_selection_enabled:
            valid_grad_methods = {
                "full_grad",
                "partial_grad",
                "attn_only",
                "grad_only",
                "ace",
            }
            if self.grad_score_method not in valid_grad_methods:
                raise ValueError(f"Invalid grad_score_method: {self.grad_score_method}")
            valid_score_attn_sources = {"action_vision", "vision_text"}
            if self.score_attn_source not in valid_score_attn_sources:
                raise ValueError(f"Invalid score_attn_source: {self.score_attn_source}")
            valid_ace_variants = {"original", "abs_heads", "direct", "dimension_independent"}
            if self.ace_variant not in valid_ace_variants:
                raise ValueError(f"Invalid ace_variant: {self.ace_variant}")
            valid_phi = {"l1", "l2"}
            if self.partial_grad_phi not in valid_phi:
                raise ValueError(f"Invalid partial_grad_phi: {self.partial_grad_phi}")
            if self.grad_score_method == "partial_grad":
                valid_head_norm = {"sum", "max"}
                if self.grad_head_norm not in valid_head_norm:
                    raise ValueError(f"Invalid grad_head_norm: {self.grad_head_norm}")
                valid_action_agg = {"sum", "max"}
                if self.grad_action_agg not in valid_action_agg:
                    raise ValueError(f"Invalid grad_action_agg: {self.grad_action_agg}")
            valid_overlay_modes = {"heatmap", "label"}
            if self.overlay_mode not in valid_overlay_modes:
                raise ValueError(f"Invalid overlay_mode: {self.overlay_mode}")
            valid_discard_modes = {"top", "bottom", "middle", "random"}
            if self.discard_mode not in valid_discard_modes:
                raise ValueError(f"Invalid discard_mode: {self.discard_mode}")
            if self.region_patch_size <= 0:
                raise ValueError("region_patch_size must be > 0")
            if self.dynamic_eval_enabled and self.max_eval_interval <= 0:
                raise ValueError("max_eval_interval must be > 0")
            valid_dynamic_prune_modes = {"none", "l1_threshold", "ema", "accel", "velocity"}
            if self.dynamic_prune_mode not in valid_dynamic_prune_modes:
                raise ValueError(f"Invalid dynamic_prune_mode: {self.dynamic_prune_mode}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
