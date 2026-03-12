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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
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
    num_inference_steps: int = 10
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
    grad_score_method: str = "full_grad"
    # full_grad | partial_grad | attn_only | transformer_interpretability

    # -- transformer_interpretability params --
    interp_denoise_step: int = -1  # -1 = avg all denoise steps; >=0 = specific step
    interp_use_residual: bool = False  # True = full Chefer residual propagation across all layers
    interp_action_start: int = 0       # first action step for objective (0-indexed)
    interp_action_end: int = -1        # last action step (exclusive); -1 = chunk_size (all)
    interp_variant: str = "original"   # "original" = ReLU | "abs_heads" = mean_h(|g*A|) | "dimension_independent" = per-dim backprop
    interp_objective: str = "vector_field_L2"
    interp_plot_actions_l1: bool = False  # save per-episode action L1 norm curve

    # -- Shared attention params --
    attn_num_layers: int = 1       # avg attention over last N Expert layers (1=last only, 18=all)
    attn_num_denoise_steps: int = 1  # avg attention over last N denoise steps (1=last only)
    attn_score_beta: float = 1.0
    grad_head_beta: float = 1.0
    grad_head_norm: str = "sum"      # sum | max
    grad_action_agg: str = "sum"     # sum | max — action-step & head aggregation

    # -- partial_grad params --
    partial_grad_phi: str = "l2"           # l1 | l2
    partial_grad_pos_weight: float = 1.0
    partial_grad_grip_weight: float = 2.0

    # -- General scoring params --
    grad_denoise_steps: int = 1
    grad_tau: float = 0.1
    grad_alpha: float = 1.0
    grad_beta: float = 1.0
    grad_region_ema: float = 0.0  # EMA smoothing for region scores (0=none)
    grad_keep_prev: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Background Filtering  (independent of scoring & pruning ratio)
    #    Cosine-similarity to first-frame reference → static background mask.
    #    Applied per-camera before pruning; protected by score_gate.
    # ═══════════════════════════════════════════════════════════════════════
    static_bg_enabled: bool = False
    static_bg_threshold: float = 0.95   # cosine sim threshold (higher = stricter)
    static_bg_camera_idx: int = 0       # image slot (0 = agentview, 1 = wrist)
    static_bg_score_gate: float = 0.3   # protect top (1-gate) scored regions (1.0 = no protection)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Pruning Ratio  (independent of scoring method & background filter)
    #    Controls *how many* tokens to keep after scoring.
    #
    #    pruning_mode selects the strategy:
    #      "fixed_count"      — keep TopK by score; count set by min/max_kept_tokens.
    #      "cumulative_mass"  — normalize scores to sum=1, keep until cumulative
    #                           sum >= grad_region_mass.  min/max still enforced.
    #      "entropy_dynamic"  — entropy-coupled: h∈[0,1] interpolates between
    #                           low/high params.  pruning_method picks "mass" or
    #                           "topk_ratio".  min/max still enforced.
    # ═══════════════════════════════════════════════════════════════════════
    pruning_mode: str = "fixed_count"  # "fixed_count" | "cumulative_mass" | "entropy_dynamic"

    # -- cumulative_mass params --
    grad_region_mass: float = 0.7  # cumulative-score threshold (used when pruning_mode="cumulative_mass")

    # -- entropy_dynamic params --
    pruning_method: str = "topk_ratio"   # "mass" | "topk_ratio"  (used when pruning_mode="entropy_dynamic")
    keep_ratio_low: float = 0.5          # keep ratio when h≈0 (focused)
    keep_ratio_high: float = 0.9         # keep ratio when h≈1 (diffuse)
    mass_low: float = 0.5               # mass threshold when h≈0
    mass_high: float = 0.95             # mass threshold when h≈1

    # -- Shared: min/max guardrails (applied in ALL pruning modes) --
    min_kept_tokens: int = 0
    max_kept_tokens: int = 1_000_000

    # -- L1 Dynamic Prune Ratio --
    #    When enabled, the pruning count switches between min_kept_tokens and
    #    max_kept_tokens based on the previous frame's action L1 norm:
    #      L1 > threshold → aggressive prune (min_kept_tokens)
    #      L1 ≤ threshold → conservative prune (max_kept_tokens)
    #    When disabled, a fixed prune_ratio is used for both min/max.
    l1_dynamic_prune_enabled: bool = False
    l1_dynamic_prune_threshold: float = 8.0
    prune_ratio: int = 64  # fixed kept-token count when l1_dynamic_prune is disabled

    region_patch_size: int = 1
    token_temporal_threshold: float = 0.9
    token_spatial_threshold: float = 0.9
    token_spatial_radius: int = 1
    token_prune_enabled: bool = False
    vision_partial_update_enabled: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Eval Interval  (independent of everything above)
    #    Controls how often the scoring + pruning pipeline runs.
    #      dynamic_eval_enabled=False → fixed interval (region_eval_interval)
    #      dynamic_eval_enabled=True  → entropy-triggered re-eval
    # ═══════════════════════════════════════════════════════════════════════
    region_eval_interval: int = 1
    dynamic_eval_enabled: bool = False
    eval_entropy_threshold: float = 3.5  # re-eval trigger threshold
    max_eval_interval: int = 5           # hard-cap: re-eval at least every N frames

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Overlay / Visualization  (independent of scoring & pruning)
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
    # 6. Logging / Debug
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

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW`
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


    def __post_init__(self):
        super().__post_init__()

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
            valid_grad_methods = {"full_grad", "partial_grad", "attn_only", "transformer_interpretability"}
            if self.grad_score_method not in valid_grad_methods:
                raise ValueError(f"Invalid grad_score_method: {self.grad_score_method}")
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
            valid_pruning_modes = {"fixed_count", "cumulative_mass", "entropy_dynamic"}
            if self.pruning_mode not in valid_pruning_modes:
                raise ValueError(f"Invalid pruning_mode: {self.pruning_mode}")
            valid_overlay_modes = {"heatmap", "label"}
            if self.overlay_mode not in valid_overlay_modes:
                raise ValueError(f"Invalid overlay_mode: {self.overlay_mode}")
            if self.region_patch_size <= 0:
                raise ValueError("region_patch_size must be > 0")
            if self.dynamic_eval_enabled and self.max_eval_interval <= 0:
                raise ValueError("max_eval_interval must be > 0")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
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
