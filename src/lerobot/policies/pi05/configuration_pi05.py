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
    grad_score_method: str = "full_grad"  # full_grad | partial_grad | attn_only
    grad_denoise_steps: int = 1
    attn_score_beta: float = 1.0
    attn_num_layers: int = 1  # number of last Expert layers to average attention over (1 = last layer only)
    attn_num_denoise_steps: int = 1  # number of last denoise steps to average attention over (1 = last step only)
    grad_head_beta: float = 1.0
    grad_head_norm: str = "sum"  # sum | max
    grad_action_agg: str = "sum"  # sum | max — how to aggregate scores over action steps and heads
    partial_grad_phi: str = "l2"  # l1 | l2
    partial_grad_pos_weight: float = 1.0
    partial_grad_grip_weight: float = 2.0
    grad_tau: float = 0.1
    grad_alpha: float = 1.0
    grad_beta: float = 1.0
    grad_region_mass: float = 0.25
    grad_region_ema: float = 0.0
    grad_keep_prev: bool = False
    # Dynamic evaluation interval (entropy-based)
    dynamic_eval_enabled: bool = False  # if True, use entropy-based adaptive eval interval
    eval_entropy_threshold: float = 3.5  # trigger re-eval when score entropy exceeds this
    max_eval_interval: int = 5  # hard-cap fallback: re-eval at least every N frames

    # ── Dynamic mass: entropy-coupled pruning aggressiveness ─────────────────
    # When enabled, grad_region_mass becomes a dynamic value that scales with
    # the normalised entropy h ∈ [0,1] of the current region score distribution:
    #   effective_mass = mass_low + h * (mass_high - mass_low)
    # Low h (focused) → mass_low → aggressive pruning
    # High h (confused) → mass_high → conservative, keep more tokens
    # grad_region_mass still acts as static fallback when dynamic_mass_enabled=False
    dynamic_mass_enabled: bool = False
    mass_low: float = 0.5   # effective mass when attention is fully focused  (h=0)
    mass_high: float = 0.95  # effective mass when attention is fully diffuse (h=1)

    # ── Pruning method selection ─────────────────────────────────────────────
    # "mass"       — cumulative-score threshold (original); sensitive to score
    #                distribution shape — see mass_low / mass_high above.
    # "topk_ratio" — keep a fixed fraction of regions by score rank.  Linear,
    #                no cliff-like jumps.  Use keep_ratio_low / keep_ratio_high
    #                below (coupled to entropy h, like mass_low / mass_high).
    pruning_method: str = "topk_ratio"  # "mass" | "topk_ratio"

    # ── TopK-ratio: entropy-coupled keep-ratio ───────────────────────────────
    # effective_keep_ratio = keep_ratio_low + h * (keep_ratio_high - keep_ratio_low)
    # h=0 (focused, stable) → keep_ratio_low → more aggressive pruning
    # h=1 (diffuse, confused) → keep_ratio_high → conservative, keep more
    keep_ratio_low: float = 0.5    # keep 50% when attention is focused
    keep_ratio_high: float = 0.9   # keep 90% when attention is diffuse
    region_eval_interval: int = 1
    region_patch_size: int = 1
    token_temporal_threshold: float = 0.9
    token_spatial_threshold: float = 0.9
    token_spatial_radius: int = 1
    token_prune_enabled: bool = False

    # ── Static background detection (cosine similarity to first frame) ───────
    # When enabled, regions in the specified camera whose cosine similarity to
    # the first-frame reference exceeds ``static_bg_threshold`` are treated as
    # static background and forcibly pruned on every frame (eval + non-eval).
    static_bg_enabled: bool = False
    static_bg_threshold: float = 0.95  # cosine similarity threshold (higher = stricter)
    static_bg_camera_idx: int = 0      # image slot index to apply bg detection (0 = agentview)
    static_bg_score_gate: float = 0.3    # protect top (1-gate) scored regions; 1.0 = no protection

    # Fixed-count mode (used when dynamic_mass_enabled=False):
    # When min == max, exactly that many tokens are kept per image.
    # Ignored when dynamic_mass_enabled=True (mass controls how many to keep).
    min_kept_tokens: int = 0
    max_kept_tokens: int = 1_000_000
    vision_partial_update_enabled: bool = False
    overlay_show_scores: bool = False
    overlay_show_ids: bool = False
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

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

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
            valid_grad_methods = {"full_grad", "partial_grad", "attn_only"}
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
