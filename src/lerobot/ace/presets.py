from __future__ import annotations

from typing import Any

DEFAULT_SPVLA_ENV_TASK = "libero_object"

_COMMON_POLICY_OVERRIDES: dict[str, Any] = {
    "compile_model": False,
    "n_action_steps": 10,
    "use_l1_regression": False,
    "use_diffusion": True,
    "num_inference_steps": 10,
    "schedule_buffer_size": 6,
    "schedule_ratio_threshold": 0.5,
    "schedule_velocity_min": 0.04,
    "schedule_velocity_max": 0.20,
    "schedule_generator_reg_lambda": 0.0001,
    "schedule_validity_max_scale": 2.5,
    "schedule_warmup_vla_steps": 2,
    "schedule_reference_min_generated_ratio": 0.6666667,
    "reference_action_clip_min": -0.9,
    "reference_action_clip_max": 0.9,
    "semantic_score_mass": 0.7,
    "spatial_edge_enabled": True,
    "canny_low_threshold": 100,
    "canny_high_threshold": 200,
    "region_patch_size": 1,
    "attn_num_layers": 6,
    "attn_num_denoise_steps": 3,
    "attn_score_beta": 2.0,
    "grad_action_agg": "max",
    "grad_region_ema": 0.0,
    "grad_keep_prev": False,
    "discard_prev_kept_ratio": 0.5,
    "discard_mode": "middle",
    "reset_discard_pool_on_gripper_close": False,
    "gripper_close_threshold": 0.0,
    "overlay_mode": "heatmap",
    "overlay_show_scores": False,
    "overlay_show_ids": False,
    "score_debug_heatmap": False,
    "use_wandb": False,
    "wandb_entity": "",
    "wandb_project": "",
}

_SPVLA_ENV_TASK_DEFAULTS: dict[str, dict[str, float | int]] = {
    "libero_spatial": {
        "step_skip": 1,
        "z_xy_rate_skip": 0.4,
        "z_max_skip": 0.3,
        "z_thre_prune": 0.5,
        "z_min_prune": 0.9,
    },
    "libero_object": {
        "step_skip": 2,
        "z_xy_rate_skip": 0.6,
        "z_max_skip": 0.5,
        "z_thre_prune": 0.5,
        "z_min_prune": 0.9,
    },
    "libero_goal": {
        "step_skip": 2,
        "z_xy_rate_skip": 1.2,
        "z_max_skip": 0.3,
        "z_thre_prune": 0.3,
        "z_min_prune": 0.7,
    },
    "libero_10": {
        "step_skip": 2,
        "z_xy_rate_skip": 1.0,
        "z_max_skip": 0.3,
        "z_thre_prune": 0.3,
        "z_min_prune": 0.85,
    },
    "default": {
        "step_skip": 1,
        "z_xy_rate_skip": 0.6,
        "z_max_skip": 0.3,
        "z_thre_prune": 0.5,
        "z_min_prune": 0.9,
    },
}

_ACE_MODE_OVERRIDES: dict[str, Any] = {
    "model_scheduling_enabled": False,
    "spvla_reference_enabled": False,
    "token_selection_enabled": True,
    "token_prune_enabled": True,
    "grad_score_method": "ace",
    "dynamic_prune_mode": "accel",
    "region_eval_interval": 2,
    "prune_ratio": 64,
    "min_kept_tokens": 64,
    "max_kept_tokens": 128,
    "prune_velocity_min": 0.04,
    "prune_velocity_max": 0.20,
    "l1_dynamic_prune_threshold": 9.0,
    "l1_ema_alpha": 0.7,
    "ema_direction": "normal",
    "ace_denoise_step": 5,
    "ace_use_residual": True,
    "ace_action_start": 5,
    "ace_action_end": 9,
    "ace_variant": "dimension_independent",
    "ace_objective": "action_sample_L1",
    "ace_plot_actions_l1": False,
}

_SPVLA_PAPER_ACC_OVERRIDES: dict[str, Any] = {
    "model_scheduling_enabled": True,
    "spvla_reference_enabled": False,
    "token_selection_enabled": True,
    "token_prune_enabled": True,
    "grad_score_method": "attn_only",
    "dynamic_prune_mode": "velocity",
    "region_eval_interval": 2,
    "prune_ratio": 64,
    "min_kept_tokens": 64,
    "max_kept_tokens": 128,
    "prune_velocity_min": 0.04,
    "prune_velocity_max": 0.20,
    "l1_dynamic_prune_threshold": 9.0,
    "l1_ema_alpha": 0.7,
    "ema_direction": "normal",
    "ace_denoise_step": 5,
    "ace_use_residual": True,
    "ace_action_start": 5,
    "ace_action_end": 9,
    "ace_variant": "dimension_independent",
    "ace_objective": "action_sample_L1",
    "ace_plot_actions_l1": False,
}

_SPVLA_PAPER_SPEED_OVERRIDES: dict[str, Any] = {
    **_SPVLA_PAPER_ACC_OVERRIDES,
    "min_kept_tokens": 48,
    "max_kept_tokens": 96,
}

_SPVLA_REFERENCE_OVERRIDES: dict[str, Any] = {
    "model_scheduling_enabled": True,
    "spvla_reference_enabled": True,
    "token_selection_enabled": True,
    "token_prune_enabled": True,
    "grad_score_method": "spvla_reference",
    "dynamic_prune_mode": "none",
    "region_eval_interval": 1,
    "prune_ratio": 64,
    "min_kept_tokens": 0,
    "max_kept_tokens": 1000000,
    "prune_velocity_min": 0.04,
    "prune_velocity_max": 0.20,
    "l1_dynamic_prune_threshold": 9.0,
    "l1_ema_alpha": 0.7,
    "ema_direction": "normal",
    "ace_denoise_step": 5,
    "ace_use_residual": True,
    "ace_action_start": 5,
    "ace_action_end": 9,
    "ace_variant": "dimension_independent",
    "ace_objective": "action_sample_L1",
    "ace_plot_actions_l1": False,
}

_SPVLA_SCHEDULE_ONLY_OVERRIDES: dict[str, Any] = {
    "model_scheduling_enabled": True,
    "spvla_reference_enabled": True,
    "token_selection_enabled": False,
    "token_prune_enabled": False,
    "grad_score_method": "spvla_reference",
    "dynamic_prune_mode": "none",
    "region_eval_interval": 1,
    "prune_ratio": 64,
    "min_kept_tokens": 128,
    "max_kept_tokens": 128,
}

_SPVLA_TOKEN_ONLY_OVERRIDES: dict[str, Any] = {
    "model_scheduling_enabled": False,
    "spvla_reference_enabled": True,
    "token_selection_enabled": True,
    "token_prune_enabled": True,
    "grad_score_method": "spvla_reference",
    "dynamic_prune_mode": "none",
    "region_eval_interval": 1,
    "prune_ratio": 64,
    "min_kept_tokens": 0,
    "max_kept_tokens": 1000000,
    "prune_velocity_min": 0.04,
    "prune_velocity_max": 0.20,
}

_BASELINE_MODE_OVERRIDES: dict[str, Any] = {
    "model_scheduling_enabled": False,
    "spvla_reference_enabled": False,
    "token_selection_enabled": False,
    "token_prune_enabled": False,
    "grad_score_method": "ace",
    "dynamic_prune_mode": "none",
    "region_eval_interval": 1,
    "prune_ratio": 64,
    "min_kept_tokens": 128,
    "max_kept_tokens": 128,
}

_CANONICAL_POLICY_MODES = (
    "ace",
    "spvla_paper_acc",
    "spvla_paper_speed",
    "spvla_reference",
    "spvla_schedule_only",
    "spvla_token_only",
    "baseline",
)

_POLICY_MODE_ALIASES = {
    "spvla": "spvla_paper_acc",
    "sp_vla": "spvla_paper_acc",
    "sp-vla": "spvla_paper_acc",
    "paper_acc": "spvla_paper_acc",
    "paper_speed": "spvla_paper_speed",
    "reference": "spvla_reference",
    "schedule_only": "spvla_schedule_only",
    "token_only": "spvla_token_only",
}

_POLICY_PRESET_DESCRIPTIONS = {
    "ace": "ACE scoring + token pruning with accel-based keep-token scheduling.",
    "spvla_paper_acc": "SP-VLA paper accuracy preset: attn_only scoring + velocity pruning + model scheduling.",
    "spvla_paper_speed": "SP-VLA paper speed preset: more aggressive token budget than paper_acc.",
    "spvla_reference": "SP-VLA reference preset: reference scoring/scheduling path with no token-budget pruning.",
    "spvla_schedule_only": "SP-VLA scheduling only: enable lightweight reference scheduling without token selection/pruning.",
    "spvla_token_only": "SP-VLA token-only: reference scoring with token pruning but no model scheduling.",
    "baseline": "No scheduling, no pruning. Plain VLA baseline with chunked action generation.",
}


def list_policy_modes() -> tuple[str, ...]:
    return _CANONICAL_POLICY_MODES


def describe_policy_modes() -> dict[str, str]:
    return dict(_POLICY_PRESET_DESCRIPTIONS)


def get_policy_preset(mode: str, *, env_task: str = DEFAULT_SPVLA_ENV_TASK) -> dict[str, Any]:
    return resolve_policy_mode(mode, env_task=env_task)


def get_all_policy_presets(*, env_task: str = DEFAULT_SPVLA_ENV_TASK) -> dict[str, dict[str, Any]]:
    return {
        mode: get_policy_preset(mode, env_task=env_task)
        for mode in _CANONICAL_POLICY_MODES
    }


def resolve_policy_mode(mode: str | None, *, env_task: str = DEFAULT_SPVLA_ENV_TASK) -> dict[str, Any]:
    if mode is None:
        return {}

    normalized_mode = _normalize_policy_mode(mode)
    env_defaults = _resolve_spvla_env_defaults(env_task)
    resolved = dict(_COMMON_POLICY_OVERRIDES)

    if normalized_mode == "ace":
        resolved.update(_ACE_MODE_OVERRIDES)
    elif normalized_mode == "spvla_paper_acc":
        resolved.update(env_defaults)
        resolved.update(_SPVLA_PAPER_ACC_OVERRIDES)
    elif normalized_mode == "spvla_paper_speed":
        resolved.update(env_defaults)
        resolved.update(_SPVLA_PAPER_SPEED_OVERRIDES)
    elif normalized_mode == "spvla_reference":
        resolved.update(env_defaults)
        resolved.update(_SPVLA_REFERENCE_OVERRIDES)
    elif normalized_mode == "spvla_schedule_only":
        resolved.update(env_defaults)
        resolved.update(_SPVLA_SCHEDULE_ONLY_OVERRIDES)
    elif normalized_mode == "spvla_token_only":
        resolved.update(env_defaults)
        resolved.update(_SPVLA_TOKEN_ONLY_OVERRIDES)
    elif normalized_mode == "baseline":
        resolved.update(_BASELINE_MODE_OVERRIDES)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported policy mode '{mode}'.")

    return resolved


def normalize_policy_mode(mode: str) -> str:
    return _normalize_policy_mode(mode)


def _normalize_policy_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    normalized = _POLICY_MODE_ALIASES.get(normalized, normalized)
    if normalized not in _CANONICAL_POLICY_MODES:
        supported = ", ".join(_CANONICAL_POLICY_MODES)
        raise ValueError(f"Unsupported policy mode '{mode}'. Supported modes: {supported}.")
    return normalized


def _resolve_spvla_env_defaults(env_task: str) -> dict[str, float | int]:
    normalized_task = "libero_10" if env_task == "libero_long" else env_task
    if normalized_task in _SPVLA_ENV_TASK_DEFAULTS:
        return dict(_SPVLA_ENV_TASK_DEFAULTS[normalized_task])
    return dict(_SPVLA_ENV_TASK_DEFAULTS["default"])
