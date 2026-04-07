import numpy as np

from src.lerobot.ace import AceInferenceRunner


def make_random_image(height: int = 256, width: int = 256) -> np.ndarray:
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


POLICY_PATH = "/home/nipeihuan/models/pi05_libero_finetuned"
POLICY_OVERRIDES = {
    "compile_model": False,
    "n_action_steps": 10,
    "use_l1_regression": False,
    "use_diffusion": True,
    "num_inference_steps": 10,
    "token_selection_enabled": True,
    "grad_score_method": "ace",
    "score_attn_source": "action_vision",
    "ace_denoise_step": 5,
    "ace_use_residual": True,
    "ace_action_start": 5,
    "ace_action_end": 10,
    "ace_variant": "abs_heads",
    "ace_objective": "action_sample_L2",
    "ace_plot_actions_l1": False,
    "attn_num_layers": 6,
    "attn_num_denoise_steps": 3,
    "attn_score_beta": 2.0,
    "grad_action_agg": "max",
    "grad_region_ema": 0.0,
    "grad_keep_prev": False,
    "min_kept_tokens": 64,
    "max_kept_tokens": 128,
    "discard_prev_kept_ratio": 0.1,
    "discard_mode": "middle",
    "reset_discard_pool_on_gripper_close": True,
    "gripper_close_threshold": 0.0,
    "dynamic_prune_mode": "accel",
    "l1_dynamic_prune_threshold": 9.0,
    "l1_ema_alpha": 0.7,
    "ema_direction": "normal",
    "prune_ratio": 96,
    "region_patch_size": 1,
    "token_prune_enabled": True,
    "region_eval_interval": 2,
    "overlay_mode": "heatmap",
    "overlay_show_scores": False,
    "overlay_show_ids": False,
    "score_debug_heatmap": False,
    "rollout_dir": "",
    "local_log_dir": "",
    "run_id_note": "normal",
    "use_wandb": False,
    "wandb_entity": "",
    "wandb_project": "",
}


runner = AceInferenceRunner(
    POLICY_PATH,
    policy_overrides=POLICY_OVERRIDES,
)

image1 = make_random_image()
image2 = make_random_image()

action = runner.predict_action(
    images=[image1, image2],
    instruction="open the drawer",
)

chunk = runner.predict_action_chunk(
    images=[image1, image2],
    instruction="open the drawer",
)

print("image1 shape:", image1.shape, image1.dtype)
print("image2 shape:", image2.shape, image2.dtype)
print("action shape:", action.shape)
print("action:", action)
print("chunk shape:", chunk.shape)
print("action chunk:")
print(chunk)
