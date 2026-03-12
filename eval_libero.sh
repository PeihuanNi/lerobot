CUDA_VISIBLE_DEVICES="$1"  # GPU id(s) passed from CLI, e.g., "0" or "0,1"

POLICY_PATH="/home/nipeihuan/models/pi05_libero_finetuned"
POLICY_TYPE="pi05"
N_ACTION_STEPS=10

MUJOCO_GL="egl"
PYOPENGL_PLATFORM="egl"
EGL_PLATFORM="surfaceless"
MUJOCO_EGL_DEVICE_ID=""
LIBGL_ALWAYS_SOFTWARE=""

ENV_TYPE="libero"
ENV_TASK="libero_spatial, libero_object, libero_goal, libero_10"
ENV_TASK_ID="[0,1,2,3,4,5,6,7,8,9]"
# ENV_TASK="libero_spatial"
# ENV_TASK_ID="[4]"
EVAL_BATCH_SIZE=1
EVAL_N_EPISODES=50
MAX_EPISODES_RENDERED=0             # 0 = no video; >0 = save that many mp4s

USE_L1_REGRESSION=false
USE_DIFFUSION=true
NUM_INFERENCE_STEPS=10

TOKEN_SELECTION_ENABLED=false

# ══════════════════════════════════════════════════════════════════════════════
# 1. Scoring Method
# ══════════════════════════════════════════════════════════════════════════════
GRAD_SCORE_METHOD="transformer_interpretability"
# transformer_interpretability | attn_only | partial_grad | full_grad

# -- transformer_interpretability --
INTERP_DENOISE_STEP=-1            # -1 = avg all denoise steps; >=0 = specific step     ######## -1
INTERP_USE_RESIDUAL=true          # true = full Chefer residual propagation across all layers
INTERP_ACTION_START=0              # first action step for objective (0-indexed)
INTERP_ACTION_END=-1               # last action step (exclusive); -1 = all (chunk_size)
INTERP_VARIANT="dimension_independent"     # "original" = ReLU | "abs_heads" = mean_h(|g*A|) | "dimension_independent" = per-action-dim backprop
INTERP_OBJECTIVE="action_sample_L1"    # action_sample_L1 | vector_field_L2
INTERP_PLOT_ACTIONS_L1=false              # save per-episode action L1 norm curve

# -- Shared attention params --
ATTN_NUM_LAYERS=6                  # avg over last N Expert layers (1=last, 18=all)
ATTN_NUM_DENOISE_STEPS=3           # avg over last N denoise steps (1=last, 10=all)
ATTN_SCORE_BETA=2.0
GRAD_HEAD_BETA=1.0
GRAD_HEAD_NORM="sum"               # sum | max
GRAD_ACTION_AGG="max"              # sum | max

# -- partial_grad --
PARTIAL_GRAD_PHI="l2"
PARTIAL_GRAD_POS_WEIGHT=1.0
PARTIAL_GRAD_GRIP_WEIGHT=1.0

# -- General scoring --
GRAD_DENOISE_STEPS=1
GRAD_TAU=0.1
GRAD_ALPHA=1.0
GRAD_BETA=1.0
GRAD_REGION_EMA=0.0                # EMA smoothing (0=none)
GRAD_KEEP_PREV=false

# ══════════════════════════════════════════════════════════════════════════════
# 2. Background Filtering
# ══════════════════════════════════════════════════════════════════════════════
STATIC_BG_ENABLED=false
STATIC_BG_THRESHOLD=0.95           # cosine sim threshold (higher = stricter)
STATIC_BG_CAMERA_IDX=0             # 0 = agentview, 1 = wrist
STATIC_BG_SCORE_GATE=0.7           # protect top (1-gate) scored regions from bg pruning

# ══════════════════════════════════════════════════════════════════════════════
# 3. Pruning Ratio
#    "fixed_count"      — TopK by score, count = min/max_kept_tokens
#    "cumulative_mass"  — norm scores to sum=1, cumulative threshold + min/max
#    "entropy_dynamic"  — entropy-coupled ratio/mass + min/max
#
#    L1 Dynamic Prune Ratio (L1_DYNAMIC_PRUNE_ENABLED):
#      When ON:  if previous frame's action L1 > threshold → MIN_KEPT_TOKENS
#                otherwise → MAX_KEPT_TOKENS
#      When OFF: fixed pruning with PRUNE_RATIO for both min and max
# ══════════════════════════════════════════════════════════════════════════════
PRUNING_MODE="fixed_count"         # "fixed_count" | "cumulative_mass" | "entropy_dynamic"

# -- cumulative_mass params --
GRAD_REGION_MASS=0.7               # cumulative-score threshold

# -- entropy_dynamic params --
PRUNING_METHOD="topk_ratio"        # "mass" | "topk_ratio"
KEEP_RATIO_LOW=0.5                 # keep 50% when focused (h≈0)
KEEP_RATIO_HIGH=0.7                # keep 70% when confused (h≈1)
MASS_LOW=0.5
MASS_HIGH=0.95

# -- Shared: min/max guardrails (always applied) --
#    L1_DYNAMIC_PRUNE_ENABLED=true  → use MIN/MAX_KEPT_TOKENS dynamically
#    L1_DYNAMIC_PRUNE_ENABLED=false → use PRUNE_RATIO (fixed, equal min=max)
L1_DYNAMIC_PRUNE_ENABLED=true     # toggle L1 dynamic prune ratio
L1_DYNAMIC_PRUNE_THRESHOLD=8.0     # L1 threshold: above → MIN, below → MAX
PRUNE_RATIO=64                     # fixed kept-token count (when L1 dynamic OFF)
MIN_KEPT_TOKENS=64                 # aggressive (fewer kept, when L1 > threshold)
MAX_KEPT_TOKENS=128                 # conservative (more kept, when L1 ≤ threshold)

REGION_PATCH_SIZE=1
TOKEN_TEMPORAL_THRESHOLD=0.9
TOKEN_SPATIAL_THRESHOLD=0.9
TOKEN_SPATIAL_RADIUS=1
TOKEN_PRUNE_ENABLED=false
VISION_PARTIAL_UPDATE_ENABLED=false

# ══════════════════════════════════════════════════════════════════════════════
# 4. Eval Interval
#    dynamic_eval_enabled=false → fixed (region_eval_interval)
#    dynamic_eval_enabled=true  → entropy-triggered
# ══════════════════════════════════════════════════════════════════════════════
REGION_EVAL_INTERVAL=2
DYNAMIC_EVAL_ENABLED=false
EVAL_ENTROPY_THRESHOLD=2.3
MAX_EVAL_INTERVAL=5

# ══════════════════════════════════════════════════════════════════════════════
# 5. Overlay / Visualization
#    "heatmap" — continuous jet colormap; pruned regions darkened+hatched
#    "label"   — discrete 5-color overlay
# ══════════════════════════════════════════════════════════════════════════════
OVERLAY_MODE="heatmap"             # "heatmap" | "label"
OVERLAY_SHOW_SCORES=false
OVERLAY_SHOW_IDS=false
SCORE_DEBUG_HEATMAP=false          # debug: no pruning, every frame scored

# ══════════════════════════════════════════════════════════════════════════════
# 6. Logging / Debug
# ══════════════════════════════════════════════════════════════════════════════
ROLLOUT_DIR=""
LOCAL_LOG_DIR=""
RUN_ID_NOTE=""
USE_WANDB=false
WANDB_ENTITY=""
WANDB_PROJECT=""
SEED=7

ARGS=(
  "--policy.n_action_steps=${N_ACTION_STEPS}"
  "--policy.use_l1_regression=${USE_L1_REGRESSION}"
  "--policy.use_diffusion=${USE_DIFFUSION}"
  "--policy.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "--policy.token_selection_enabled=${TOKEN_SELECTION_ENABLED}"
  # 1. Scoring
  "--policy.grad_score_method=${GRAD_SCORE_METHOD}"
  "--policy.interp_denoise_step=${INTERP_DENOISE_STEP}"
  "--policy.interp_use_residual=${INTERP_USE_RESIDUAL}"
  "--policy.interp_action_start=${INTERP_ACTION_START}"
  "--policy.interp_action_end=${INTERP_ACTION_END}"
  "--policy.interp_variant=${INTERP_VARIANT}"
  "--policy.interp_objective=${INTERP_OBJECTIVE}"
  "--policy.interp_plot_actions_l1=${INTERP_PLOT_ACTIONS_L1}"
  "--policy.attn_num_layers=${ATTN_NUM_LAYERS}"
  "--policy.attn_num_denoise_steps=${ATTN_NUM_DENOISE_STEPS}"
  "--policy.attn_score_beta=${ATTN_SCORE_BETA}"
  "--policy.grad_head_beta=${GRAD_HEAD_BETA}"
  "--policy.grad_head_norm=${GRAD_HEAD_NORM}"
  "--policy.grad_action_agg=${GRAD_ACTION_AGG}"
  "--policy.partial_grad_phi=${PARTIAL_GRAD_PHI}"
  "--policy.partial_grad_pos_weight=${PARTIAL_GRAD_POS_WEIGHT}"
  "--policy.partial_grad_grip_weight=${PARTIAL_GRAD_GRIP_WEIGHT}"
  "--policy.grad_denoise_steps=${GRAD_DENOISE_STEPS}"
  "--policy.grad_tau=${GRAD_TAU}"
  "--policy.grad_alpha=${GRAD_ALPHA}"
  "--policy.grad_beta=${GRAD_BETA}"
  "--policy.grad_region_ema=${GRAD_REGION_EMA}"
  "--policy.grad_keep_prev=${GRAD_KEEP_PREV}"
  # 2. Background
  "--policy.static_bg_enabled=${STATIC_BG_ENABLED}"
  "--policy.static_bg_threshold=${STATIC_BG_THRESHOLD}"
  "--policy.static_bg_camera_idx=${STATIC_BG_CAMERA_IDX}"
  "--policy.static_bg_score_gate=${STATIC_BG_SCORE_GATE}"
  # 3. Pruning Ratio
  "--policy.pruning_mode=${PRUNING_MODE}"
  "--policy.grad_region_mass=${GRAD_REGION_MASS}"
  "--policy.pruning_method=${PRUNING_METHOD}"
  "--policy.keep_ratio_low=${KEEP_RATIO_LOW}"
  "--policy.keep_ratio_high=${KEEP_RATIO_HIGH}"
  "--policy.mass_low=${MASS_LOW}"
  "--policy.mass_high=${MASS_HIGH}"
  "--policy.min_kept_tokens=${MIN_KEPT_TOKENS}"
  "--policy.max_kept_tokens=${MAX_KEPT_TOKENS}"
  "--policy.l1_dynamic_prune_enabled=${L1_DYNAMIC_PRUNE_ENABLED}"
  "--policy.l1_dynamic_prune_threshold=${L1_DYNAMIC_PRUNE_THRESHOLD}"
  "--policy.prune_ratio=${PRUNE_RATIO}"
  "--policy.region_patch_size=${REGION_PATCH_SIZE}"
  "--policy.token_temporal_threshold=${TOKEN_TEMPORAL_THRESHOLD}"
  "--policy.token_spatial_threshold=${TOKEN_SPATIAL_THRESHOLD}"
  "--policy.token_spatial_radius=${TOKEN_SPATIAL_RADIUS}"
  "--policy.token_prune_enabled=${TOKEN_PRUNE_ENABLED}"
  "--policy.vision_partial_update_enabled=${VISION_PARTIAL_UPDATE_ENABLED}"
  # 4. Eval Interval
  "--policy.region_eval_interval=${REGION_EVAL_INTERVAL}"
  "--policy.dynamic_eval_enabled=${DYNAMIC_EVAL_ENABLED}"
  "--policy.eval_entropy_threshold=${EVAL_ENTROPY_THRESHOLD}"
  "--policy.max_eval_interval=${MAX_EVAL_INTERVAL}"
  # 5. Overlay
  "--policy.overlay_mode=${OVERLAY_MODE}"
  "--policy.overlay_show_scores=${OVERLAY_SHOW_SCORES}"
  "--policy.overlay_show_ids=${OVERLAY_SHOW_IDS}"
  "--policy.score_debug_heatmap=${SCORE_DEBUG_HEATMAP}"
  # 6. Logging
  "--policy.rollout_dir=${ROLLOUT_DIR}"
  "--policy.local_log_dir=${LOCAL_LOG_DIR}"
  "--policy.run_id_note=${RUN_ID_NOTE}"
  "--policy.use_wandb=${USE_WANDB}"
  "--policy.wandb_entity=${WANDB_ENTITY}"
  "--policy.wandb_project=${WANDB_PROJECT}"
  # Env / Eval
  "--env.type=${ENV_TYPE}"
  "--env.task=${ENV_TASK}"
  "--env.task_id=${ENV_TASK_ID}"
  "--eval.batch_size=${EVAL_BATCH_SIZE}"
  "--eval.n_episodes=${EVAL_N_EPISODES}"
  "--eval.max_episodes_rendered=${MAX_EPISODES_RENDERED}"
  "--policy.compile_model=false"
)
if [ -n "${POLICY_PATH}" ]; then
  ARGS+=("--policy.path=${POLICY_PATH}")
else
  ARGS+=("--policy.type=${POLICY_TYPE}")
fi
if [ -n "${SEED}" ]; then
  ARGS+=("--seed=${SEED}")
fi

ENV_VARS=(
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "TOKENIZERS_PARALLELISM=false"
  "MUJOCO_GL=${MUJOCO_GL}"
  "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
)
if [ -n "${EGL_PLATFORM}" ]; then
  ENV_VARS+=("EGL_PLATFORM=${EGL_PLATFORM}")
fi
if [ -n "${MUJOCO_EGL_DEVICE_ID}" ]; then
  ENV_VARS+=("MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}")
fi
if [ -n "${LIBGL_ALWAYS_SOFTWARE}" ]; then
  ENV_VARS+=("LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}")
fi

TMPDIR="/home/nipeihuan/tmp"
mkdir -p "$TMPDIR"
export TMPDIR TMP TEMP

env "${ENV_VARS[@]}" lerobot-eval "${ARGS[@]}"
