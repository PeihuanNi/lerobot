CUDA_VISIBLE_DEVICES="$1"  # GPU id(s) passed from CLI, e.g., "0" or "0,1"

POLICY_PATH="/home/nipeihuan/models/pi05_libero_finetuned"
POLICY_TYPE="pi05"
N_ACTION_STEPS=1

MUJOCO_GL="egl"
PYOPENGL_PLATFORM="egl"
EGL_PLATFORM="surfaceless"
MUJOCO_EGL_DEVICE_ID=""
LIBGL_ALWAYS_SOFTWARE=""

ENV_TYPE="libero"
# ENV_TASK="libero_spatial, libero_object, libero_goal, libero_10"
# ENV_TASK_ID="[0,1,2,3,4,5,6,7,8,9]"
ENV_TASK="libero_spatial"
ENV_TASK_ID="[4]"
EVAL_BATCH_SIZE=1
EVAL_N_EPISODES=5

USE_L1_REGRESSION=false
USE_DIFFUSION=true
NUM_INFERENCE_STEPS=10

TOKEN_SELECTION_ENABLED=true
GRAD_SCORE_METHOD="partial_grad"
GRAD_DENOISE_STEPS=1
ATTN_SCORE_BETA=2.0
ATTN_NUM_LAYERS=6                  # average attention over last N Expert layers (1=last only, 18=all)
ATTN_NUM_DENOISE_STEPS=3           # average attention over last N denoise steps (1=last only, 10=all)
GRAD_HEAD_BETA=1.0
GRAD_HEAD_NORM="sum"
GRAD_ACTION_AGG="max"  # sum | max — action steps & heads aggregation
PARTIAL_GRAD_PHI="l2"
PARTIAL_GRAD_POS_WEIGHT=1.0
PARTIAL_GRAD_GRIP_WEIGHT=1.0
GRAD_TAU=0.1
GRAD_ALPHA=1.0
GRAD_BETA=1.0
GRAD_REGION_MASS=0.7               # 对 region score 降序排列，累加到总分的 70% 就截止，低于阈值的 region 被剪掉
GRAD_REGION_EMA=0.0                # EMA smoothing for region scores; 0 = no smoothing, 1 = keep previous score. 注意，2026/2/26之前是0.7，但是实际上对于当前时间步是没有score的，也就是说这个平滑是用t-2和t-1的score来平滑得到最终分数从而决定t时刻的token保留，这样反而引入了过时信息，可能会导致性能下降
GRAD_KEEP_PREV=false
DYNAMIC_EVAL_ENABLED=true          # entropy-based adaptive eval interval
EVAL_ENTROPY_THRESHOLD=2.0           # re-eval trigger threshold (0~5.55, 256 tokens)
MAX_EVAL_INTERVAL=5                # hard-cap: re-eval at least every N frames

# ── Dynamic pruning: entropy-coupled aggressiveness ──────────────────────────
# Requires DYNAMIC_MASS_ENABLED=true; GRAD_REGION_MASS is used as static fallback when false.
#
# PRUNING_METHOD controls how regions are selected:
#   "topk_ratio" — keep a fixed *fraction* of regions by score rank.
#                  Linear control, no cliff-like jumps. (RECOMMENDED)
#                  effective_ratio = KEEP_RATIO_LOW + h * (KEEP_RATIO_HIGH - KEEP_RATIO_LOW)
#   "mass"       — cumulative-score threshold (original).
#                  Sensitive near 1.0 due to long-tail score distributions.
#                  effective_mass = MASS_LOW + h * (MASS_HIGH - MASS_LOW)
#
# h = normalised region-score entropy ∈ [0,1]:
#   h≈0 (focused)  → use LOW  param → aggressive pruning
#   h≈1 (diffuse)  → use HIGH param → conservative, keep more
DYNAMIC_MASS_ENABLED=true
PRUNING_METHOD="topk_ratio"        # "topk_ratio" | "mass"
# -- TopK-ratio params (used when PRUNING_METHOD="topk_ratio") --
KEEP_RATIO_LOW=0.5                 # keep 50% when model is focused (aggressive)
KEEP_RATIO_HIGH=0.9                # keep 90% when model is confused (conservative)
# -- Mass params (used when PRUNING_METHOD="mass") --
MASS_LOW=0.5                       # cumulative-score threshold when focused
MASS_HIGH=0.95                     # cumulative-score threshold when confused
REGION_EVAL_INTERVAL=2
REGION_PATCH_SIZE=1
TOKEN_TEMPORAL_THRESHOLD=0.7
TOKEN_SPATIAL_THRESHOLD=0.7
TOKEN_SPATIAL_RADIUS=1
TOKEN_PRUNE_ENABLED=true
MIN_KEPT_TOKENS=128
MAX_KEPT_TOKENS=128
VISION_PARTIAL_UPDATE_ENABLED=false

# ── Static background detection (cosine similarity to 1st frame) ─────────────
# Prune regions in the overhead (agentview) camera whose cosine similarity to
# the first-frame reference exceeds STATIC_BG_THRESHOLD.  Applied on ALL frames.
STATIC_BG_ENABLED=true
STATIC_BG_THRESHOLD=0.95            # cosine similarity threshold (higher = stricter)
STATIC_BG_CAMERA_IDX=0              # image slot: 0 = agentview, 1 = wrist
STATIC_BG_SCORE_GATE=0.5             # protect top 50% scored regions from bg pruning (1.0 = no protection) 这个gate是指可以有多少被认为是背景

OVERLAY_SHOW_SCORES=false
OVERLAY_SHOW_IDS=false
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
  "--policy.grad_score_method=${GRAD_SCORE_METHOD}"
  "--policy.grad_denoise_steps=${GRAD_DENOISE_STEPS}"
  "--policy.attn_score_beta=${ATTN_SCORE_BETA}"
  "--policy.attn_num_layers=${ATTN_NUM_LAYERS}"
  "--policy.attn_num_denoise_steps=${ATTN_NUM_DENOISE_STEPS}"
  "--policy.grad_head_beta=${GRAD_HEAD_BETA}"
  "--policy.grad_head_norm=${GRAD_HEAD_NORM}"
  "--policy.grad_action_agg=${GRAD_ACTION_AGG}"
  "--policy.partial_grad_phi=${PARTIAL_GRAD_PHI}"
  "--policy.partial_grad_pos_weight=${PARTIAL_GRAD_POS_WEIGHT}"
  "--policy.partial_grad_grip_weight=${PARTIAL_GRAD_GRIP_WEIGHT}"
  "--policy.grad_tau=${GRAD_TAU}"
  "--policy.grad_alpha=${GRAD_ALPHA}"
  "--policy.grad_beta=${GRAD_BETA}"
  "--policy.grad_region_mass=${GRAD_REGION_MASS}"
  "--policy.dynamic_mass_enabled=${DYNAMIC_MASS_ENABLED}"
  "--policy.pruning_method=${PRUNING_METHOD}"
  "--policy.keep_ratio_low=${KEEP_RATIO_LOW}"
  "--policy.keep_ratio_high=${KEEP_RATIO_HIGH}"
  "--policy.mass_low=${MASS_LOW}"
  "--policy.mass_high=${MASS_HIGH}"
  "--policy.grad_region_ema=${GRAD_REGION_EMA}"
  "--policy.grad_keep_prev=${GRAD_KEEP_PREV}"
  "--policy.dynamic_eval_enabled=${DYNAMIC_EVAL_ENABLED}"
  "--policy.eval_entropy_threshold=${EVAL_ENTROPY_THRESHOLD}"
  "--policy.max_eval_interval=${MAX_EVAL_INTERVAL}"
  "--policy.region_eval_interval=${REGION_EVAL_INTERVAL}"
  "--policy.region_patch_size=${REGION_PATCH_SIZE}"
  "--policy.token_temporal_threshold=${TOKEN_TEMPORAL_THRESHOLD}"
  "--policy.token_spatial_threshold=${TOKEN_SPATIAL_THRESHOLD}"
  "--policy.token_spatial_radius=${TOKEN_SPATIAL_RADIUS}"
  "--policy.token_prune_enabled=${TOKEN_PRUNE_ENABLED}"
  "--policy.min_kept_tokens=${MIN_KEPT_TOKENS}"
  "--policy.max_kept_tokens=${MAX_KEPT_TOKENS}"
  "--policy.vision_partial_update_enabled=${VISION_PARTIAL_UPDATE_ENABLED}"
  "--policy.static_bg_enabled=${STATIC_BG_ENABLED}"
  "--policy.static_bg_threshold=${STATIC_BG_THRESHOLD}"
  "--policy.static_bg_camera_idx=${STATIC_BG_CAMERA_IDX}"
  "--policy.static_bg_score_gate=${STATIC_BG_SCORE_GATE}"
  "--policy.overlay_show_scores=${OVERLAY_SHOW_SCORES}"
  "--policy.overlay_show_ids=${OVERLAY_SHOW_IDS}"
  "--policy.rollout_dir=${ROLLOUT_DIR}"
  "--policy.local_log_dir=${LOCAL_LOG_DIR}"
  "--policy.run_id_note=${RUN_ID_NOTE}"
  "--policy.use_wandb=${USE_WANDB}"
  "--policy.wandb_entity=${WANDB_ENTITY}"
  "--policy.wandb_project=${WANDB_PROJECT}"
  "--env.type=${ENV_TYPE}"
  "--env.task=${ENV_TASK}"
  "--env.task_id=${ENV_TASK_ID}"
  "--eval.batch_size=${EVAL_BATCH_SIZE}"
  "--eval.n_episodes=${EVAL_N_EPISODES}"
  "--policy.compile_model=false"
)
if [ -n "${POLICY_PATH}" ]; then
  ARGS+=("--policy.path=${POLICY_PATH}")
else
  ARGS+=("--policy.type=${POLICY_TYPE}")
fi
if [ -n "${SEED}" ]; then
  ARGS+=("--policy.seed=${SEED}")
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
