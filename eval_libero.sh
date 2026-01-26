CUDA_VISIBLE_DEVICES="$1"  # GPU id(s) passed from CLI, e.g., "0" or "0,1"

POLICY_PATH="/home/nipeihuan/models/pi0_libero_finetuned"
POLICY_TYPE="pi0"
N_ACTION_STEPS=10

ENV_TYPE="libero"
ENV_TASK="libero_spatial"
ENV_TASK_ID="[0]"
EVAL_BATCH_SIZE=1
EVAL_N_EPISODES=100

USE_L1_REGRESSION=false
USE_DIFFUSION=true
NUM_INFERENCE_STEPS=10

TOKEN_SELECTION_ENABLED=true
GRAD_SCORE_METHOD="partial_grad"
GRAD_DENOISE_STEPS=1
ATTN_SCORE_BETA=2.0
PARTIAL_GRAD_PHI="l2"
PARTIAL_GRAD_POS_WEIGHT=1.0
PARTIAL_GRAD_GRIP_WEIGHT=2.0
GRAD_TAU=0.1
GRAD_ALPHA=1.0
GRAD_BETA=1.0
GRAD_REGION_MASS=0.7
GRAD_REGION_EMA=0.7
GRAD_KEEP_PREV=false
REGION_EVAL_INTERVAL=2
REGION_PATCH_SIZE=1
TOKEN_TEMPORAL_THRESHOLD=0.7
TOKEN_SPATIAL_THRESHOLD=0.7
TOKEN_SPATIAL_RADIUS=1
TOKEN_PRUNE_ENABLED=true
MIN_KEPT_TOKENS=1
MAX_KEPT_TOKENS=128
VISION_PARTIAL_UPDATE_ENABLED=false

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
  "--policy.partial_grad_phi=${PARTIAL_GRAD_PHI}"
  "--policy.partial_grad_pos_weight=${PARTIAL_GRAD_POS_WEIGHT}"
  "--policy.partial_grad_grip_weight=${PARTIAL_GRAD_GRIP_WEIGHT}"
  "--policy.grad_tau=${GRAD_TAU}"
  "--policy.grad_alpha=${GRAD_ALPHA}"
  "--policy.grad_beta=${GRAD_BETA}"
  "--policy.grad_region_mass=${GRAD_REGION_MASS}"
  "--policy.grad_region_ema=${GRAD_REGION_EMA}"
  "--policy.grad_keep_prev=${GRAD_KEEP_PREV}"
  "--policy.region_eval_interval=${REGION_EVAL_INTERVAL}"
  "--policy.region_patch_size=${REGION_PATCH_SIZE}"
  "--policy.token_temporal_threshold=${TOKEN_TEMPORAL_THRESHOLD}"
  "--policy.token_spatial_threshold=${TOKEN_SPATIAL_THRESHOLD}"
  "--policy.token_spatial_radius=${TOKEN_SPATIAL_RADIUS}"
  "--policy.token_prune_enabled=${TOKEN_PRUNE_ENABLED}"
  "--policy.min_kept_tokens=${MIN_KEPT_TOKENS}"
  "--policy.max_kept_tokens=${MAX_KEPT_TOKENS}"
  "--policy.vision_partial_update_enabled=${VISION_PARTIAL_UPDATE_ENABLED}"
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
)
if [ -n "${POLICY_PATH}" ]; then
  ARGS+=("--policy.path=${POLICY_PATH}")
else
  ARGS+=("--policy.type=${POLICY_TYPE}")
fi
if [ -n "${SEED}" ]; then
  ARGS+=("--policy.seed=${SEED}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" TOKENIZERS_PARALLELISM=false MUJOCO_GL="egl" PYOPENGL_PLATFORM="egl" lerobot-eval "${ARGS[@]}"
