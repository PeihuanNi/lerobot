#!/usr/bin/env bash

set -euo pipefail

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-0}}"

# -----------------------------------------------------------------------------
# Basic experiment settings
# -----------------------------------------------------------------------------
POLICY_TYPE="${POLICY_TYPE:-pi05}"           # pi0 | pi05
SPVLA_PRESET="${SPVLA_PRESET:-paper_acc}"   # baseline | schedule_only | token_only | reference | paper_acc | paper_speed
ENV_TYPE="${ENV_TYPE:-libero}"
ENV_TASK="${ENV_TASK:-libero_object}"
ENV_TASK_ID="${ENV_TASK_ID:-[0]}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-10}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-20}"
MEASURE_CUDA_LATENCY="${MEASURE_CUDA_LATENCY:-true}"
CUDA_LATENCY_WARMUP_STEPS="${CUDA_LATENCY_WARMUP_STEPS:-5}"
SEED="${SEED:-7}"

MODEL_ROOT="${MODEL_ROOT:-/home/nipeihuan/models}"
case "${POLICY_TYPE}" in
  pi0)
    DEFAULT_POLICY_PATH="${MODEL_ROOT}/pi0_libero_finetuned"
    ;;
  pi05)
    DEFAULT_POLICY_PATH="${MODEL_ROOT}/pi05_libero_finetuned"
    ;;
  *)
    echo "Unsupported POLICY_TYPE: ${POLICY_TYPE}" >&2
    exit 1
    ;;
esac
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"

# LeRobot's pi0/pi0.5 implementation always predicts a chunk internally. This
# controls how many queued actions are executed from one VLA refresh.
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
USE_DIFFUSION="${USE_DIFFUSION:-true}"
USE_L1_REGRESSION="${USE_L1_REGRESSION:-false}"

if [ "${ENV_TASK}" = "libero_long" ]; then
  ENV_TASK="libero_10"
fi

# -----------------------------------------------------------------------------
# SP-VLA preset defaults
# -----------------------------------------------------------------------------
case "${SPVLA_PRESET}" in
  baseline)
    PRESET_MODEL_SCHEDULING_ENABLED=false
    PRESET_SPVLA_REFERENCE_ENABLED=false
    PRESET_TOKEN_SELECTION_ENABLED=false
    PRESET_TOKEN_PRUNE_ENABLED=false
    PRESET_GRAD_SCORE_METHOD=ace
    PRESET_DYNAMIC_PRUNE_MODE=none
    PRESET_REGION_EVAL_INTERVAL=1
    PRESET_MIN_KEPT_TOKENS=128
    PRESET_MAX_KEPT_TOKENS=128
    ;;
  schedule_only)
    PRESET_MODEL_SCHEDULING_ENABLED=true
    PRESET_SPVLA_REFERENCE_ENABLED=true
    PRESET_TOKEN_SELECTION_ENABLED=false
    PRESET_TOKEN_PRUNE_ENABLED=false
    PRESET_GRAD_SCORE_METHOD=spvla_reference
    PRESET_DYNAMIC_PRUNE_MODE=none
    PRESET_REGION_EVAL_INTERVAL=1
    PRESET_MIN_KEPT_TOKENS=128
    PRESET_MAX_KEPT_TOKENS=128
    ;;
  token_only)
    PRESET_MODEL_SCHEDULING_ENABLED=false
    PRESET_SPVLA_REFERENCE_ENABLED=true
    PRESET_TOKEN_SELECTION_ENABLED=true
    PRESET_TOKEN_PRUNE_ENABLED=true
    PRESET_GRAD_SCORE_METHOD=spvla_reference
    PRESET_DYNAMIC_PRUNE_MODE=none
    PRESET_REGION_EVAL_INTERVAL=1
    PRESET_MIN_KEPT_TOKENS=0
    PRESET_MAX_KEPT_TOKENS=1000000
    ;;
  reference)
    PRESET_MODEL_SCHEDULING_ENABLED=true
    PRESET_SPVLA_REFERENCE_ENABLED=true
    PRESET_TOKEN_SELECTION_ENABLED=true
    PRESET_TOKEN_PRUNE_ENABLED=true
    PRESET_GRAD_SCORE_METHOD=spvla_reference
    PRESET_DYNAMIC_PRUNE_MODE=none
    PRESET_REGION_EVAL_INTERVAL=1
    PRESET_MIN_KEPT_TOKENS=0
    PRESET_MAX_KEPT_TOKENS=1000000
    ;;
  paper_speed)
    PRESET_MODEL_SCHEDULING_ENABLED=true
    PRESET_SPVLA_REFERENCE_ENABLED=false
    PRESET_TOKEN_SELECTION_ENABLED=true
    PRESET_TOKEN_PRUNE_ENABLED=true
    PRESET_GRAD_SCORE_METHOD=attn_only
    PRESET_DYNAMIC_PRUNE_MODE=velocity
    PRESET_REGION_EVAL_INTERVAL=2
    PRESET_MIN_KEPT_TOKENS=48
    PRESET_MAX_KEPT_TOKENS=96
    ;;
  paper_acc)
    PRESET_MODEL_SCHEDULING_ENABLED=true
    PRESET_SPVLA_REFERENCE_ENABLED=false
    PRESET_TOKEN_SELECTION_ENABLED=true
    PRESET_TOKEN_PRUNE_ENABLED=true
    PRESET_GRAD_SCORE_METHOD=attn_only
    PRESET_DYNAMIC_PRUNE_MODE=velocity
    PRESET_REGION_EVAL_INTERVAL=2
    PRESET_MIN_KEPT_TOKENS=64
    PRESET_MAX_KEPT_TOKENS=128
    ;;
  *)
    echo "Unsupported SPVLA_PRESET: ${SPVLA_PRESET}" >&2
    exit 1
    ;;
esac

case "${ENV_TASK}" in
  libero_spatial)
    REF_DEFAULT_Z_XY_RATE_SKIP=0.4
    REF_DEFAULT_Z_MAX_SKIP=0.3
    REF_DEFAULT_Z_THRE_PRUNE=0.5
    REF_DEFAULT_Z_MIN_PRUNE=0.9
    REF_DEFAULT_STEP_SKIP=1
    ;;
  libero_object)
    REF_DEFAULT_Z_XY_RATE_SKIP=0.6
    REF_DEFAULT_Z_MAX_SKIP=0.5
    REF_DEFAULT_Z_THRE_PRUNE=0.5
    REF_DEFAULT_Z_MIN_PRUNE=0.9
    REF_DEFAULT_STEP_SKIP=2
    ;;
  libero_goal)
    REF_DEFAULT_Z_XY_RATE_SKIP=1.2
    REF_DEFAULT_Z_MAX_SKIP=0.3
    REF_DEFAULT_Z_THRE_PRUNE=0.3
    REF_DEFAULT_Z_MIN_PRUNE=0.7
    REF_DEFAULT_STEP_SKIP=2
    ;;
  libero_10)
    REF_DEFAULT_Z_XY_RATE_SKIP=1.0
    REF_DEFAULT_Z_MAX_SKIP=0.3
    REF_DEFAULT_Z_THRE_PRUNE=0.3
    REF_DEFAULT_Z_MIN_PRUNE=0.85
    REF_DEFAULT_STEP_SKIP=2
    ;;
  *)
    REF_DEFAULT_Z_XY_RATE_SKIP=0.6
    REF_DEFAULT_Z_MAX_SKIP=0.3
    REF_DEFAULT_Z_THRE_PRUNE=0.5
    REF_DEFAULT_Z_MIN_PRUNE=0.9
    REF_DEFAULT_STEP_SKIP=1
    ;;
esac

# -----------------------------------------------------------------------------
# SP-VLA model scheduling
# -----------------------------------------------------------------------------
MODEL_SCHEDULING_ENABLED="${MODEL_SCHEDULING_ENABLED:-${PRESET_MODEL_SCHEDULING_ENABLED}}"
SPVLA_REFERENCE_ENABLED="${SPVLA_REFERENCE_ENABLED:-${PRESET_SPVLA_REFERENCE_ENABLED}}"
SCHEDULE_BUFFER_SIZE="${SCHEDULE_BUFFER_SIZE:-6}"
SCHEDULE_RATIO_THRESHOLD="${SCHEDULE_RATIO_THRESHOLD:-0.5}"
SCHEDULE_VELOCITY_MIN="${SCHEDULE_VELOCITY_MIN:-0.04}"
SCHEDULE_VELOCITY_MAX="${SCHEDULE_VELOCITY_MAX:-0.20}"
SCHEDULE_GENERATOR_REG_LAMBDA="${SCHEDULE_GENERATOR_REG_LAMBDA:-0.0001}"
SCHEDULE_VALIDITY_MAX_SCALE="${SCHEDULE_VALIDITY_MAX_SCALE:-2.5}"
SCHEDULE_WARMUP_VLA_STEPS="${SCHEDULE_WARMUP_VLA_STEPS:-2}"
SCHEDULE_REFERENCE_MIN_GENERATED_RATIO="${SCHEDULE_REFERENCE_MIN_GENERATED_RATIO:-0.6666667}"
STEP_SKIP="${STEP_SKIP:-${REF_DEFAULT_STEP_SKIP}}"
Z_XY_RATE_SKIP="${Z_XY_RATE_SKIP:-${REF_DEFAULT_Z_XY_RATE_SKIP}}"
Z_MAX_SKIP="${Z_MAX_SKIP:-${REF_DEFAULT_Z_MAX_SKIP}}"
REFERENCE_ACTION_CLIP_MIN="${REFERENCE_ACTION_CLIP_MIN:--0.9}"
REFERENCE_ACTION_CLIP_MAX="${REFERENCE_ACTION_CLIP_MAX:-0.9}"

# -----------------------------------------------------------------------------
# SP-VLA token pruning: paper-facing switches
# -----------------------------------------------------------------------------
TOKEN_SELECTION_ENABLED="${TOKEN_SELECTION_ENABLED:-${PRESET_TOKEN_SELECTION_ENABLED}}"
TOKEN_PRUNE_ENABLED="${TOKEN_PRUNE_ENABLED:-${PRESET_TOKEN_PRUNE_ENABLED}}"
SEMANTIC_SCORE_MASS="${SEMANTIC_SCORE_MASS:-0.7}"
SPATIAL_EDGE_ENABLED="${SPATIAL_EDGE_ENABLED:-true}"
CANNY_LOW_THRESHOLD="${CANNY_LOW_THRESHOLD:-100}"
CANNY_HIGH_THRESHOLD="${CANNY_HIGH_THRESHOLD:-200}"
REGION_PATCH_SIZE="${REGION_PATCH_SIZE:-1}"
Z_THRE_PRUNE="${Z_THRE_PRUNE:-${REF_DEFAULT_Z_THRE_PRUNE}}"
Z_MIN_PRUNE="${Z_MIN_PRUNE:-${REF_DEFAULT_Z_MIN_PRUNE}}"

DYNAMIC_PRUNE_MODE="${DYNAMIC_PRUNE_MODE:-${PRESET_DYNAMIC_PRUNE_MODE}}" # none | l1_threshold | ema | accel | velocity
PRUNE_RATIO="${PRUNE_RATIO:-64}"
MIN_KEPT_TOKENS="${MIN_KEPT_TOKENS:-${PRESET_MIN_KEPT_TOKENS}}"
MAX_KEPT_TOKENS="${MAX_KEPT_TOKENS:-${PRESET_MAX_KEPT_TOKENS}}"
PRUNE_VELOCITY_MIN="${PRUNE_VELOCITY_MIN:-${SCHEDULE_VELOCITY_MIN}}"
PRUNE_VELOCITY_MAX="${PRUNE_VELOCITY_MAX:-${SCHEDULE_VELOCITY_MAX}}"

# -----------------------------------------------------------------------------
# pi0 / pi0.5 semantic-score backend knobs
# These are implementation details for mapping SP-VLA's "semantic importance"
# to this repo's diffusion-VLA architecture. They are not paper-native knobs.
# -----------------------------------------------------------------------------
GRAD_SCORE_METHOD="${GRAD_SCORE_METHOD:-${PRESET_GRAD_SCORE_METHOD}}"   # spvla_reference | ace | attn_only
ACE_DENOISE_STEP="${ACE_DENOISE_STEP:-5}"                               # -1 = average all denoise steps
ACE_USE_RESIDUAL="${ACE_USE_RESIDUAL:-true}"
ACE_ACTION_START="${ACE_ACTION_START:-5}"
ACE_ACTION_END="${ACE_ACTION_END:-9}"
ACE_VARIANT="${ACE_VARIANT:-dimension_independent}"                    # original | abs_heads | dimension_independent
ACE_OBJECTIVE="${ACE_OBJECTIVE:-action_sample_L1}"                     # action_sample_L1 | vector_field_L2
ACE_PLOT_ACTIONS_L1="${ACE_PLOT_ACTIONS_L1:-false}"
ATTN_NUM_LAYERS="${ATTN_NUM_LAYERS:-6}"
ATTN_NUM_DENOISE_STEPS="${ATTN_NUM_DENOISE_STEPS:-3}"
ATTN_SCORE_BETA="${ATTN_SCORE_BETA:-2.0}"
GRAD_ACTION_AGG="${GRAD_ACTION_AGG:-max}"
GRAD_REGION_EMA="${GRAD_REGION_EMA:-0.0}"
GRAD_KEEP_PREV="${GRAD_KEEP_PREV:-false}"

# -----------------------------------------------------------------------------
# Extra pruning controls from this fork, not required by SP-VLA.
# -----------------------------------------------------------------------------
DISCARD_PREV_KEPT_RATIO="${DISCARD_PREV_KEPT_RATIO:-0.0}"
DISCARD_MODE="${DISCARD_MODE:-middle}"                                  # top | bottom | middle
RESET_DISCARD_POOL_ON_GRIPPER_CLOSE="${RESET_DISCARD_POOL_ON_GRIPPER_CLOSE:-false}"
GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.0}"
L1_DYNAMIC_PRUNE_THRESHOLD="${L1_DYNAMIC_PRUNE_THRESHOLD:-9.0}"
L1_EMA_ALPHA="${L1_EMA_ALPHA:-0.7}"
EMA_DIRECTION="${EMA_DIRECTION:-normal}"

REGION_EVAL_INTERVAL="${REGION_EVAL_INTERVAL:-${PRESET_REGION_EVAL_INTERVAL}}"

# -----------------------------------------------------------------------------
# Visualization / logging
# -----------------------------------------------------------------------------
OVERLAY_MODE="${OVERLAY_MODE:-heatmap}"                                 # heatmap | label
OVERLAY_SHOW_SCORES="${OVERLAY_SHOW_SCORES:-false}"
OVERLAY_SHOW_IDS="${OVERLAY_SHOW_IDS:-false}"
SCORE_DEBUG_HEATMAP="${SCORE_DEBUG_HEATMAP:-false}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

RUN_NAME="${RUN_NAME:-${POLICY_TYPE}_${ENV_TASK#libero_}_${SPVLA_PRESET}_cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_DIR}/rollout_artifacts}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${OUTPUT_DIR}/debug_artifacts}"
RUN_ID_NOTE="${RUN_ID_NOTE:-${RUN_NAME}}"

# -----------------------------------------------------------------------------
# Rendering backend
# -----------------------------------------------------------------------------
MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-}"
LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-}"

ARGS=(
  "--policy.n_action_steps=${N_ACTION_STEPS}"
  "--policy.use_l1_regression=${USE_L1_REGRESSION}"
  "--policy.use_diffusion=${USE_DIFFUSION}"
  "--policy.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "--policy.model_scheduling_enabled=${MODEL_SCHEDULING_ENABLED}"
  "--policy.spvla_reference_enabled=${SPVLA_REFERENCE_ENABLED}"
  "--policy.schedule_buffer_size=${SCHEDULE_BUFFER_SIZE}"
  "--policy.schedule_ratio_threshold=${SCHEDULE_RATIO_THRESHOLD}"
  "--policy.schedule_velocity_min=${SCHEDULE_VELOCITY_MIN}"
  "--policy.schedule_velocity_max=${SCHEDULE_VELOCITY_MAX}"
  "--policy.schedule_generator_reg_lambda=${SCHEDULE_GENERATOR_REG_LAMBDA}"
  "--policy.schedule_validity_max_scale=${SCHEDULE_VALIDITY_MAX_SCALE}"
  "--policy.schedule_warmup_vla_steps=${SCHEDULE_WARMUP_VLA_STEPS}"
  "--policy.schedule_reference_min_generated_ratio=${SCHEDULE_REFERENCE_MIN_GENERATED_RATIO}"
  "--policy.step_skip=${STEP_SKIP}"
  "--policy.z_xy_rate_skip=${Z_XY_RATE_SKIP}"
  "--policy.z_max_skip=${Z_MAX_SKIP}"
  "--policy.reference_action_clip_min=${REFERENCE_ACTION_CLIP_MIN}"
  "--policy.reference_action_clip_max=${REFERENCE_ACTION_CLIP_MAX}"
  "--policy.token_selection_enabled=${TOKEN_SELECTION_ENABLED}"
  "--policy.token_prune_enabled=${TOKEN_PRUNE_ENABLED}"
  "--policy.grad_score_method=${GRAD_SCORE_METHOD}"
  "--policy.ace_denoise_step=${ACE_DENOISE_STEP}"
  "--policy.ace_use_residual=${ACE_USE_RESIDUAL}"
  "--policy.ace_action_start=${ACE_ACTION_START}"
  "--policy.ace_action_end=${ACE_ACTION_END}"
  "--policy.ace_variant=${ACE_VARIANT}"
  "--policy.ace_objective=${ACE_OBJECTIVE}"
  "--policy.ace_plot_actions_l1=${ACE_PLOT_ACTIONS_L1}"
  "--policy.attn_num_layers=${ATTN_NUM_LAYERS}"
  "--policy.attn_num_denoise_steps=${ATTN_NUM_DENOISE_STEPS}"
  "--policy.attn_score_beta=${ATTN_SCORE_BETA}"
  "--policy.grad_action_agg=${GRAD_ACTION_AGG}"
  "--policy.grad_region_ema=${GRAD_REGION_EMA}"
  "--policy.grad_keep_prev=${GRAD_KEEP_PREV}"
  "--policy.semantic_score_mass=${SEMANTIC_SCORE_MASS}"
  "--policy.spatial_edge_enabled=${SPATIAL_EDGE_ENABLED}"
  "--policy.canny_low_threshold=${CANNY_LOW_THRESHOLD}"
  "--policy.canny_high_threshold=${CANNY_HIGH_THRESHOLD}"
  "--policy.z_thre_prune=${Z_THRE_PRUNE}"
  "--policy.z_min_prune=${Z_MIN_PRUNE}"
  "--policy.min_kept_tokens=${MIN_KEPT_TOKENS}"
  "--policy.max_kept_tokens=${MAX_KEPT_TOKENS}"
  "--policy.dynamic_prune_mode=${DYNAMIC_PRUNE_MODE}"
  "--policy.prune_ratio=${PRUNE_RATIO}"
  "--policy.prune_velocity_min=${PRUNE_VELOCITY_MIN}"
  "--policy.prune_velocity_max=${PRUNE_VELOCITY_MAX}"
  "--policy.l1_dynamic_prune_threshold=${L1_DYNAMIC_PRUNE_THRESHOLD}"
  "--policy.l1_ema_alpha=${L1_EMA_ALPHA}"
  "--policy.ema_direction=${EMA_DIRECTION}"
  "--policy.discard_prev_kept_ratio=${DISCARD_PREV_KEPT_RATIO}"
  "--policy.discard_mode=${DISCARD_MODE}"
  "--policy.reset_discard_pool_on_gripper_close=${RESET_DISCARD_POOL_ON_GRIPPER_CLOSE}"
  "--policy.gripper_close_threshold=${GRIPPER_CLOSE_THRESHOLD}"
  "--policy.region_patch_size=${REGION_PATCH_SIZE}"
  "--policy.region_eval_interval=${REGION_EVAL_INTERVAL}"
  "--policy.overlay_mode=${OVERLAY_MODE}"
  "--policy.overlay_show_scores=${OVERLAY_SHOW_SCORES}"
  "--policy.overlay_show_ids=${OVERLAY_SHOW_IDS}"
  "--policy.score_debug_heatmap=${SCORE_DEBUG_HEATMAP}"
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
  "--eval.max_episodes_rendered=${MAX_EPISODES_RENDERED}"
  "--eval.measure_cuda_latency=${MEASURE_CUDA_LATENCY}"
  "--eval.cuda_latency_warmup_steps=${CUDA_LATENCY_WARMUP_STEPS}"
  "--policy.compile_model=false"
)

if [ -n "${POLICY_PATH}" ]; then
  ARGS+=("--policy.path=${POLICY_PATH}")
else
  ARGS+=("--policy.type=${POLICY_TYPE}")
fi

if [ -n "${OUTPUT_DIR}" ]; then
  ARGS+=("--output_dir=${OUTPUT_DIR}")
fi

if [ -n "${SEED}" ]; then
  ARGS+=("--seed=${SEED}")
fi

ENV_VARS=(
  "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
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
mkdir -p "${TMPDIR}" "${OUTPUT_DIR}" "${ROLLOUT_DIR}" "${LOCAL_LOG_DIR}"
export TMPDIR TMP TEMP

echo "Running ${POLICY_TYPE} on ${ENV_TASK} with preset=${SPVLA_PRESET}"
echo "Output dir: ${OUTPUT_DIR}"

env "${ENV_VARS[@]}" lerobot-eval "${ARGS[@]}"
