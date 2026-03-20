CUDA_VISIBLE_DEVICES="$1"  # GPU id(s) passed from CLI, e.g., "0" or "0,1"
OUTPUT_DIR="./outputs/eval/gt-pi0_object"   # e.g. "./outputs/eval/my_eval"; empty = lerobot-eval auto-generates one
# POLICY_PATH="/home/nipeihuan/models/pi05_libero_finetuned"
POLICY_PATH="/home/nipeihuan/models/pi0_libero_finetuned"
POLICY_TYPE="pi0"
N_ACTION_STEPS=10 # available action chunk

MUJOCO_GL="egl"
PYOPENGL_PLATFORM="egl"
EGL_PLATFORM="surfaceless"
MUJOCO_EGL_DEVICE_ID=""
LIBGL_ALWAYS_SOFTWARE=""

ENV_TYPE="libero"
# ENV_TASK="libero_spatial, libero_object, libero_goal, libero_10"
# ENV_TASK_ID="[0,1,2,3,4,5,6,7,8,9]"
ENV_TASK="libero_object"
ENV_TASK_ID="[0,1,2,3,4,5,6,7,8,9]"
EVAL_BATCH_SIZE=2
EVAL_N_EPISODES=50
MAX_EPISODES_RENDERED=50             # 0 = no video; >0 = save that many mp4s

USE_L1_REGRESSION=false
USE_DIFFUSION=true
NUM_INFERENCE_STEPS=10 # denoise step

TOKEN_SELECTION_ENABLED=true
TOKEN_PRUNE_ENABLED=false

# ══════════════════════════════════════════════════════════════════════════════
# 1. Scoring Method
# ══════════════════════════════════════════════════════════════════════════════
GRAD_SCORE_METHOD="transformer_interpretability"
# transformer_interpretability | attn_only

# -- transformer_interpretability --
INTERP_DENOISE_STEP=5            # -1 = avg all denoise steps; >=0 = specific step     ######## -1
INTERP_USE_RESIDUAL=true          # true = full Chefer residual propagation across all layers
INTERP_ACTION_START=5              # first action step for objective (0-indexed)
INTERP_ACTION_END=9               # last action step (exclusive); -1 = all (chunk_size)
INTERP_VARIANT="dimension_independent"     # "original" = ReLU | "abs_heads" = mean_h(|g*A|) | "dimension_independent" = per-action-dim backprop
INTERP_OBJECTIVE="action_sample_L1"    # action_sample_L1 | vector_field_L2
INTERP_PLOT_ACTIONS_L1=false              # save per-episode action L1 norm curve

# -- Shared attention params --
ATTN_NUM_LAYERS=6                  # avg over last N Expert layers (1=last, 18=all)
ATTN_NUM_DENOISE_STEPS=3           # avg over last N denoise steps (1=last, 10=all)
ATTN_SCORE_BETA=2.0
GRAD_ACTION_AGG="max"              # sum | max

# -- General scoring --
GRAD_REGION_EMA=0.0                # EMA smoothing (0=none)
GRAD_KEEP_PREV=false

# ══════════════════════════════════════════════════════════════════════════════
# 2. Pruning Ratio
# ══════════════════════════════════════════════════════════════════════════════
#    Dynamic Prune Mode:
#      "none"          — fixed PRUNE_RATIO
#      "l1_threshold"  — binary: L1 > threshold → MIN, else → MAX
#      "ema"           — continuous: EMA-smoothed L1 → sigmoid → [MAX, MIN]
# ══════════════════════════════════════════════════════════════════════════════
# -- Shared: min/max guardrails (always applied) --
DYNAMIC_PRUNE_MODE="accel"  # "none" | "l1_threshold" | "ema" | "accel"
PRUNE_RATIO=32                     # fixed kept-token count (when mode="none")
MIN_KEPT_TOKENS=64                 # aggressive (fewer kept)
MAX_KEPT_TOKENS=128                # conservative (more kept)

# -- Global Active Token Pool --
DISCARD_PREV_KEPT_RATIO=0.1        # discard highest-scoring 10% from previous frame
DISCARD_MODE="middle"              # "top" | "bottom" | "middle"
RESET_DISCARD_POOL_ON_GRIPPER_CLOSE=true
GRIPPER_CLOSE_THRESHOLD=0.0        # action[..., -1] > threshold => schedule full-pool restore

# -- l1_threshold params --
L1_DYNAMIC_PRUNE_THRESHOLD=9.0     # L1 threshold: above → MIN, below → MAX

# -- ema params --
L1_EMA_ALPHA=0.7                   # EMA smoothing (higher = slower response)
EMA_DIRECTION="normal"             # "normal" = fast→fewer tokens; "reverse" = fast→more tokens

REGION_PATCH_SIZE=1

# ══════════════════════════════════════════════════════════════════════════════
# 3. Eval Interval
# ══════════════════════════════════════════════════════════════════════════════
REGION_EVAL_INTERVAL=2

# ══════════════════════════════════════════════════════════════════════════════
# 4. Overlay / Visualization
#    "heatmap" — continuous jet colormap; pruned regions darkened+hatched
#    "label"   — discrete 5-color overlay
# ══════════════════════════════════════════════════════════════════════════════
OVERLAY_MODE="heatmap"             # "heatmap" | "label"
OVERLAY_SHOW_SCORES=false
OVERLAY_SHOW_IDS=false
SCORE_DEBUG_HEATMAP=false          # debug: no pruning, every frame scored

# ══════════════════════════════════════════════════════════════════════════════
# 5. Logging / Debug
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
  "--policy.grad_action_agg=${GRAD_ACTION_AGG}"
  "--policy.grad_region_ema=${GRAD_REGION_EMA}"
  "--policy.grad_keep_prev=${GRAD_KEEP_PREV}"
  # 2. Pruning Ratio
  "--policy.min_kept_tokens=${MIN_KEPT_TOKENS}"
  "--policy.max_kept_tokens=${MAX_KEPT_TOKENS}"
  "--policy.discard_prev_kept_ratio=${DISCARD_PREV_KEPT_RATIO}"
  "--policy.discard_mode=${DISCARD_MODE}"
  "--policy.reset_discard_pool_on_gripper_close=${RESET_DISCARD_POOL_ON_GRIPPER_CLOSE}"
  "--policy.gripper_close_threshold=${GRIPPER_CLOSE_THRESHOLD}"
  "--policy.dynamic_prune_mode=${DYNAMIC_PRUNE_MODE}"
  "--policy.l1_dynamic_prune_threshold=${L1_DYNAMIC_PRUNE_THRESHOLD}"
  "--policy.l1_ema_alpha=${L1_EMA_ALPHA}"
  "--policy.ema_direction=${EMA_DIRECTION}"
  "--policy.prune_ratio=${PRUNE_RATIO}"
  "--policy.region_patch_size=${REGION_PATCH_SIZE}"
  "--policy.token_prune_enabled=${TOKEN_PRUNE_ENABLED}"
  # 3. Eval Interval
  "--policy.region_eval_interval=${REGION_EVAL_INTERVAL}"
  # 4. Overlay
  "--policy.overlay_mode=${OVERLAY_MODE}"
  "--policy.overlay_show_scores=${OVERLAY_SHOW_SCORES}"
  "--policy.overlay_show_ids=${OVERLAY_SHOW_IDS}"
  "--policy.score_debug_heatmap=${SCORE_DEBUG_HEATMAP}"
  # 5. Logging
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
if [ -n "${OUTPUT_DIR}" ]; then
  ARGS+=("--output_dir=${OUTPUT_DIR}")
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
