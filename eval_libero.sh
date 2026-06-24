CUDA_VISIBLE_DEVICES="$1"  # GPU id(s) passed from CLI, e.g., "0" or "0,1"
EXTRA_ARGS=("${@:2}")
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_EVAL_MODULE="lerobot.scripts.lerobot_eval"
OUTPUT_DIR="${OUTPUT_DIR:-}"   # e.g. "./outputs/eval/my_eval"; empty = auto-generate from score config
# POLICY_PATH="/home/nipeihuan/models/pi05_libero_finetuned"
POLICY_PATH="/home/nipeihuan/models/pi05_libero_finetuned"
POLICY_TYPE="pi05"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"  # empty => sdpa only for ACE token-selection, eager otherwise.
ACE_PARALLEL_VIT_STREAMS="${ACE_PARALLEL_VIT_STREAMS:-true}"  # ACE-only multi-camera ViT parallelism.
N_ACTION_STEPS=10 # available action chunk

MUJOCO_GL="egl"
PYOPENGL_PLATFORM="egl"
EGL_PLATFORM="surfaceless"
MUJOCO_EGL_DEVICE_ID=""
LIBGL_ALWAYS_SOFTWARE=""

ENV_TYPE="libero"
# ENV_TASK="libero_spatial, libero_object, libero_goal, libero_10"
# ENV_TASK_ID="[0,1,2,3,4,5,6,7,8,9]"
ENV_TASK="libero_10"
ENV_TASK_ID="[0]"
EVAL_BATCH_SIZE=1
EVAL_N_EPISODES=1
MAX_EPISODES_RENDERED=10             # 0 = no video; >0 = save that many mp4s

# ══════════════════════════════════════════════════════════════════════════════
# 0. Profiling
#    PROFILE_MODE:
#      "none"  - normal eval
#      "torch" - torch.profiler trace -> output_dir/PROFILE_DIR
#      "nsys"  - Nsight Systems capture with NVTX ranges
# ══════════════════════════════════════════════════════════════════════════════
PROFILE_MODE="nsys"                  # "none" | "torch" | "nsys"

# -- shared profiler labels --
PROFILE_EMIT_RECORD_FUNCTION=false    # recommended for torch profiler
PROFILE_EMIT_NVTX=true               # recommended for nsys

# -- torch.profiler --
PROFILE_DIR="torch_profile"
PROFILE_WAIT_STEPS=2
PROFILE_WARMUP_STEPS=2
PROFILE_ACTIVE_STEPS=6
PROFILE_REPEAT=1
PROFILE_RECORD_SHAPES=true
PROFILE_WITH_STACK=false

# -- nsys --
NSYS_BIN="${NSYS_BIN:-/usr/local/cuda-12.8/nsight-systems-2024.6.2/bin/nsys}"
NSYS_OUTPUT="${NSYS_OUTPUT:-}"       # empty => auto-generate under ./outputs/nsys
NSYS_TRACE="cuda,nvtx,osrt"
NSYS_SAMPLE="none"                  # matches the confirmed-working smoke test
NSYS_CAPTURE_RANGE=""                # keep empty unless you want custom capture range
NSYS_GPU_METRICS_DEVICES="none"      # "none" | "all" | GPU index list, e.g. "0" or "0,1"
NSYS_FORCE_OVERWRITE=true
NSYS_SHOW_OUTPUT=false              # smoke test succeeded without -w true
NSYS_STATS=false                    # run `nsys stats <report>` manually after capture

USE_L1_REGRESSION=false
USE_DIFFUSION=true
NUM_INFERENCE_STEPS=10 # denoise step

TOKEN_SELECTION_ENABLED="${TOKEN_SELECTION_ENABLED:-false}"  # Allow override via env var
TOKEN_PRUNE_ENABLED="${TOKEN_PRUNE_ENABLED:-false}"  # Allow override via env var

# ══════════════════════════════════════════════════════════════════════════════
# 1. Scoring Method
# ══════════════════════════════════════════════════════════════════════════════
GRAD_SCORE_METHOD="${GRAD_SCORE_METHOD:-ace}" # ace | grad_only | action_vision(attn_only alias)
SCORE_ATTN_SOURCE="action_vision"  # "action_vision" | "vision_text"
ACE_VARIANT="abs_heads"              # original | abs_heads | direct | dimension_independent
if [ -z "${ATTN_IMPLEMENTATION}" ]; then
  if [ "${TOKEN_SELECTION_ENABLED}" = "true" ] && [ "${GRAD_SCORE_METHOD}" = "ace" ]; then
    ATTN_IMPLEMENTATION="sdpa"
  else
    ATTN_IMPLEMENTATION="eager"
  fi
fi

# -- ace --
ACE_DENOISE_STEP=5            # -1 = avg all denoise steps; >=0 = specific step     ######## -1
ACE_USE_RESIDUAL=true         # true = full Chefer residual propagation across all layers; false is cheaper
ACE_ACTION_START=5              # first action step for objective (0-indexed)
ACE_ACTION_END=9               # last action step (exclusive); -1 = all (chunk_size)
ACE_OBJECTIVE="action_sample_L1"    # action_sample_L1 | action_sample_L2 | vector_field_L2
ACE_PLOT_ACTIONS_L1=false              # save per-episode action L1 norm curve

# -- Shared attention params --
ATTN_NUM_LAYERS=6                  # avg over last N Expert layers (1=last, 18=all)
ATTN_NUM_DENOISE_STEPS=3           # avg over last N denoise steps (1=last, 10=all)
ATTN_SCORE_BETA=2.0
GRAD_ACTION_AGG="sum"              # sum | max

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
#      "velocity"      — action chunk magnitude based dynamic pruning
# ══════════════════════════════════════════════════════════════════════════════
# -- Shared: min/max guardrails (always applied) --
DYNAMIC_PRUNE_MODE="accel"  # "none" | "l1_threshold" | "ema" | "accel" | "velocity"
PRUNE_RATIO=96                     # fixed kept-token count (when mode="none")
MIN_KEPT_TOKENS=64                 # aggressive (fewer kept)
MAX_KEPT_TOKENS=64                # conservative (more kept)

# -- Global Active Token Pool --
DISCARD_PREV_KEPT_RATIO=0.5       # discard highest-scoring 10% from previous frame
DISCARD_MODE="middle"              # "top" | "bottom" | "middle" | "random"
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
REGION_EVAL_INTERVAL=3

# ══════════════════════════════════════════════════════════════════════════════
# 4. Overlay / Visualization
#    "heatmap"      — continuous jet colormap; pruned regions darkened+hatched
#    "heatmap_topk" — only top-k high-score regions are colored; others stay raw
#    "label"        — discrete 5-color overlay
# ══════════════════════════════════════════════════════════════════════════════
OVERLAY_ENABLED=true                 # false = disable overlay extraction/rendering for timing comparisons
OVERLAY_MODE="heatmap"                    # "heatmap" | "heatmap_topk" | "label"; label avoids heatmap-grid work when not rendering
OVERLAY_TOPK=64                    # only used by "heatmap_topk"; 0 = no top-k cap
OVERLAY_SCORE_THRESHOLD=0.0        # only used by "heatmap_topk"; normalized [0, 1]
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

if [ -z "${RUN_ID_NOTE}" ]; then
  RUN_ID_NOTE="${GRAD_SCORE_METHOD}_${ACE_VARIANT}"
fi
if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="./outputs/profile/${GRAD_SCORE_METHOD}_${ACE_VARIANT}"
fi
if [ -z "${NSYS_OUTPUT}" ]; then
  NSYS_OUTPUT="./outputs/nsys/${ENV_TASK}_${GRAD_SCORE_METHOD}_${ACE_VARIANT}"
fi

ARGS=(
  "--policy.n_action_steps=${N_ACTION_STEPS}"
  "--policy.attn_implementation=${ATTN_IMPLEMENTATION}"
  "--policy.use_l1_regression=${USE_L1_REGRESSION}"
  "--policy.use_diffusion=${USE_DIFFUSION}"
  "--policy.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "--policy.token_selection_enabled=${TOKEN_SELECTION_ENABLED}"
  # 1. Scoring
  "--policy.grad_score_method=${GRAD_SCORE_METHOD}"
  "--policy.score_attn_source=${SCORE_ATTN_SOURCE}"
  "--policy.ace_denoise_step=${ACE_DENOISE_STEP}"
  "--policy.ace_use_residual=${ACE_USE_RESIDUAL}"
  "--policy.ace_action_start=${ACE_ACTION_START}"
  "--policy.ace_action_end=${ACE_ACTION_END}"
  "--policy.ace_variant=${ACE_VARIANT}"
  "--policy.ace_objective=${ACE_OBJECTIVE}"
  "--policy.ace_plot_actions_l1=${ACE_PLOT_ACTIONS_L1}"
  "--policy.ace_parallel_vit_streams=${ACE_PARALLEL_VIT_STREAMS}"
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
  "--policy.overlay_enabled=${OVERLAY_ENABLED}"
  "--policy.overlay_mode=${OVERLAY_MODE}"
  "--policy.overlay_topk=${OVERLAY_TOPK}"
  "--policy.overlay_score_threshold=${OVERLAY_SCORE_THRESHOLD}"
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
  "--env.task_ids=${ENV_TASK_ID}"
  "--eval.batch_size=${EVAL_BATCH_SIZE}"
  "--eval.n_episodes=${EVAL_N_EPISODES}"
  "--eval.max_episodes_rendered=${MAX_EPISODES_RENDERED}"
  "--policy.compile_model=false"
)

if [ "${PROFILE_MODE}" = "torch" ]; then
  ARGS+=(
    "--eval.profile_backend=pytorch"
    "--eval.profile_dir=${PROFILE_DIR}"
    "--eval.profile_emit_record_function=${PROFILE_EMIT_RECORD_FUNCTION}"
    "--eval.profile_emit_nvtx=${PROFILE_EMIT_NVTX}"
    "--eval.profile_wait_steps=${PROFILE_WAIT_STEPS}"
    "--eval.profile_warmup_steps=${PROFILE_WARMUP_STEPS}"
    "--eval.profile_active_steps=${PROFILE_ACTIVE_STEPS}"
    "--eval.profile_repeat=${PROFILE_REPEAT}"
    "--eval.profile_record_shapes=${PROFILE_RECORD_SHAPES}"
    "--eval.profile_with_stack=${PROFILE_WITH_STACK}"
  )
elif [ "${PROFILE_MODE}" = "nsys" ]; then
  PYTHON_EVAL_MODULE="lerobot.scripts.nsys_eval_bootstrap"
  ARGS+=(
    "--eval.profile_backend=none"
    "--eval.profile_emit_nvtx=${PROFILE_EMIT_NVTX}"
    "--eval.profile_emit_record_function=false"
  )
fi

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
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
  "MUJOCO_GL=${MUJOCO_GL}"
  "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
)
if [ "${PROFILE_MODE}" = "nsys" ]; then
  ENV_VARS+=("LEROBOT_NSYS_DISABLE_SYNC=true")
fi
if [ -n "${EGL_PLATFORM}" ]; then
  ENV_VARS+=("EGL_PLATFORM=${EGL_PLATFORM}")
fi
if [ -n "${MUJOCO_EGL_DEVICE_ID}" ]; then
  ENV_VARS+=("MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}")
fi
if [ -n "${LIBGL_ALWAYS_SOFTWARE}" ]; then
  ENV_VARS+=("LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}")
fi

# Do not hardcode TMPDIR here. On gpu01, forcing TMPDIR to a home-directory temp
# path breaks Nsight Systems injection; falling back to the system default tmp dir works.
if [ "${PROFILE_MODE}" = "nsys" ] && [ "${TMPDIR:-}" = "${HOME}/tmp" ]; then
  unset TMPDIR TMP TEMP
fi

for env_var in "${ENV_VARS[@]}"; do
  export "${env_var}"
done

CMD=("${PYTHON_BIN}" -m "${PYTHON_EVAL_MODULE}" "${ARGS[@]}" "${EXTRA_ARGS[@]}")

if [ "${PROFILE_MODE}" = "nsys" ]; then
  if [ ! -x "${NSYS_BIN}" ]; then
    NSYS_BIN="$(command -v nsys 2>/dev/null || true)"
  fi
  if [ -z "${NSYS_BIN}" ]; then
    echo "Error: nsys binary not found. Set NSYS_BIN=/path/to/nsys." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${NSYS_OUTPUT}")"
  rm -f "${NSYS_OUTPUT}.sqlite"
  NSYS_CMD=("${NSYS_BIN}" profile)
  if [ -n "${NSYS_SAMPLE}" ]; then
    NSYS_CMD+=(--sample "${NSYS_SAMPLE}")
  fi
  if [ "${NSYS_SHOW_OUTPUT}" = "true" ]; then
    NSYS_CMD+=(-w true)
  fi
  if [ "${NSYS_STATS}" = "true" ]; then
    NSYS_CMD+=(--stats=true)
  fi
  NSYS_CMD+=(-t "${NSYS_TRACE}" -o "${NSYS_OUTPUT}")
  if [ "${NSYS_FORCE_OVERWRITE}" = "true" ]; then
    NSYS_CMD+=(--force-overwrite true)
  fi
  if [ -n "${NSYS_CAPTURE_RANGE}" ]; then
    NSYS_CMD+=(--capture-range "${NSYS_CAPTURE_RANGE}")
  fi
  if [ "${NSYS_GPU_METRICS_DEVICES}" != "none" ]; then
    NSYS_CMD+=(--gpu-metrics-devices "${NSYS_GPU_METRICS_DEVICES}")
  fi
  printf 'Running:'
  printf ' %q' "${NSYS_CMD[@]}" "${CMD[@]}"
  printf '\n'
  "${NSYS_CMD[@]}" "${CMD[@]}"
else
  printf 'Running:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
fi
