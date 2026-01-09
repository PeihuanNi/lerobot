TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  lerobot-train \
  --policy.path=/home/nipeihuan/lerobot/outputs/train/spatial/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir=./outputs/train/spatial_c7_f5_finetune \
  --wandb.enable=false \
  --steps=100000 \
  --batch_size=64 \
  --eval.batch_size=1 \
  --eval.n_episodes=100 \
  --eval_freq=200000 \


if [ $? -ne 0 ]; then
    echo "❌ Training failed! Exiting."
    exit 1
fi

echo "✅ Training finished!"

# ==========================================
# 第二步：评估 (Evaluation)
# ==========================================
# 训练完成后，加载最新的权重 (checkpoints/last)
# 专门跑 100 个 episode
# ==========================================

echo "🔎 Starting Evaluation (100 episodes)..."

TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
  --policy.path=/home/nipeihuan/lerobot/outputs/train/spatial_c7_f5_finetune/checkpoints/last/pretrained_model \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=100