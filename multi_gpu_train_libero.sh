MUJOCO_GL=egl PYOPENGL_PLATFORM=egl accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
  --policy.path=/home/nipeihuan/models/smolvla_libero \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_object \
  --output_dir=./outputs/train/2025-12-30-object \
  --wandb.enable=false \
  --steps=100000 \
  --batch_size=16 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=100000 \