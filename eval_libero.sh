TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
  --policy.path=/home/nipeihuan/lerobot/outputs/train/spatial/checkpoints/last/pretrained_model \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_id='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=100