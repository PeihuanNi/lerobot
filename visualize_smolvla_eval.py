#!/usr/bin/env python
"""
在 LIBERO 评估过程中可视化 SmolVLA vision model 各层输出

基于 lerobot_eval.py 修改，添加了可视化功能

使用方法:
CUDA_VISIBLE_DEVICES=1 python visualize_smolvla_eval.py \
    --policy.path=/home/nipeihuan/lerobot/outputs/train/spatial/checkpoints/last/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial_0 \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --viz_save_dir=./smolvla_layer_viz \
    --viz_every_n_steps=10
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm

# 导入可视化工具
from lerobot.policies.smolvla.visualization_utils import SmolVLAVisualizationHook


def rollout_with_viz(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    viz_hook: SmolVLAVisualizationHook,
    seeds: list[int] | None = None,
    viz_every_n_steps: int = 10,
    episode_idx: int = 0,
) -> dict:
    """带可视化的 rollout"""
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    policy.reset()
    viz_hook.reset()
    
    observation, info = env.reset(seed=seeds)

    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    
    # 保存图像用于可视化
    prev_image = None

    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    
    progbar = trange(max_steps, desc=f"Episode {episode_idx}", disable=inside_slurm(), leave=False)
    check_env_attributes_and_types(env)
    
    while not np.all(done) and step < max_steps:
        # 保存当前图像 - 处理嵌套字典结构
        curr_image = None
        
        def extract_image(obs_dict, depth=0):
            """递归提取图像"""
            for key, val in obs_dict.items():
                if isinstance(val, dict):
                    result = extract_image(val, depth + 1)
                    if result is not None:
                        return result
                elif isinstance(val, np.ndarray):
                    # 检查是否是图像（3D 或 4D 数组）
                    if val.ndim >= 3 and (val.shape[-1] == 3 or val.shape[0] == 3 or val.shape[1] == 3):
                        if step == 0:
                            print(f"[DEBUG] Found image: {key}, shape: {val.shape}, dtype: {val.dtype}")
                        return val
            return None
        
        img = extract_image(observation)
        if img is not None:
            # 处理维度
            if img.ndim == 4:  # (batch, H, W, C) or (batch, C, H, W)
                img = img[0]
            if img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))
            # 处理数据类型
            if img.dtype in [np.float32, np.float64]:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            curr_image = img
        
        # 预处理
        obs_processed = preprocess_observation(observation)
        obs_processed = add_envs_task(env, obs_processed)
        obs_processed = env_preprocessor(obs_processed)
        obs_processed = preprocessor(obs_processed)
        
        with torch.inference_mode():
            action = policy.select_action(obs_processed)
        
        # 调试：打印图像状态（只在前几步）
        if step < 3:
            print(f"[DEBUG] step={step}, prev_image is None: {prev_image is None}, curr_image is None: {curr_image is None}")
            if curr_image is not None:
                print(f"[DEBUG]   curr_image shape: {curr_image.shape}, dtype: {curr_image.dtype}")
        
        # 可视化
        if prev_image is not None and curr_image is not None:
            if step % viz_every_n_steps == 0:
                print(f"[DEBUG] Calling visualize_frame_comparison at step {step}")
                viz_hook.visualize_frame_comparison(
                    prev_image=prev_image,
                    curr_image=curr_image,
                    num_layers_to_show=4,
                    episode_idx=episode_idx
                )
                print(f"[DEBUG] Saved visualization for step {step}")
            # 打印统计
            if step % 50 == 0:
                viz_hook.print_frame_stats()
        
        prev_image = curr_image.copy() if curr_image is not None else None
        viz_hook.increment_frame()
        
        # 后处理动作
        action = postprocessor(action)
        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]
        
        action_numpy = action.to("cpu").numpy()
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        
        # 处理成功标志
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict):
                successes = final_info["is_success"].tolist()
            else:
                successes = [False] * env.num_envs
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"success": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # 保存统计信息
    viz_hook.save_all_stats(episode_idx)

    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }

    return ret


def eval_with_viz(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    viz_save_dir: str = "./smolvla_layer_viz",
    viz_every_n_steps: int = 10,
    start_seed: int | None = None,
) -> dict:
    """带可视化的评估"""
    
    start = time.time()
    policy.eval()
    
    # 创建可视化 hook
    viz_hook = SmolVLAVisualizationHook(save_dir=viz_save_dir)
    
    try:
        viz_hook.register_hooks(policy)
    except ValueError as e:
        print(f"[WARN] Failed to register hooks: {e}")
        print("[WARN] Continuing without visualization...")
        viz_hook = None

    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards = []
    max_rewards = []
    all_successes = []

    progbar = trange(n_batches, desc="Eval batches", disable=inside_slurm())
    
    try:
        for batch_ix in progbar:
            if start_seed is None:
                seeds = None
            else:
                seeds = list(range(
                    start_seed + (batch_ix * env.num_envs),
                    start_seed + ((batch_ix + 1) * env.num_envs)
                ))
            
            if viz_hook is not None:
                rollout_data = rollout_with_viz(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    viz_hook=viz_hook,
                    seeds=seeds,
                    viz_every_n_steps=viz_every_n_steps,
                    episode_idx=batch_ix,
                )
            else:
                # 没有 hook 时使用原始 rollout
                from lerobot.scripts.lerobot_eval import rollout
                rollout_data = rollout(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seeds=seeds,
                )

            n_steps = rollout_data["done"].shape[1]
            done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
            mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
            
            batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
            sum_rewards.extend(batch_sum_rewards.tolist())
            
            batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
            max_rewards.extend(batch_max_rewards.tolist())
            
            batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
            all_successes.extend(batch_successes.tolist())

            progbar.set_postfix({
                "success_rate": f"{np.mean(all_successes[:n_episodes]) * 100:.1f}%"
            })
    
    finally:
        if viz_hook is not None:
            viz_hook.remove_hooks()

    info = {
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    return info


def main():
    init_logging()
    
    # 解析命令行参数
    import sys
    
    # 提取自定义参数
    viz_save_dir = "./smolvla_layer_viz"
    viz_every_n_steps = 10
    task_id = None  # 指定单个任务 ID
    
    new_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--viz_save_dir="):
            viz_save_dir = arg.split("=", 1)[1]
        elif arg == "--viz_save_dir" and i + 1 < len(sys.argv):
            viz_save_dir = sys.argv[i + 1]
            i += 1
        elif arg.startswith("--viz_every_n_steps="):
            viz_every_n_steps = int(arg.split("=", 1)[1])
        elif arg == "--viz_every_n_steps" and i + 1 < len(sys.argv):
            viz_every_n_steps = int(sys.argv[i + 1])
            i += 1
        elif arg.startswith("--task_id="):
            task_id = int(arg.split("=", 1)[1])
        elif arg == "--task_id" and i + 1 < len(sys.argv):
            task_id = int(sys.argv[i + 1])
            i += 1
        else:
            new_argv.append(arg)
        i += 1
    
    sys.argv = new_argv
    
    # 使用 LeRobot 的配置解析
    @parser.wrap()
    def run_eval(cfg: EvalPipelineConfig):
        logging.info(pformat(asdict(cfg)))
        
        device = get_safe_torch_device(cfg.policy.device, log=True)
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"Visualization save dir: {viz_save_dir}")

        logging.info("Making environment.")
        # 如果指定了 task_id，则只运行该任务
        if task_id is not None:
            logging.info(f"Running only task_id={task_id}")
            # 直接导入并调用 create_libero_envs，绕过配置系统
            from lerobot.envs.libero import create_libero_envs
            import gymnasium as gym
            env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
            
            # 获取相机名称
            camera_name = cfg.env.camera_name if hasattr(cfg.env, 'camera_name') and cfg.env.camera_name else "agentview_image,robot0_eye_in_hand_image"
            
            envs = create_libero_envs(
                task=cfg.env.task,
                n_envs=cfg.eval.batch_size,
                camera_name=camera_name,
                init_states=cfg.env.init_states if hasattr(cfg.env, 'init_states') else True,
                env_cls=env_cls,
                control_mode=cfg.env.control_mode if hasattr(cfg.env, 'control_mode') else "relative",
                episode_length=cfg.env.episode_length if hasattr(cfg.env, 'episode_length') else None,
                gym_kwargs={
                    "task_ids": [task_id],
                    "obs_type": "pixels_agent_pos",  # 包含机器人状态
                },
            )
        else:
            envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

        logging.info("Making policy.")
        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )
        policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )

        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

        # 只处理第一个环境（简化）
        if isinstance(envs, dict):
            # 嵌套字典结构
            first_group = list(envs.keys())[0]
            first_task = list(envs[first_group].keys())[0]
            env = envs[first_group][first_task]
        else:
            env = envs

        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            info = eval_with_viz(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                viz_save_dir=viz_save_dir,
                viz_every_n_steps=viz_every_n_steps,
                start_seed=cfg.seed,
            )
            
            print("\n" + "=" * 50)
            print("Evaluation Results:")
            print(info["aggregated"])
            print("=" * 50)

        # 关闭环境
        if isinstance(envs, dict):
            for group in envs.values():
                for task_env in group.values():
                    task_env.close()
        else:
            env.close()

        # 保存结果
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
            json.dump(info, f, indent=2)

        logging.info(f"Visualization saved to: {viz_save_dir}")
        logging.info("End of eval")
    
    run_eval()


if __name__ == "__main__":
    main()