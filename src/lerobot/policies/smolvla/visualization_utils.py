"""
SmolVLA Vision Model 可视化工具

可视化内容：
1. 连续两帧的 patch embedding 差异（哪些图像区域变化了）
2. 每层 Transformer 的残差（该层对 hidden state 做了多大改变）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import csv


class SmolVLAVisualizationHook:
    """捕获和可视化 SmolVLA vision model 各层输出"""
    
    def __init__(self, save_dir: str = "./smolvla_viz"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.patch_embeds = {}
        self.hooks = []
        self.frame_idx = 0
        self.num_layers = 0
        
        # 用于保存所有帧的统计信息
        self.all_stats = []
        
    def register_hooks(self, policy):
        """注册 hooks"""
        model = policy.model
        vision_model = self._find_vision_model(model)
        
        if vision_model is None:
            print("[DEBUG] Model structure:")
            self._print_model_structure(model, max_depth=4)
            raise ValueError("Cannot find vision_model")
        
        print(f"[INFO] Vision model: {type(vision_model).__name__}")
        self._register_patch_embed_hook(vision_model)
        self._register_encoder_hooks(vision_model)
        print(f"[INFO] Registered {len(self.hooks)} hooks")
        
    def _find_vision_model(self, module, visited=None):
        """递归查找 vision model"""
        if visited is None:
            visited = set()
        
        if id(module) in visited:
            return None
        visited.add(id(module))
        
        # 常见的 vision model 属性名
        vision_names = ['vision_model', 'vision_tower', 'image_encoder', 'visual']
        
        for name in vision_names:
            if hasattr(module, name):
                vm = getattr(module, name)
                if vm is not None:
                    return vm
        
        # 递归搜索子模块
        for name, child in module.named_children():
            result = self._find_vision_model(child, visited)
            if result is not None:
                return result
        
        return None
    
    def _print_model_structure(self, module, prefix="", max_depth=3, depth=0):
        """打印模型结构用于调试"""
        if depth >= max_depth:
            return
        for name, child in module.named_children():
            print(f"{prefix}{name}: {type(child).__name__}")
            self._print_model_structure(child, prefix + "  ", max_depth, depth + 1)
    
    def _register_patch_embed_hook(self, vision_model):
        """注册 patch embedding hook"""
        embed_module = None
        
        # 尝试不同的属性名
        for name in ['embeddings', 'patch_embed', 'patch_embedding', 'embed']:
            if hasattr(vision_model, name):
                embed_module = getattr(vision_model, name)
                break
        
        if embed_module is not None:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.patch_embeds[self.frame_idx] = output.detach().cpu()
            
            self.hooks.append(embed_module.register_forward_hook(hook))
            print(f"[INFO] Registered patch embed hook on {type(embed_module).__name__}")
    
    def _register_encoder_hooks(self, vision_model):
        """注册 encoder layers hooks"""
        encoder = None
        layers = None
        
        # 查找 encoder
        for name in ['encoder', 'transformer', 'blocks']:
            if hasattr(vision_model, name):
                encoder = getattr(vision_model, name)
                break
        
        if encoder is None:
            encoder = vision_model
        
        # 查找 layers
        for name in ['layers', 'blocks', 'layer']:
            if hasattr(encoder, name):
                layers = getattr(encoder, name)
                break
        
        if layers is None:
            return
        
        self.num_layers = len(layers)
        print(f"[INFO] Found {self.num_layers} encoder layers")
        
        for layer_idx, layer in enumerate(layers):
            def make_hook(idx):
                def hook(module, input, output):
                    inp = input[0] if isinstance(input, tuple) else input
                    out = output[0] if isinstance(output, tuple) else output
                    self.layer_inputs[(self.frame_idx, idx)] = inp.detach().cpu()
                    self.layer_outputs[(self.frame_idx, idx)] = out.detach().cpu()
                return hook
            
            self.hooks.append(layer.register_forward_hook(make_hook(layer_idx)))
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def increment_frame(self):
        """增加帧计数"""
        self.frame_idx += 1
        
    def reset(self):
        """重置状态（新 episode 开始时调用）"""
        self.layer_inputs.clear()
        self.layer_outputs.clear()
        self.patch_embeds.clear()
        self.frame_idx = 0
        # 不清除 all_stats，保留跨 episode 的统计
        
    def compute_patch_embed_diff(self) -> Optional[torch.Tensor]:
        """计算当前帧与前一帧 patch embedding 的差异"""
        if self.frame_idx < 1:
            return None
        
        curr_embed = self.patch_embeds.get(self.frame_idx)
        prev_embed = self.patch_embeds.get(self.frame_idx - 1)
        
        if curr_embed is None or prev_embed is None:
            return None
        
        diff = (curr_embed - prev_embed).norm(dim=-1)
        return diff
    
    def compute_layer_residual(self, layer_idx: int) -> Optional[torch.Tensor]:
        """计算指定层的残差"""
        inp = self.layer_inputs.get((self.frame_idx, layer_idx))
        out = self.layer_outputs.get((self.frame_idx, layer_idx))
        
        if inp is None or out is None:
            return None
        
        residual = (out - inp).norm(dim=-1)
        return residual
    
    def collect_frame_stats(self, episode_idx: int = 0):
        """收集当前帧的统计信息"""
        for layer_idx in range(self.num_layers):
            inp = self.layer_inputs.get((self.frame_idx, layer_idx))
            out = self.layer_outputs.get((self.frame_idx, layer_idx))
            
            if inp is not None and out is not None:
                residual = (out - inp).abs()
                residual_l2 = (out - inp).norm()
                
                stats = {
                    'episode': episode_idx,
                    'frame': self.frame_idx,
                    'layer': layer_idx,
                    'input_norm': inp.float().norm().item(),
                    'output_norm': out.float().norm().item(),
                    'residual_mean': residual.float().mean().item(),
                    'residual_max': residual.float().max().item(),
                    'residual_l2': residual_l2.float().item(),
                }
                self.all_stats.append(stats)
    
    def save_all_stats(self, episode_idx: int = 0):
        """保存所有统计信息到 CSV"""
        if not self.all_stats:
            print("[WARN] No stats to save")
            return
        
        csv_path = self.save_dir / f"layer_stats_ep{episode_idx:03d}.csv"
        
        fieldnames = ['episode', 'frame', 'layer', 'input_norm', 'output_norm', 
                      'residual_mean', 'residual_max', 'residual_l2']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_stats)
        
        print(f"[INFO] Saved {len(self.all_stats)} stats to {csv_path}")
        
        # 清空以便下一个 episode
        self.all_stats.clear()
    
    def print_frame_stats(self):
        """打印当前帧的统计信息"""
        print(f"\n=== Frame {self.frame_idx} Stats ===")
        for layer_idx in range(self.num_layers):
            inp = self.layer_inputs.get((self.frame_idx, layer_idx))
            out = self.layer_outputs.get((self.frame_idx, layer_idx))
            
            if inp is not None and out is not None:
                residual = (out - inp).abs()
                print(f"  Layer {layer_idx:2d}: "
                      f"in_norm={inp.float().norm():.2f}, "
                      f"out_norm={out.float().norm():.2f}, "
                      f"res_mean={residual.float().mean():.4f}, "
                      f"res_max={residual.float().max():.4f}")
    
    def visualize_frame_comparison(
        self, 
        prev_image: np.ndarray, 
        curr_image: np.ndarray,
        num_layers_to_show: int = 4,
        episode_idx: int = 0
    ):
        """可视化前后帧对比和各层残差"""
        
        # 收集统计信息
        self.collect_frame_stats(episode_idx)
        
        # 选择要显示的层（均匀分布）
        if self.num_layers > num_layers_to_show:
            step = self.num_layers // num_layers_to_show
            selected_layers = list(range(0, self.num_layers, step))[:num_layers_to_show]
        else:
            selected_layers = list(range(self.num_layers))
        
        fig, axes = plt.subplots(2, len(selected_layers) + 2, 
                                  figsize=(4 * (len(selected_layers) + 2), 8))
        
        # 第一行：图像和 patch embed 差异
        axes[0, 0].imshow(prev_image)
        axes[0, 0].set_title(f"Frame {self.frame_idx - 1}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(curr_image)
        axes[0, 1].set_title(f"Frame {self.frame_idx}")
        axes[0, 1].axis('off')
        
        # Patch embed 差异
        patch_diff = self.compute_patch_embed_diff()
        if patch_diff is not None:
            diff_map = patch_diff[0].float().numpy()  # 先转 float32 再转 numpy
            num_patches = diff_map.shape[0]
            side = int(np.sqrt(num_patches))
            if side * side == num_patches:
                diff_map_2d = diff_map.reshape(side, side)
            else:
                diff_map_2d = diff_map.reshape(1, -1)
            
            im = axes[1, 0].imshow(diff_map_2d, cmap='hot')
            axes[1, 0].set_title("Patch Embed Diff")
            plt.colorbar(im, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, "N/A", ha='center', va='center')
            axes[1, 0].set_title("Patch Embed Diff")
        axes[1, 0].axis('off')
        
        axes[1, 1].axis('off')
        
        # 各层残差
        for i, layer_idx in enumerate(selected_layers):
            residual = self.compute_layer_residual(layer_idx)
            if residual is not None:
                res_map = residual[0].float().numpy()  # 先转 float32 再转 numpy
                num_tokens = res_map.shape[0]
                side = int(np.sqrt(num_tokens))
                if side * side == num_tokens:
                    res_map_2d = res_map.reshape(side, side)
                else:
                    res_map_2d = res_map.reshape(1, -1)
                
                im = axes[0, i + 2].imshow(res_map_2d, cmap='viridis')
                axes[0, i + 2].set_title(f"Layer {layer_idx}\nResidual")
                plt.colorbar(im, ax=axes[0, i + 2])
                
                # 统计信息
                stats_text = f"Mean: {residual.float().mean():.4f}\nMax: {residual.float().max():.4f}"
                axes[1, i + 2].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10)
                axes[1, i + 2].set_title(f"Layer {layer_idx} Stats")
            else:
                axes[0, i + 2].text(0.5, 0.5, "N/A", ha='center', va='center')
                axes[0, i + 2].set_title(f"Layer {layer_idx}")
            
            axes[0, i + 2].axis('off')
            axes[1, i + 2].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f"ep{episode_idx:03d}_frame_{self.frame_idx:04d}.png"
        plt.savefig(save_path, dpi=100)
        plt.close()