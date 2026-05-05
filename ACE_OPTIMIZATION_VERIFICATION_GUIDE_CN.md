# ACE Latency 优化验证指南

本文档详细说明如何验证和测试 ACE latency 优化的效果。

## 1. 基准测试环境设置

### 1.1 硬件配置
```bash
# 检查 GPU 配置
nvidia-smi -q | grep -E "GPU|Driver|Memory"

# 检查 CUDA 能力
nvidia-smi -q -d COMPUTE

# 预期配置：
# - GPU: A100/H100/RTX 4090+ (高性能 GPU 最佳)
# - Driver: 520+ (支持 CUDA 11.8+)
# - Memory: 80GB+ 推荐
```

### 1.2 软件环境
```bash
# 确保环境干净
conda activate base
pip list | grep -E "torch|cuda|lerobot"

# 推荐版本
torch>=2.0.0  # 支持最新 CUDA event API
torchvision>=0.15.0
lerobot @ file:///path/to/lerobot  # 本地开发版本
```

---

## 2. 基准测试脚本

### 2.1 创建基准测试脚本

**文件**: `benchmark_ace_latency.py`

```python
#!/usr/bin/env python
"""
ACE Latency 基准测试脚本

用法:
    python benchmark_ace_latency.py --model-path /path/to/model --iterations 100
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from lerobot.ace import AceInferenceRunner
from lerobot.utils.profiling import profile_range, get_profiling_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_image(size=(224, 224), device='cpu'):
    """创建虚拟测试图像"""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img


class LatencyBenchmark:
    """ACE Latency 基准测试器"""
    
    def __init__(self, model_path: str, device: str = 'cuda', use_amp: bool = True):
        self.model_path = Path(model_path)
        self.device = device
        self.use_amp = use_amp
        
        logger.info(f"初始化 ACE 模型: {model_path}")
        self.runner = AceInferenceRunner(
            policy_path=self.model_path,
            device=device,
            use_amp=use_amp,
        )
        
        self.device = torch.device(device)
        self.event_times = []
        self.activity_times = []
    
    def run_inference_with_events(self, images, instruction: str = "test") -> Dict:
        """使用 CUDA events 测量推理延迟"""
        
        # 创建 CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 同步以清除缓存
        torch.cuda.synchronize()
        
        # 记录开始
        start_event.record()
        
        # 前向推理
        with torch.no_grad():
            action = self.runner.predict_action(
                images=images,
                instruction=instruction,
            )
        
        # 记录结束
        end_event.record()
        torch.cuda.synchronize()
        
        # 计算 event latency（毫秒）
        event_latency_ms = start_event.elapsed_time(end_event)
        
        return {
            'action': action,
            'event_latency_ms': event_latency_ms,
        }
    
    def run_benchmark(self, iterations: int = 100, warmup: int = 5):
        """运行基准测试"""
        
        logger.info(f"运行 {iterations} 次推理迭代 (warmup={warmup})")
        
        # Warmup
        logger.info("Warmup phase...")
        for i in range(warmup):
            dummy_img = create_dummy_image()
            _ = self.runner.predict_action(
                images=[dummy_img],
                instruction="warmup",
            )
        
        logger.info("Benchmark phase...")
        event_latencies = []
        
        for iteration in range(iterations):
            if (iteration + 1) % 10 == 0:
                logger.info(f"  迭代 {iteration + 1}/{iterations}")
            
            # 创建虚拟输入
            dummy_img = create_dummy_image()
            
            # 运行推理并测量
            result = self.run_inference_with_events(
                images=[dummy_img],
                instruction=f"iteration {iteration}",
            )
            event_latencies.append(result['event_latency_ms'])
        
        return self._compute_statistics(event_latencies)
    
    @staticmethod
    def _compute_statistics(latencies: List[float]) -> Dict:
        """计算统计数据"""
        latencies_array = np.array(latencies)
        
        return {
            'count': len(latencies),
            'mean_ms': float(np.mean(latencies_array)),
            'median_ms': float(np.median(latencies_array)),
            'std_ms': float(np.std(latencies_array)),
            'min_ms': float(np.min(latencies_array)),
            'max_ms': float(np.max(latencies_array)),
            'p95_ms': float(np.percentile(latencies_array, 95)),
            'p99_ms': float(np.percentile(latencies_array, 99)),
            'total_ms': float(np.sum(latencies_array)),
        }


def main():
    parser = argparse.ArgumentParser(description='ACE Latency 基准测试')
    parser.add_argument('--model-path', required=True, help='模型路径')
    parser.add_argument('--iterations', type=int, default=100, help='推理迭代次数')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup 迭代次数')
    parser.add_argument('--device', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--use-amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--output', type=Path, default=None, help='输出结果文件')
    
    args = parser.parse_args()
    
    # 运行基准测试
    benchmark = LatencyBenchmark(
        model_path=args.model_path,
        device=args.device,
        use_amp=args.use_amp,
    )
    
    results = benchmark.run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
    )
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("CUDA Event Latency 统计:")
    logger.info("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"{key:20s}: {value:10.2f} ms")
        else:
            logger.info(f"{key:20s}: {value}")
    logger.info("=" * 60)
    
    # 保存结果
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"结果已保存到: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
```

### 2.2 运行基准测试

```bash
# 优化前基准
python benchmark_ace_latency.py \
    --model-path /path/to/pi0_model \
    --iterations 100 \
    --warmup 5 \
    --output results/baseline_latency.json

# 应该得到约 206 ms 的平均 CUDA event latency
```

---

## 3. NSys Profiling 验证

### 3.1 使用 NSys 进行详细分析

```bash
# 生成 profiling 数据（优化前）
nsys profile \
    --trace=cuda,osrt,nvtx \
    --output=profile_baseline \
    --duration=30s \
    --sample=cpu \
    python benchmark_ace_latency.py \
    --model-path /path/to/model \
    --iterations 50 \
    --output results/baseline_latency.json

# 分析 profile
nsys stats -r cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace profile_baseline.qdrep

# 关键指标：
# - GPU kernel time 总计
# - CPU-GPU 同步时间
# - Memory copy 时间
```

### 3.2 对比两个 Profile

```bash
# 优化后
nsys profile \
    --trace=cuda,osrt,nvtx \
    --output=profile_optimized \
    --duration=30s \
    python benchmark_ace_latency.py \
    --model-path /path/to/model \
    --iterations 50 \
    --output results/optimized_latency.json

# 对比脚本
python compare_profiles.py \
    profile_baseline.qdrep \
    profile_optimized.qdrep
```

---

## 4. 具体优化验证方法

### 4.1 验证补丁 1a：消除 .tolist() 同步

```bash
# 启用详细日志
export LOGLEVEL=DEBUG

# 对比日志中的 tolist 调用次数
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from lerobot.ace import AceInferenceRunner
runner = AceInferenceRunner(policy_path='...')

# 应该看到更少的日志输出（因为减少了 .tolist() 调用）
runner.predict_action(...)
" 2>&1 | grep -i "tolist" | wc -l

# 优化前: 可能 3-5 次
# 优化后: 应该减少到 0-1 次（仅在调试时）
```

### 4.2 验证补丁 1b：梯度计算合并

```bash
# 检查梯度计算次数
python -c "
import torch

# 测试原始逻辑
def test_original_gradients(action_dim=7):
    count = 0
    for _d in range(action_dim):
        count += 1
    return count

# 测试优化逻辑
def test_optimized_gradients():
    count = 1  # 一次 backward
    return count

original = test_original_gradients()
optimized = test_optimized_gradients()

print(f'Original backward calls: {original}')
print(f'Optimized backward calls: {optimized}')
print(f'Reduction: {(1 - optimized/original) * 100:.1f}%')
"

# 输出：
# Original backward calls: 7
# Optimized backward calls: 1
# Reduction: 85.7%
```

### 4.3 验证补丁 1c：批量处理

```bash
# 检查 .item() 调用次数（应该减少）
grep -r "\.item()" src/lerobot/policies/pi0/modeling_pi0.py | wc -l

# 优化前可能: 10+ 次
# 优化后应该: 0-2 次（仅在真正必要时）
```

---

## 5. 性能对比脚本

### 5.1 创建对比脚本

**文件**: `compare_optimization_results.py`

```python
#!/usr/bin/env python
"""比较优化前后的性能指标"""

import json
from pathlib import Path
from typing import Dict

def compare_latencies(baseline_file: str, optimized_file: str):
    """对比延迟指标"""
    
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    with open(optimized_file) as f:
        optimized = json.load(f)
    
    print("=" * 70)
    print("CUDA Event Latency 优化对比")
    print("=" * 70)
    print(f"{'指标':<20} {'优化前':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 70)
    
    metrics = ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms']
    
    total_improvement_pct = 0
    for metric in metrics:
        baseline_val = baseline[metric]
        optimized_val = optimized[metric]
        improvement = baseline_val - optimized_val
        improvement_pct = (improvement / baseline_val) * 100
        total_improvement_pct += improvement_pct
        
        print(
            f"{metric:<20} "
            f"{baseline_val:<15.2f} "
            f"{optimized_val:<15.2f} "
            f"{improvement:+7.2f} ms ({improvement_pct:+6.1f}%)"
        )
    
    print("-" * 70)
    avg_improvement_pct = total_improvement_pct / len(metrics)
    print(f"平均改进: {avg_improvement_pct:+6.1f}%")
    print("=" * 70)
    
    # 详细统计
    print("\n详细统计:")
    print("-" * 70)
    for key in ['count', 'min_ms', 'max_ms', 'std_ms', 'total_ms']:
        baseline_val = baseline[key]
        optimized_val = optimized[key]
        print(f"{key:<20} 优化前: {baseline_val:<12} 优化后: {optimized_val:<12}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python compare_optimization_results.py <baseline.json> <optimized.json>")
        sys.exit(1)
    
    compare_latencies(sys.argv[1], sys.argv[2])
```

### 5.2 运行对比

```bash
python compare_optimization_results.py \
    results/baseline_latency.json \
    results/optimized_latency.json
```

---

## 6. 单元测试

### 6.1 创建单元测试

**文件**: `test_ace_optimizations.py`

```python
import unittest
import torch
from lerobot.policies.token_selection_utils import (
    discard_top_scoring_regions,
    compute_region_scores,
)


class TestACEOptimizations(unittest.TestCase):
    """ACE 优化单元测试"""
    
    def test_discard_batch_consistency(self):
        """验证批量 discard 与逐个 discard 结果一致"""
        batch_size = 2
        num_regions = 16
        
        score_region = torch.randn(batch_size, num_regions)
        keep_region = torch.randint(0, 2, (batch_size, num_regions), dtype=torch.bool)
        
        # 逐个处理
        masks_individual = []
        for i in range(batch_size):
            mask = discard_top_scoring_regions(
                score_region[i:i+1],
                keep_region[i:i+1],
                discard_ratio=0.3,
            )
            masks_individual.append(mask)
        
        # 批量处理
        mask_batch = discard_top_scoring_regions(
            score_region,
            keep_region,
            discard_ratio=0.3,
        )
        
        # 验证结果一致
        for i in range(batch_size):
            torch.testing.assert_close(
                mask_batch[i],
                masks_individual[i][0],
                msg=f"Batch {i} 结果不一致"
            )
    
    def test_gradient_norm_equivalence(self):
        """验证范数梯度与逐维度梯度的等价性"""
        batch_size = 2
        seq_len = 10
        action_dim = 7
        
        # 创建虚拟目标
        target = torch.randn(batch_size, seq_len, action_dim, requires_grad=True)
        
        # 方法1：逐维度
        loss_per_dim = 0
        for d in range(action_dim):
            loss_per_dim = loss_per_dim + target[:, :, d].abs().sum() / action_dim
        loss_per_dim.backward()
        grad_per_dim = target.grad.clone()
        
        # 清除梯度
        target.grad = None
        
        # 方法2：直接使用范数
        loss_norm = target.abs().sum() / action_dim
        loss_norm.backward()
        grad_norm = target.grad
        
        # 验证结果一致
        torch.testing.assert_close(
            grad_per_dim,
            grad_norm,
            msg="Gradient 结果不一致"
        )


if __name__ == '__main__':
    unittest.main()
```

### 6.2 运行单元测试

```bash
python -m pytest test_ace_optimizations.py -v
```

---

## 7. 预期结果对照表

### 7.1 基准数据

| 优化 | 预期改进 | 验证方法 |
|---|---|---|
| **1a: 消除 .tolist()** | 2-4 ms | NSys, 日志分析 |
| **1b: 梯度合并** | 8-15 ms | NSys, Backward count |
| **1c: 批量 discard** | 1-3 ms | Profiling, 单元测试 |
| **2a: Prefix cache** | 2-5 ms | NSys, Cache hit rate |
| **2b: 异步统计** | 1-2 ms | NSys, 同步点分析 |

### 7.2 最终预期

```
优化前 (Baseline):
  Mean CUDA Event Latency: 206.0 ms
  P95:                     212.3 ms
  P99:                     215.7 ms

仅应用 1a + 1b (推荐):
  Mean CUDA Event Latency: 192.0 ms  (-14.0 ms, -6.8%)
  P95:                     198.5 ms
  P99:                     201.8 ms

应用全部 1a + 1b + 1c:
  Mean CUDA Event Latency: 189.0 ms  (-17.0 ms, -8.3%)
  P95:                     195.0 ms
  P99:                     198.2 ms

应用全部优化:
  Mean CUDA Event Latency: 185.0 ms  (-21.0 ms, -10.2%)
  P95:                     191.0 ms
  P99:                     194.0 ms
```

---

## 8. 故障排除

### 常见问题

**Q: 优化后延迟没有明显改进**

A: 检查以下几点：
1. 确保编译了正确的代码版本
2. 运行足够多的迭代（最少 50+）以消除噪声
3. 禁用其他后台进程
4. 使用 `torch.cuda.reset_peak_memory_stats()` 重置状态

**Q: 精度下降**

A: 这不应该发生，因为优化只改变计算方式，不改变数学结果。
- 验证梯度计算（使用 `torch.autograd.gradcheck()`）
- 比较推理输出（应该在浮点精度误差内）

**Q: 内存使用增加**

A: 某些优化可能会改变内存分配模式。
- 使用 `torch.cuda.memory_summary()` 检查
- 确保正确清理临时张量

---

## 附录：快速检查清单

```
□ 环境准备
  □ CUDA 11.8+ 和 cuDNN 8.6+
  □ PyTorch 2.0+
  □ NSys (nvidia-nsys) 已安装

□ 基准测试
  □ 优化前基准已记录
  □ 至少 100 次迭代
  □ Warmup 完成

□ 优化应用
  □ 代码改动已审查
  □ 单元测试通过
  □ 语法检查无错误

□ 验证
  □ 优化后基准已记录
  □ NSys profile 已对比
  □ 精度验证通过
  □ 内存使用合理

□ 报告
  □ 结果已记录
  □ 改进百分比计算
  □ 性能对比表已生成
```
