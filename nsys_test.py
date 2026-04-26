import torch
import torch.cuda.nvtx as nvtx

def my_function(a, b):
    # 使用 nvtx 范围标记，方便在 nsys 中定位
    nvtx.range_push("Matrix_Multiplication")
    c = torch.matmul(a, b)
    nvtx.range_pop()
    
    nvtx.range_push("ReLU_Activation")
    d = torch.relu(c)
    nvtx.range_pop()
    
    return d

# 准备数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)

# 热身（Warm up）：排除 CUDA 初始化带来的延迟
for _ in range(10):
    _ = my_function(a, b)

# 开始正式分析
nvtx.range_push("Profile_Loop")
for _ in range(5):
    result = my_function(a, b)
torch.cuda.synchronize() # 确保所有 GPU 任务完成
nvtx.range_pop()