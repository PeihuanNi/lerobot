import cv2
import numpy as np
import os

# 1. 读取视频并提取所有帧
video_path = "outputs/eval/2026-01-05/14-50-29_libero_smolvla/videos/libero_spatial_0/eval_episode_1.mp4"  # 替换为你的 MP4 文件路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

if len(frames) < 2:
    print("视频帧数不足")
    exit()

# 转换为 NumPy 数组，形状 (num_frames, H, W, 3)
frames = np.array(frames)
num_frames, height, width, channels = frames.shape

# 2. 控制比较帧数：用百分比或绝对数字
percentage = 0.2  # 用前 50% 的帧进行比较（0.0-1.0），或注释掉用绝对数字
# num_frames_to_compare = 2  # 绝对数字：用前 10 帧（注释掉 percentage 用这个）

if 'percentage' in locals():
    num_to_compare = max(1, int(num_frames * percentage))
else:
    num_to_compare = min(num_frames, num_frames_to_compare)

print(f"使用前 {num_to_compare} 帧进行比较")

# 找出不变像素：允许微小差别（阈值 5），每个像素在选定帧中值相似
threshold = 50  # 像素值差异阈值（0-255），可调整
first_frame = frames[0:1]  # 第一帧
selected_frames = frames[:num_to_compare]  # 选定帧
diff = np.abs(selected_frames - first_frame)  # 计算与第一帧的差异
unchanged_mask = np.all(diff < threshold, axis=(0, 3))  # 所有选定帧和通道差异 < 阈值，形状 (H, W)

# 3. 可视化：使用 RGBA 图像，支持透明度
# 创建 RGBA 图像（基础为第一帧 + alpha 通道）
visualization = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA
visualization[:, :, :3] = frames[0]  # BGR 通道
visualization[:, :, 3] = 255  # 默认不透明

# 不变像素：保持原色（无叠加）

# 变像素：叠加半透明紫色（突出变化，alpha=128），仅当有变像素时
if np.any(~unchanged_mask):
    purple_overlay = np.full((height, width, 4), [128, 0, 128, 255], dtype=np.uint8)  # 紫色半透明
    visualization[~unchanged_mask] = cv2.addWeighted(visualization[~unchanged_mask], 0.5, purple_overlay[~unchanged_mask], 0.5, 0)

# 4. 生成输出文件名
# 从 video_path 提取路径部分，如 "libero_spatial_0/eval_episode_1.mp4" -> "libero_spatial_0_eval_episode_1_unchanged_pixels.png"
video_dir = os.path.dirname(video_path)  # e.g., "outputs/eval/2026-01-05/14-50-29_libero_smolvla/videos"
video_filename = os.path.basename(video_path)  # e.g., "libero_spatial_0/eval_episode_1.mp4"
# 替换路径分隔符和扩展名
output_filename = video_filename.replace('/', '_').replace('\\', '_').replace('.mp4', '_unchanged_pixels.png')
output_path = os.path.join(video_dir, output_filename)  # 保存到同一目录

cv2.imwrite(output_path, visualization)
print(f"可视化结果已保存到 {output_path}")
print(f"总帧数: {num_frames}, 使用帧数: {num_to_compare}, 不变像素比例: {np.sum(unchanged_mask) / (height * width):.2%}")