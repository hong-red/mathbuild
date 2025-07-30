import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 定义图像尺寸和颜色通道数
height, width = 64, 64
channels = 3

# 模拟每个颜色通道的响应数据
data = {}
sheets = ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']
for sheet in sheets:
    if sheet in ['R_R', 'G_G', 'B_B']:  # 主通道（例如 R 色光下的 R 通道响应）
        data[sheet] = np.random.uniform(200, 239, size=(height, width))
    else:  # 串扰通道（例如 R 色光下的 G 或 B 通道响应）
        data[sheet] = np.random.uniform(5, 20, size=(height, width))

# 构造每种基色照明下的 RGB 响应张量
red_measured = np.stack([data['R_R'], data['R_G'], data['R_B']], axis=-1)
green_measured = np.stack([data['G_R'], data['G_G'], data['G_B']], axis=-1)
blue_measured = np.stack([data['B_R'], data['B_G'], data['B_B']], axis=-1)

# 定义目标颜色向量（单位为 220）
target_red = np.array([220, 0, 0])
target_green = np.array([0, 220, 0])
target_blue = np.array([0, 0, 220])
target_white = np.array([220, 220, 220])

# 初始化混色矩阵（每个像素点一个 3x3 矩阵）
mixing_matrices = np.zeros((height, width, 3, 3))
for i in range(height):
    for j in range(width):
        # 构建测量矩阵（每列为一种基色的响应 / 220）
        measured_matrix = np.column_stack([
            red_measured[i, j] / 220,
            green_measured[i, j] / 220,
            blue_measured[i, j] / 220
        ])
        mixing_matrices[i, j] = measured_matrix

# 计算校正输入（目标颜色 * 逆混色矩阵）
corrected_inputs = np.zeros((height, width, 3, 4))  # 最后一维为四种目标色
for i in range(height):
    for j in range(width):
        try:
            # 求逆矩阵
            inv_matrix = np.linalg.inv(mixing_matrices[i, j])
            # 对每个目标色计算需要的输入值
            corrected_inputs[i, j, :, 0] = inv_matrix @ target_red
            corrected_inputs[i, j, :, 1] = inv_matrix @ target_green
            corrected_inputs[i, j, :, 2] = inv_matrix @ target_blue
            corrected_inputs[i, j, :, 3] = inv_matrix @ target_white
        except np.linalg.LinAlgError:
            # 若矩阵不可逆，使用兜底方法估算输入值
            corrected_inputs[i, j, 0, 0] = 220 * (220 / (data['R_R'][i, j] + 1e-6))
            corrected_inputs[i, j, 1, 1] = 220 * (220 / (data['G_G'][i, j] + 1e-6))
            corrected_inputs[i, j, 2, 2] = 220 * (220 / (data['B_B'][i, j] + 1e-6))
            corrected_inputs[i, j, :, 3] = 220  # 白色默认全开

# 将输入值限制在 0 到 255 之间
corrected_inputs = np.clip(corrected_inputs, 0, 255)

# 初始化校正后输出
corrected_red_output = np.zeros_like(red_measured)
corrected_green_output = np.zeros_like(green_measured)
corrected_blue_output = np.zeros_like(blue_measured)
corrected_white_output = np.zeros_like(red_measured)

# 模拟校正后的颜色输出
for i in range(height):
    for j in range(width):
        corrected_red_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 0]
        corrected_green_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 1]
        corrected_blue_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 2]
        corrected_white_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 3]

# 输出限制在 0-255 并转换为整数（图像格式）
corrected_red_output = np.clip(corrected_red_output, 0, 255).astype(np.uint8)
corrected_green_output = np.clip(corrected_green_output, 0, 255).astype(np.uint8)
corrected_blue_output = np.clip(corrected_blue_output, 0, 255).astype(np.uint8)
corrected_white_output = np.clip(corrected_white_output, 0, 255).astype(np.uint8)

# 打印每种基色输出的均值
print("Corrected Red Output (mean):", np.mean(corrected_red_output, axis=(0, 1)))
print("Corrected Green Output (mean):", np.mean(corrected_green_output, axis=(0, 1)))
print("Corrected Blue Output (mean):", np.mean(corrected_blue_output, axis=(0, 1)))

# =================== 可视化部分 =================== #

# 可视化：原始 vs 校正后的图像
plt.figure(figsize=(20, 10))

# 红色基色
plt.subplot(2, 4, 1)
plt.imshow(red_measured.astype(np.uint8))
plt.title("Original Red Base")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(corrected_red_output)
plt.title("Corrected Red Base")
plt.axis('off')

# 绿色基色
plt.subplot(2, 4, 2)
plt.imshow(green_measured.astype(np.uint8))
plt.title("Original Green Base")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(corrected_green_output)
plt.title("Corrected Green Base")
plt.axis('off')

# 蓝色基色
plt.subplot(2, 4, 3)
plt.imshow(blue_measured.astype(np.uint8))
plt.title("Original Blue Base")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(corrected_blue_output)
plt.title("Corrected Blue Base")
plt.axis('off')

# 白色基色
plt.subplot(2, 4, 4)
plt.imshow(np.stack([data['R_R'], data['G_G'], data['B_B']], axis=-1).astype(np.uint8))
plt.title("Original White Base")
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(corrected_white_output)
plt.title("Corrected White Base")
plt.axis('off')

plt.tight_layout()
plt.show()

# 差异热图：原始 - 目标值
plt.figure(figsize=(20, 5))

# 红色差异图
red_diff = red_measured[:, :, 0] - target_red[0]
plt.subplot(1, 4, 1)
plt.imshow(red_diff, cmap='RdBu', vmin=-50, vmax=50)
plt.title("Red Base Diff (R - 220)")
plt.colorbar()
plt.axis('off')

# 绿色差异图
green_diff = green_measured[:, :, 1] - target_green[1]
plt.subplot(1, 4, 2)
plt.imshow(green_diff, cmap='RdBu', vmin=-50, vmax=50)
plt.title("Green Base Diff (G - 220)")
plt.colorbar()
plt.axis('off')

# 蓝色差异图
blue_diff = blue_measured[:, :, 2] - target_blue[2]
plt.subplot(1, 4, 3)
plt.imshow(blue_diff, cmap='RdBu', vmin=-50, vmax=50)
plt.title("Blue Base Diff (B - 220)")
plt.colorbar()
plt.axis('off')

# 白色差异图（均值）
white_diff = np.mean(np.stack([data['R_R'], data['G_G'], data['B_B']], axis=-1) - target_white, axis=-1)
plt.subplot(1, 4, 4)
plt.imshow(white_diff, cmap='RdBu', vmin=-50, vmax=50)
plt.title("White Base Diff (Avg)")
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()

# 复合色校正测试：对随机 RGB 输入进行校正
random_rgb_input = np.random.uniform(0, 255, size=(height, width, 3))
corrected_rgb_output = np.zeros_like(random_rgb_input)

# 应用每个像素的逆矩阵进行校正
for i in range(height):
    for j in range(width):
        try:
            inv_matrix = np.linalg.inv(mixing_matrices[i, j])
            target_rgb = random_rgb_input[i, j]
            corrected_input = inv_matrix @ target_rgb
            corrected_rgb_output[i, j] = mixing_matrices[i, j] @ corrected_input
        except np.linalg.LinAlgError:
            corrected_rgb_output[i, j] = random_rgb_input[i, j]  # 兜底返回原输入

# 限制在 0~255 并转为整数图像
corrected_rgb_output = np.clip(corrected_rgb_output, 0, 255).astype(np.uint8)

# 输出复合图像结果
print("Corrected RGB Output (min, max, mean):", np.min(corrected_rgb_output), np.max(corrected_rgb_output), np.mean(corrected_rgb_output, axis=(0, 1)))

plt.figure(figsize=(10, 5))
plt.imshow(corrected_rgb_output)
plt.title(a)
plt.axis('off')
plt.show()

# 记录校正时间
from datetime import datetime
correction_time = datetime.now().strftime("%Y-%m-%d %I:%M %p HKT")
print(f"Correction applied at: {correction_time}")
