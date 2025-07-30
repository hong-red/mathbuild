import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the hybrid alignment functions from the document
def centroid_align(source, target):
    """
    质心对齐：平移source使其质心与target重合
    :param source: 源三角形顶点 (3×2)
    :param target: 目标三角形顶点 (3×2)
    :return: 平移后的源三角形
    """
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    return source - source_centroid + target_centroid

def weighted_kabsch(P, Q, weights=None):
    """
    带权重的Kabsch算法
    :param P: 待对齐三角形 (3×2)
    :param Q: 目标三角形 (3×2)
    :param weights: 顶点权重 (None时退化为普通最小二乘)
    :return: 最优旋转矩阵R和平移向量t
    """
    if weights is None:
        weights = np.ones(len(P))
    
    # 计算加权质心
    w_sum = np.sum(weights)
    centroid_P = np.sum(P * weights[:, None], axis=0) / w_sum
    centroid_Q = np.sum(Q * weights[:, None], axis=0) / w_sum
    
    # 中心化
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 计算协方差矩阵
    H = (P_centered.T * weights) @ Q_centered
    
    # SVD分解
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算平移
    t = centroid_Q - R @ centroid_P
    
    return R, t

def hybrid_alignment(source_tri, target_tri, weights=None):
    """
    混合对齐流程
    :param source_tri: 待对齐三角形 (3×2)
    :param target_tri: 目标三角形 (3×2)
    :param weights: 顶点权重数组 (3,)
    :return: 对齐后的源三角形, 旋转矩阵R, 平移向量t
    """
    # 阶段1：质心粗对齐
    rough_aligned = centroid_align(source_tri, target_tri)
    
    # 阶段2：加权最小二乘精调
    R, t = weighted_kabsch(rough_aligned, target_tri, weights)
    
    # 应用最终变换
    final_aligned = (rough_aligned @ R.T) + t
    
    return final_aligned, R, t

# Step 2: Simulate the 64x64 grids for each channel
np.random.seed(42)  # For reproducibility
height, width = 64, 64
channels = 3

# Simulate data for R_R, R_G, R_B, etc., with values similar to the problem (200-239)
data = {}
sheets = ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']
for sheet in sheets:
    if sheet in ['R_R', 'G_G', 'B_B']:  # Primary channels
        data[sheet] = np.random.uniform(200, 239, size=(height, width))
    else:  # Crosstalk channels (e.g., R_G, R_B), smaller values
        data[sheet] = np.random.uniform(5, 20, size=(height, width))

# Construct RGB vectors for each color
red_measured = np.stack([data['R_R'], data['R_G'], data['R_B']], axis=-1)  # Shape: (64, 64, 3)
green_measured = np.stack([data['G_R'], data['G_G'], data['G_B']], axis=-1)
blue_measured = np.stack([data['B_R'], data['B_G'], data['B_B']], axis=-1)

# Target outputs for each color
target_red = np.array([220, 0, 0])
target_green = np.array([0, 220, 0])
target_blue = np.array([0, 0, 220])
target_white = np.array([220, 220, 220])

# Compute the color mixing matrix for each pixel
mixing_matrices = np.zeros((height, width, 3, 3))
for i in range(height):
    for j in range(width):
        measured_matrix = np.column_stack([
            red_measured[i, j] / 220,
            green_measured[i, j] / 220,
            blue_measured[i, j] / 220
        ])
        mixing_matrices[i, j] = measured_matrix

# Compute the corrected inputs
corrected_inputs = np.zeros((height, width, 3, 4))  # 4 colors: red, green, blue, white
for i in range(height):
    for j in range(width):
        try:
            inv_matrix = np.linalg.inv(mixing_matrices[i, j])
            corrected_inputs[i, j, :, 0] = inv_matrix @ target_red    # Red
            corrected_inputs[i, j, :, 1] = inv_matrix @ target_green  # Green
            corrected_inputs[i, j, :, 2] = inv_matrix @ target_blue   # Blue
            corrected_inputs[i, j, :, 3] = inv_matrix @ target_white  # White
        except np.linalg.LinAlgError:
            corrected_inputs[i, j, 0, 0] = 220 * (220 / (data['R_R'][i, j] + 1e-6))
            corrected_inputs[i, j, 1, 1] = 220 * (220 / (data['G_G'][i, j] + 1e-6))
            corrected_inputs[i, j, 2, 2] = 220 * (220 / (data['B_B'][i, j] + 1e-6))
            corrected_inputs[i, j, :, 3] = 220  # Fallback for white: equal input

# Clip corrected inputs to valid range (0 to 255)
corrected_inputs = np.clip(corrected_inputs, 0, 255)

# Simulate the corrected output
corrected_red_output = np.zeros_like(red_measured)
corrected_green_output = np.zeros_like(green_measured)
corrected_blue_output = np.zeros_like(blue_measured)
corrected_white_output = np.zeros_like(red_measured)

for i in range(height):
    for j in range(width):
        corrected_red_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 0]
        corrected_green_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 1]
        corrected_blue_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 2]
        corrected_white_output[i, j] = mixing_matrices[i, j] @ corrected_inputs[i, j, :, 3]

# Clip outputs to valid range
corrected_red_output = np.clip(corrected_red_output, 0, 255).astype(np.uint8)
corrected_green_output = np.clip(corrected_green_output, 0, 255).astype(np.uint8)
corrected_blue_output = np.clip(corrected_blue_output, 0, 255).astype(np.uint8)
corrected_white_output = np.clip(corrected_white_output, 0, 255).astype(np.uint8)

# Step 3: Extract gamut triangles in CIE 1931 chromaticity coordinates
# Simulate display gamut triangle (approximate from measured data averages)
display_red_xy = np.array([np.mean(data['R_R'] / (data['R_R'] + data['R_G'] + data['R_B'])),
                          np.mean(data['R_G'] / (data['R_R'] + data['R_G'] + data['R_B']))])
display_green_xy = np.array([np.mean(data['G_R'] / (data['G_R'] + data['G_G'] + data['G_B'])),
                            np.mean(data['G_G'] / (data['G_R'] + data['G_G'] + data['G_B']))])
display_blue_xy = np.array([np.mean(data['B_R'] / (data['B_R'] + data['B_G'] + data['B_B'])),
                           np.mean(data['B_G'] / (data['B_R'] + data['B_G'] + data['B_B']))])

display_tri = np.array([display_red_xy, display_green_xy, display_blue_xy])  # Shape: (3, 2)

# Target gamut triangle (BT.2020)
bt2020_tri = np.array([[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]])  # BT.2020 triangle

# Step 4: Align the display gamut to BT.2020 using hybrid alignment
weights = np.array([0.3, 0.4, 0.3])  # [R, G, B] weights based on human eye sensitivity
aligned_display_tri, R, t = hybrid_alignment(display_tri, bt2020_tri, weights)

# Step 5: Generate random BT.2020 samples and map them to display gamut
n_samples = 300
x_s = np.random.uniform(0.13, 0.71, n_samples)
y_s = np.random.uniform(0.05, 0.79, n_samples)

# Convert BT.2020 samples to RGB (simplified approximation)
def xy_to_rgb(xy):
    x, y = xy[:, 0], xy[:, 1]
    z = 1 - x - y
    Y = 1.0  # Assume constant luminance
    X = (Y / y) * x
    Z = (Y / y) * z
    r = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    g = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    b = X * 0.0557 + Y * -0.2040 + Z * 1.0570
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb

bt2020_rgb = xy_to_rgb(np.column_stack([x_s, y_s]))

# Map BT.2020 RGB to display gamut using the inverse mixing matrix
mapped_rgb = np.zeros_like(bt2020_rgb)
for i in range(n_samples):
    for j in range(3):
        try:
            inv_matrix = np.linalg.inv(mixing_matrices[0, 0])  # Use average mixing matrix
            mapped_rgb[i] = inv_matrix @ bt2020_rgb[i]
        except np.linalg.LinAlgError:
            mapped_rgb[i] = bt2020_rgb[i]
mapped_rgb = np.clip(mapped_rgb, 0, 255).astype(np.uint8)

# Convert mapped RGB back to xy
def rgb_to_xy(rgb):
    rgb = rgb / 255.0
    X = rgb[:, 0] * 0.4124 + rgb[:, 1] * 0.3576 + rgb[:, 2] * 0.1805
    Y = rgb[:, 0] * 0.2126 + rgb[:, 1] * 0.7152 + rgb[:, 2] * 0.0722
    Z = rgb[:, 0] * 0.0193 + rgb[:, 1] * 0.1192 + rgb[:, 2] * 0.9505
    sum_XYZ = X + Y + Z
    x = X / sum_XYZ
    y = Y / sum_XYZ
    return np.column_stack([x, y])

x_d, y_d = rgb_to_xy(mapped_rgb).T

# Calculate chromaticity distance Δxy
delta_xy = np.sqrt((x_d - x_s)**2 + (y_d - y_s)**2)

# Output statistics
print("Δxy 色度压缩统计：")
print(f"均值 Δxy: {np.mean(delta_xy):.4f}")
print(f"最大 Δxy: {np.max(delta_xy):.4f}")
print(f"最小 Δxy: {np.min(delta_xy):.4f}")

# Step 6: Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Δxy distribution histogram
axs[0].hist(delta_xy, bins=30, color='darkcyan', edgecolor='black', alpha=0.8)
axs[0].axvline(np.mean(delta_xy), color='red', linestyle='--', label=f'Mean = {np.mean(delta_xy):.4f}')
axs[0].axvline(np.max(delta_xy) * 0.3, color='purple', linestyle='--', label=f'Threshold = {np.max(delta_xy)*0.3:.4f}')
axs[0].set_title('Distribution of Chromaticity Distance Δxy')
axs[0].set_xlabel('Δxy')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Right plot: Color gamut compression visualization
axs[1].scatter(x_s, y_s, color='lightblue', label='BT.2020 Source', alpha=0.5)
axs[1].scatter(x_d, y_d, color='orangered', label='Mapped to Display', alpha=0.6)

# Connect original and mapped points (every 10th point)
for i in range(0, n_samples, 10):
    axs[1].plot([x_s[i], x_d[i]], [y_s[i], y_d[i]], color='gray', alpha=0.3)

axs[1].set_xlim(0.1, 0.8)
axs[1].set_ylim(0.0, 0.9)
axs[1].set_xlabel('x (CIE1931)')
axs[1].set_ylabel('y (CIE1931)')
axs[1].set_title('Color Gamut Compression Visualization')
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Log the correction time
from datetime import datetime
correction_time = datetime.now().strftime("%Y-%m-%d %I:%M %p HKT")
print(f"Correction applied at: {correction_time}")
