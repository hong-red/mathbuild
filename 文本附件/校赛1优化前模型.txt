import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# 第一步：定义混合对齐函数
def centroid_align(source, target):
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    return source - source_centroid + target_centroid

def weighted_kabsch(P, Q, weights=None):
    if weights is None:
        weights = np.ones(len(P))
    w_sum = np.sum(weights)
    centroid_P = np.sum(P * weights[:, None], axis=0) / w_sum
    centroid_Q = np.sum(Q * weights[:, None], axis=0) / w_sum
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = (P_centered.T * weights) @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t

def hybrid_alignment(source_tri, target_tri, weights=None):
    rough_aligned = centroid_align(source_tri, target_tri)
    R, t = weighted_kabsch(rough_aligned, target_tri, weights)
    final_aligned = (rough_aligned @ R.T) + t
    return final_aligned, R, t

# 第二步：迭代优化
def apply_transform(points, theta, tx, ty):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    t = np.array([tx, ty])
    return (points @ R.T) + t

def loss_function(params, source_points, target_points):
    theta, tx, ty = params
    transformed = apply_transform(source_points, theta, tx, ty)
    delta_xy = np.sqrt(np.sum((transformed - target_points)**2, axis=1))
    return np.sum(delta_xy)

def iterative_refinement(source_points, target_points, initial_R, initial_t, max_iter=100):
    theta = np.arctan2(initial_R[1, 0], initial_R[0, 0])
    initial_params = [theta, initial_t[0], initial_t[1]]
    result = minimize(
        loss_function,
        initial_params,
        args=(source_points, target_points),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )
    theta_opt, tx_opt, ty_opt = result.x
    R_opt = np.array([[np.cos(theta_opt), -np.sin(theta_opt)],
                      [np.sin(theta_opt), np.cos(theta_opt)]])
    t_opt = np.array([tx_opt, ty_opt])
    return R_opt, t_opt

# 第三步：基于三角形插值的色域映射
def barycentric_coordinates(p, tri):
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = p - tri[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
    w2 = (d11 * d20 - d01 * d21) / denom
    w3 = (d00 * d21 - d01 * d20) / denom
    w1 = 1.0 - w2 - w3
    w = np.clip([w1, w2, w3], 0, None)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.array([1/3, 1/3, 1/3])
    return w

def gamut_map_triangle(source_points, source_tri, target_tri):
    mapped_points = np.zeros_like(source_points)
    for i, p in enumerate(source_points):
        weights = barycentric_coordinates(p, source_tri)
        mapped_points[i] = weights[0] * target_tri[0] + weights[1] * target_tri[1] + weights[2] * target_tri[2]
    return mapped_points

# 第四步：模拟 64x64 网格
np.random.seed(42)
height, width = 64, 64
channels = 3

data = {}
sheets = ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']
for sheet in sheets:
    if sheet in ['R_R', 'G_G', 'B_B']:
        data[sheet] = np.random.uniform(200, 239, size=(height, width))
    else:
        data[sheet] = np.random.uniform(5, 20, size=(height, width))

red_measured = np.stack([data['R_R'], data['R_G'], data['R_B']], axis=-1)
green_measured = np.stack([data['G_R'], data['G_G'], data['G_B']], axis=-1)
blue_measured = np.stack([data['B_R'], data['B_G'], data['B_B']], axis=-1)

target_red = np.array([220, 0, 0])
target_green = np.array([0, 220, 0])
target_blue = np.array([0, 0, 220])
target_white = np.array([220, 220, 220])

mixing_matrices = np.zeros((height, width, 3, 3))
for i in range(height):
    for j in range(width):
        measured_matrix = np.column_stack([
            red_measured[i, j] / 220,
            green_measured[i, j] / 220,
            blue_measured[i, j] / 220
        ])
        mixing_matrices[i, j] = measured_matrix

corrected_inputs = np.zeros((height, width, 3, 4))
for i in range(height):
    for j in range(width):
        try:
            inv_matrix = np.linalg.inv(mixing_matrices[i, j])
            corrected_inputs[i, j, :, 0] = inv_matrix @ target_red
            corrected_inputs[i, j, :, 1] = inv_matrix @ target_green
            corrected_inputs[i, j, :, 2] = inv_matrix @ target_blue
            corrected_inputs[i, j, :, 3] = inv_matrix @ target_white
        except np.linalg.LinAlgError:
            corrected_inputs[i, j, 0, 0] = 220 * (220 / (data['R_R'][i, j] + 1e-6))
            corrected_inputs[i, j, 1, 1] = 220 * (220 / (data['G_G'][i, j] + 1e-6))
            corrected_inputs[i, j, 2, 2] = 220 * (220 / (data['B_B'][i, j] + 1e-6))
            corrected_inputs[i, j, :, 3] = 220

corrected_inputs = np.clip(corrected_inputs, 0, 255)

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

corrected_red_output = np.clip(corrected_red_output, 0, 255).astype(np.uint8)
corrected_green_output = np.clip(corrected_green_output, 0, 255).astype(np.uint8)
corrected_blue_output = np.clip(corrected_blue_output, 0, 255).astype(np.uint8)
corrected_white_output = np.clip(corrected_white_output, 0, 255).astype(np.uint8)

# 第五步：提取色域三角形
display_red_xy = np.array([np.mean(data['R_R'] / (data['R_R'] + data['R_G'] + data['R_B'])),
                          np.mean(data['R_G'] / (data['R_R'] + data['R_G'] + data['R_B']))])
display_green_xy = np.array([np.mean(data['G_R'] / (data['G_R'] + data['G_G'] + data['G_B'])),
                            np.mean(data['G_G'] / (data['G_R'] + data['G_G'] + data['G_B']))])
display_blue_xy = np.array([np.mean(data['B_R'] / (data['B_R'] + data['B_G'] + data['B_B'])),
                           np.mean(data['B_G'] / (data['B_R'] + data['B_G'] + data['B_B']))])

display_tri = np.array([display_red_xy, display_green_xy, display_blue_xy])

bt2020_tri = np.array([[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]])

# 第六步：将显示色域对齐到 BT.2020
weights = np.array([0.4, 0.5, 0.1])
aligned_display_tri, R, t = hybrid_alignment(display_tri, bt2020_tri, weights)

# 第七步：生成随机 BT.2020 样本
n_samples = 300
x_s = np.random.uniform(0.15, 0.68, n_samples)
y_s = np.random.uniform(0.06, 0.69, n_samples)

def is_inside_bt2020(x, y):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign([x, y], bt2020_tri[0], bt2020_tri[1])
    d2 = sign([x, y], bt2020_tri[1], bt2020_tri[2])
    d3 = sign([x, y], bt2020_tri[2], bt2020_tri[0])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

mask = np.array([is_inside_bt2020(x, y) for x, y in zip(x_s, y_s)])
x_s = x_s[mask][:n_samples//2]
y_s = y_s[mask][:n_samples//2]
n_samples = len(x_s)

aligned_xy_s = np.zeros((n_samples, 2))
for i in range(n_samples):
    aligned_xy_s[i] = (np.array([x_s[i], y_s[i]]) @ R.T) + t

# 第八步：应用基于三角形插值的色域映射
source_points = aligned_xy_s
mapped_points = gamut_map_triangle(source_points, bt2020_tri, display_tri)
x_d, y_d = mapped_points[:, 0], mapped_points[:, 1]

# 第九步：迭代优化
R_opt, t_opt = iterative_refinement(source_points, mapped_points, R, t, max_iter=100)

# 应用优化后的变换
final_aligned_xy_s = apply_transform(aligned_xy_s, np.arctan2(R_opt[1, 0], R_opt[0, 0]), t_opt[0], t_opt[1])

# 计算最终色度距离 Δxy
delta_xy = np.sqrt((final_aligned_xy_s[:, 0] - x_d)**2 + (final_aligned_xy_s[:, 1] - y_d)**2)

# 输出统计信息
print("Δxy 色度压缩统计：")
print(f"均值 Δxy: {np.mean(delta_xy):.4f}")
print(f"最大 Δxy: {np.max(delta_xy):.4f}")
print(f"最小 Δxy: {np.min(delta_xy):.4f}")

# 第十步：可视化
plt.style.use('seaborn-v0_8-darkgrid')

fig, axs = plt.subplots(1, 2, figsize=(16, 6), facecolor='#f5f5f5')

# 左图：Δxy 分布直方图
hist_color = '#1f77b4'
axs[0].hist(delta_xy, bins=30, color=hist_color, edgecolor='white', alpha=0.8, linewidth=1.2)
axs[0].axvline(np.mean(delta_xy), color='#ff4d4d', linestyle='-', linewidth=2.5, label=f'Mean = {np.mean(delta_xy):.4f}', alpha=0.9)
axs[0].axvline(np.max(delta_xy) * 0.66, color='#ffcc00', linestyle='--', linewidth=2.5, label=f'Threshold = {np.max(delta_xy)*0.66:.4f}', alpha=0.9)
axs[0].set_title('Chromaticity Distance Δxy Distribution', fontsize=14, fontweight='bold', color='#333333')
axs[0].set_xlabel('Δxy', fontsize=12, color='#333333')
axs[0].set_ylabel('Frequency', fontsize=12, color='#333333')
axs[0].legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', framealpha=0.9)
axs[0].tick_params(axis='both', labelsize=10, colors='#555555')
axs[0].grid(True, linestyle='--', alpha=0.6, color='#999999')

# 右图：色域压缩可视化
axs[1].scatter(x_s, y_s, color='#00ccff', label='BT.2020 Source', alpha=0.6, s=60, edgecolors='white', linewidth=0.5)
axs[1].scatter(x_d, y_d, color='#ff3366', label='Mapped to Display', alpha=0.6, s=60, edgecolors='white', linewidth=0.5)
for i in range(0, n_samples, 10):
    axs[1].plot([x_s[i], x_d[i]], [y_s[i], y_d[i]], color='#666666', linestyle='-', alpha=0.4, linewidth=1.2, zorder=1)
bt2020_tri_closed = np.vstack([bt2020_tri, bt2020_tri[0]])
display_tri_closed = np.vstack([display_tri, display_tri[0]])
axs[1].plot(bt2020_tri_closed[:, 0], bt2020_tri_closed[:, 1], color='#00ccff', linestyle='--', linewidth=2, label='BT.2020 Gamut', alpha=0.7)
axs[1].plot(display_tri_closed[:, 0], display_tri_closed[:, 1], color='#ff3366', linestyle='--', linewidth=2, label='Display Gamut', alpha=0.7)
axs[1].set_xlim(0.1, 0.8)
axs[1].set_ylim(0.0, 0.9)
axs[1].set_xlabel('x (CIE 1931)', fontsize=12, color='#333333')
axs[1].set_ylabel('y (CIE 1931)', fontsize=12, color='#333333')
axs[1].set_title('Color Gamut Compression Visualization', fontsize=14, fontweight='bold', color='#333333')
axs[1].legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', framealpha=0.9, loc='upper right')
axs[1].tick_params(axis='both', labelsize=10, colors='#555555')
axs[1].grid(True, linestyle='--', alpha=0.6, color='#999999')

plt.tight_layout()
plt.show()

from datetime import datetime
correction_time = datetime.now().strftime("%Y-%m-%d %I:%M %p HKT")
print(f"Correction applied at: {correction_time}")
