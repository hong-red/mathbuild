import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def hybrid_alignment(source_tri, target_tri, weights=None):
    """混合对齐流程"""
    # 质心对齐
    source_centroid = np.mean(source_tri, axis=0)
    target_centroid = np.mean(target_tri, axis=0)
    rough_aligned = source_tri - source_centroid + target_centroid
    
    # 加权Kabsch算法
    if weights is None:
        weights = np.ones(len(source_tri))
    w_sum = np.sum(weights)
    P_centered = rough_aligned - np.sum(rough_aligned * weights[:, None], axis=0) / w_sum
    Q_centered = target_tri - np.sum(target_tri * weights[:, None], axis=0) / w_sum
    
    # SVD分解和旋转矩阵计算
    H = (P_centered.T * weights) @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算最终变换
    t = target_centroid - R @ source_centroid
    final_aligned = (rough_aligned @ R.T) + t
    return final_aligned, R, t

# 初始化数据
np.random.seed(42)
height, width = 64, 64
data = {sheet: np.random.uniform(200 if sheet in ['R_R', 'G_G', 'B_B'] else 5,
                               239 if sheet in ['R_R', 'G_G', 'B_B'] else 20,
                               size=(height, width))
        for sheet in ['R_R', 'R_G', 'R_B', 'G_R', 'G_G', 'G_B', 'B_R', 'B_G', 'B_B']}

# 构建RGB向量和混合矩阵
rgb_measured = {color: np.stack([data[f'{color}_{ch}'] for ch in 'RGB'], axis=-1)
                for color in 'RGB'}
mixing_matrices = np.array([rgb_measured[color] / 220 for color in 'RGB']).transpose(1, 2, 3, 0)

# 计算校正输入和输出
targets = {'R': [220, 0, 0], 'G': [0, 220, 0], 'B': [0, 0, 220], 'W': [220, 220, 220]}
corrected = {}
for color, target in targets.items():
    corrected[color] = np.zeros_like(rgb_measured['R'])
    for i in range(height):
        for j in range(width):
            try:
                inv_matrix = np.linalg.inv(mixing_matrices[i, j])
                corrected[color][i, j] = np.clip(inv_matrix @ target, 0, 255)
            except np.linalg.LinAlgError:
                corrected[color][i, j] = [220 if k == l else 0 for k, l in enumerate('RGB')]
                if color == 'W':
                    corrected[color][i, j] = [220, 220, 220]

# 计算色域三角形
get_xy = lambda data, ch: np.mean(data[f'{ch}_R'] / np.sum([data[f'{ch}_{c}'] for c in 'RGB'], axis=0))
display_tri = np.array([[get_xy(data, ch), get_xy(data, ch)] for ch in 'RGB'])
bt2020_tri = np.array([[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]])

# 色域映射
weights = np.array([0.3, 0.4, 0.3])
aligned_display_tri, R, t = hybrid_alignment(display_tri, bt2020_tri, weights)

# 生成和映射样本
n_samples = 300
x_s, y_s = np.random.uniform(0.13, 0.71, n_samples), np.random.uniform(0.05, 0.79, n_samples)

def xy_to_rgb(xy):
    x, y = xy[:, 0], xy[:, 1]
    z = 1 - x - y
    Y = np.ones_like(x)
    X, Z = (Y / y) * x, (Y / y) * z
    rgb = np.clip(np.stack([
        X * 3.2406 + Y * -1.5372 + Z * -0.4986,
        X * -0.9689 + Y * 1.8758 + Z * 0.0415,
        X * 0.0557 + Y * -0.2040 + Z * 1.0570
    ], axis=-1) * 255, 0, 255).astype(np.uint8)
    return rgb

bt2020_rgb = xy_to_rgb(np.column_stack([x_s, y_s]))
mapped_rgb = np.zeros_like(bt2020_rgb)
inv_matrix = np.linalg.inv(mixing_matrices[0, 0])
for i in range(n_samples):
    try:
        mapped_rgb[i] = np.clip(inv_matrix @ bt2020_rgb[i], 0, 255)
    except np.linalg.LinAlgError:
        mapped_rgb[i] = bt2020_rgb[i]

# 计算色度距离
def rgb_to_xy(rgb):
    rgb = rgb / 255.0
    X = rgb @ [0.4124, 0.3576, 0.1805]
    Y = rgb @ [0.2126, 0.7152, 0.0722]
    Z = rgb @ [0.0193, 0.1192, 0.9505]
    sum_XYZ = X + Y + Z
    return np.column_stack([X / sum_XYZ, Y / sum_XYZ])

x_d, y_d = rgb_to_xy(mapped_rgb).T
delta_xy = np.sqrt((x_d - x_s)**2 + (y_d - y_s)**2)

# 可视化
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].hist(delta_xy, bins=30, color='darkcyan', edgecolor='black', alpha=0.8)
axs[0].axvline(np.mean(delta_xy), color='red', linestyle='--', label=f'均值 = {np.mean(delta_xy):.4f}')
axs[0].axvline(np.max(delta_xy) * 0.3, color='purple', linestyle='--', label=f'阈值 = {np.max(delta_xy)*0.3:.4f}')
axs[0].set_title('色度距离Δxy分布')
axs[0].set_xlabel('色度距离Δxy')
axs[0].set_ylabel('频数')
axs[0].legend()

axs[1].scatter(x_s, y_s, color='lightblue', label='BT.2020源色域', alpha=0.5)
axs[1].scatter(x_d, y_d, color='orangered', label='显示器映射色域', alpha=0.6)
for i in range(0, n_samples, 10):
    axs[1].plot([x_s[i], x_d[i]], [y_s[i], y_d[i]], color='gray', alpha=0.3)
axs[1].set_xlim(0.1, 0.8)
axs[1].set_ylim(0.0, 0.9)
axs[1].set_xlabel('色度坐标x (CIE1931)')
axs[1].set_ylabel('色度坐标y (CIE1931)')
axs[1].set_title('色域压缩可视化')
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.show()

print(f"Δxy 色度压缩统计：\n均值: {np.mean(delta_xy):.4f}\n最大值: {np.max(delta_xy):.4f}\n最小值: {np.min(delta_xy):.4f}")
