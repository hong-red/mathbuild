import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =========================== 基础函数 ===========================
def apply_transform(points, theta, tx, ty):
    """
    应用旋转和平移变换到点集
    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    transformed = np.dot(points, R.T) + np.array([tx, ty])
    return transformed

def loss_function(params, source_points, target_points):
    theta, tx, ty = params
    transformed = apply_transform(source_points, theta, tx, ty)
    delta_xy = np.sqrt(np.sum((transformed - target_points) ** 2, axis=1))
    return np.sum(delta_xy)

def calculate_delta_xy(source, target):
    return np.sqrt(np.sum((source - target) ** 2, axis=1))

# =========================== 主优化函数 ===========================
def optimize_transform(source_points, target_points, trials=10, max_iter=100):
    best_result = None

    for _ in range(trials):
        initial_params = [
            np.random.uniform(-np.pi, np.pi),  # theta
            np.random.uniform(-0.1, 0.1),      # tx
            np.random.uniform(-0.1, 0.1)       # ty
        ]

        result = minimize(
            loss_function,
            initial_params,
            args=(source_points, target_points),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result

# =========================== 可视化函数 ===========================
def plot_gamuts(source, target, transformed):
    plt.figure(figsize=(6, 6))
    plt.plot(*np.append(source, [source[0]], axis=0).T, 'ro-', label='Source')
    plt.plot(*np.append(target, [target[0]], axis=0).T, 'go-', label='Target (BT.2020)')
    plt.plot(*np.append(transformed, [transformed[0]], axis=0).T, 'bo-', label='Optimized')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title("Gamut Mapping Optimization")
    plt.show()

def plot_delta_xy_distribution(source_points, target_points, transformed_points):
    delta_xy = calculate_delta_xy(transformed_points, target_points)
    plt.figure(figsize=(12, 5))

    # 直方图 - 进一步匹配第一个图的样式
    plt.subplot(1, 2, 1)
    plt.hist(delta_xy, bins=50, color='skyblue', edgecolor='black', range=(0, 0.3))  # 更细的bins
    mean_delta_xy = np.mean(delta_xy)
    threshold = 0.1979
    plt.axvline(mean_delta_xy, color='red', linestyle='--', label=f'Mean = {mean_delta_xy:.4f}')
    plt.axvline(threshold, color='yellow', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.title("Chromaticity Distance Δxy Distribution")
    plt.xlabel("Δxy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()

    # 散点图 - 改为色域映射样式，匹配第一个图
    plt.subplot(1, 2, 2)
    plt.scatter(source_points[:, 0], source_points[:, 1], color='cyan', label='BT2020 Source', alpha=0.6)
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color='pink', label='Mapped to Display', alpha=0.6)
    
    # 绘制Display Gamut边界（模拟）
    display_gamut = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
    plt.plot(*np.append(display_gamut, [display_gamut[0]], axis=0).T, 'r--', label='Display Gamut')
    
    # 绘制BT.2020 Gamut边界（模拟）
    bt2020_gamut = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]])
    plt.plot(*np.append(bt2020_gamut, [bt2020_gamut[0]], axis=0).T, 'b--', label='BT.2020 Gamut')
    
    plt.title("Color Gamut Compression Visualization")
    plt.xlabel("x (CIE 1931)")
    plt.ylabel("y (CIE 1931)")
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# =========================== 主程序入口 ===========================
if __name__ == "__main__":
    # 示例点 - 模拟更真实的色域点
    np.random.seed(42)
    # 模拟BT.2020色域内的点
    bt2020_gamut = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]])
    source_points = np.random.uniform(0, 1, (100, 2))
    # 确保点在BT.2020色域内（简化处理）
    source_points = source_points * 0.8  # 缩小范围以接近色域
    source_points[:, 0] = np.clip(source_points[:, 0], 0.131, 0.708)
    source_points[:, 1] = np.clip(source_points[:, 1], 0.046, 0.797)

    # 模拟目标点（映射后的点）
    target_points = source_points + np.random.normal(0, 0.02, source_points.shape)
    target_points[:, 0] = np.clip(target_points[:, 0], 0.15, 0.64)
    target_points[:, 1] = np.clip(target_points[:, 1], 0.06, 0.60)

    result = optimize_transform(source_points, target_points)
    theta_opt, tx_opt, ty_opt = result.x

    transformed_points = apply_transform(source_points, theta_opt, tx_opt, ty_opt)

    print("优化后 Δxy 指标：")
    delta_xy = calculate_delta_xy(transformed_points, target_points)
    print(f"  平均 Δxy: {np.mean(delta_xy):.4f}")
    print(f"  最大 Δxy: {np.max(delta_xy):.4f}")
    print(f"  最小 Δxy: {np.min(delta_xy):.4f}")

    plot_gamuts(source_points, target_points, transformed_points)
    plot_delta_xy_distribution(source_points, target_points, transformed_points)
