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

# =========================== 示例数据与可视化 ===========================
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

# =========================== 主程序入口 ===========================
if __name__ == "__main__":
    # 示例点（模拟）
    source_points = np.array([[0.68, 0.32], [0.265, 0.690], [0.150, 0.060]])
    target_points = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]])

    result = optimize_transform(source_points, target_points)
    theta_opt, tx_opt, ty_opt = result.x

    transformed_points = apply_transform(source_points, theta_opt, tx_opt, ty_opt)
    delta_xy = calculate_delta_xy(transformed_points, target_points)

    print("优化后 Δxy 指标：")
    print(f"  平均 Δxy: {np.mean(delta_xy):.4f}")
    print(f"  最大 Δxy: {np.max(delta_xy):.4f}")
    print(f"  最小 Δxy: {np.min(delta_xy):.4f}")

    plot_gamuts(source_points, target_points, transformed_points)
