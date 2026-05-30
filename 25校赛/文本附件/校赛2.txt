import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # Windows系统支持中文的字体
plt.rcParams['axes.unicode_minus'] = False

# 1. 改进的模拟数据生成
def generate_color_data(n_samples=1000):
    np.random.seed(42)
    RGB = np.random.rand(n_samples, 3)
    # V通道与RGB相关
    V = 0.3*RGB[:,0] + 0.5*RGB[:,1] + 0.2*RGB[:,2] + 0.1*np.random.randn(n_samples)
    V = np.clip(V, 0, 1).reshape(-1,1)
    RGBV = np.hstack([RGB, V])
    
    # 更复杂的非线性变换
    RGBCX = np.zeros((n_samples, 5))
    RGBCX[:,0] = 0.9*RGBV[:,0] + 0.1*RGBV[:,3] - 0.2*RGBV[:,1]*RGBV[:,2]  # R'
    RGBCX[:,1] = 0.85*RGBV[:,1] + 0.15*RGBV[:,3] + 0.1*RGBV[:,0]*RGBV[:,2] # G'
    RGBCX[:,2] = 0.88*RGBV[:,2] + 0.12*RGBV[:,3] - 0.05*RGBV[:,0]*RGBV[:,1] # B'
    RGBCX[:,3] = 0.5*(RGBV[:,1] + RGBV[:,2]) + 0.2*RGBV[:,0]*RGBV[:,3]     # C'
    RGBCX[:,4] = 0.6*(RGBV[:,0] + RGBV[:,3]) - 0.1*RGBV[:,1]*RGBV[:,2]     # X'
    
    return np.clip(RGBV, 0, 1), np.clip(RGBCX, 0, 1)

RGBV, RGBCX = generate_color_data()

# 2. 改进的色度空间转换
def rgb_to_xyb(rgb_data):
    """RGB到xyY转换"""
    rgb = rgb_data[:,:3]
    v = rgb_data[:,3]
    
    # sRGB到XYZ转换矩阵
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    XYZ = np.dot(rgb, M.T)
    sum_xyz = np.sum(XYZ, axis=1, keepdims=True) + 1e-10  # 避免除以零
    
    # 计算色度坐标
    xy = XYZ[:,:2] / sum_xyz
    Y = XYZ[:,1]  # 亮度
    
    # 亮度加权
    weighted_xy = xy * (v[:,np.newaxis]**0.4)  # 非线性亮度加权
    
    return weighted_xy, Y, v

xy_rgbv, Y_rgbv, v_rgbv = rgb_to_xyb(RGBV)
xy_rgbcx, Y_rgbcx, v_rgbcx = rgb_to_xyb(RGBCX[:,:4])

# 3. 改进的加权Kabsch算法
def weighted_kabsch(P, Q, weights):
    weights = np.clip(weights, 0.2, 1.0)  # 限制权重范围
    
    # 计算加权质心
    centroid_P = np.sum(P * weights[:,np.newaxis], axis=0) / np.sum(weights)
    centroid_Q = np.sum(Q * weights[:,np.newaxis], axis=0) / np.sum(weights)
    
    # 中心化
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 计算协方差矩阵
    H = (P_centered * weights[:,np.newaxis]).T @ Q_centered
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 确保是右手系
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    # 计算缩放因子
    scale = np.sqrt(np.sum((Q_centered * weights[:,np.newaxis])**2) / 
                   np.sum((P_centered * weights[:,np.newaxis])**2))
    
    return R, scale, centroid_P, centroid_Q

R, scale, centroid_P, centroid_Q = weighted_kabsch(xy_rgbv, xy_rgbcx, v_rgbv)

# 应用变换
def apply_transform(xy, R, scale, centroid_P, centroid_Q):
    xy_aligned = (xy - centroid_P) @ (scale * R) + centroid_Q
    return xy_aligned

xy_aligned = apply_transform(xy_rgbv, R, scale, centroid_P, centroid_Q)

# 4. 改进的模型架构
class ColorMapper:
    def __init__(self):
        # 各通道独立模型
        self.channel_models = []
        for _ in range(5):
            pipeline = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=2, include_bias=False),
                Lasso(alpha=0.001, max_iter=5000, random_state=42)
            )
            self.channel_models.append(pipeline)
        
        # 全局微调模型
        self.final_model = MLPRegressor(
            hidden_layer_sizes=(64,32),
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    
    def fit(self, X, y):
        # 训练各通道模型
        for i, model in enumerate(self.channel_models):
            model.fit(X, y[:,i])
        
        # 生成初步预测
        initial_pred = np.column_stack([model.predict(X) for model in self.channel_models])
        
        # 全局微调
        self.final_model.fit(initial_pred, y)
    
    def predict(self, X):
        initial_pred = np.column_stack([model.predict(X) for model in self.channel_models])
        return np.clip(self.final_model.predict(initial_pred), 0, 1)

# 训练模型
model = ColorMapper()
model.fit(RGBV, RGBCX)
predicted = model.predict(RGBV)

# 5. 评估指标
def evaluate_model(y_true, y_pred):
    delta = np.sqrt(np.sum((y_pred - y_true)**2, axis=1))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"平均 Δxy: {np.mean(delta):.4f}")
    print(f"最大 Δxy: {np.max(delta):.4f}")
    print(f"最小 Δxy: {np.min(delta):.4f}")
    print(f"RMSE: {rmse:.4f}")

print("模型性能指标:")
evaluate_model(RGBCX, predicted)

# 6. 测试用例
test_colors = {
    "红": [1.0, 0.0, 0.0, 0.5],
    "绿": [0.0, 1.0, 0.0, 0.5],
    "蓝": [0.0, 0.0, 1.0, 0.5],
    "黄": [1.0, 1.0, 0.0, 0.8],
    "青": [0.0, 1.0, 1.0, 0.7],
    "品红": [1.0, 0.0, 1.0, 0.6],
    "白": [1.0, 1.0, 1.0, 1.0],
    "黑": [0.0, 0.0, 0.0, 0.1]
}

print("\n测试颜色映射结果:")
for name, color in test_colors.items():
    input_rgbv = np.array(color).reshape(1,-1)
    output_rgbcx = model.predict(input_rgbv)[0]
    print(f"\n{name}色:")
    print(f"输入 RGBV: {input_rgbv[0]}")
    print(f"输出 RGBCX: {output_rgbcx.round(4)}")

# 7. 可视化（分开图）
# 通道预测精度
plt.figure(figsize=(6, 5))
for i in range(5):
    plt.scatter(RGBCX[:,i], predicted[:,i], alpha=0.3, label=f'通道{i+1}')
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('各通道预测精度')
plt.legend()
plt.grid(True)
plt.show()

# 颜色映射对比
color_names = ["红", "绿", "蓝"]
for i, (name, color) in enumerate(test_colors.items()):
    if name not in color_names:
        continue
    plt.figure(figsize=(9, 3))
    
    input_rgbv = np.array(color).reshape(1,-1)
    output_rgbcx = model.predict(input_rgbv)[0]
    
    plt.subplot(1, 3, 1)
    plt.imshow([[input_rgbv[0,:3]]], aspect='auto')
    plt.xticks([]); plt.yticks([])
    plt.title(f"{name}输入")
    
    plt.subplot(1, 3, 2)
    plt.imshow([[input_rgbv[0,:3]]], aspect='auto')  # 原始映射
    plt.xticks([]); plt.yticks([])
    plt.title("线性映射")
    
    plt.subplot(1, 3, 3)
    plt.imshow([[output_rgbcx[:3]]], aspect='auto')  # 优化映射
    plt.xticks([]); plt.yticks([])
    plt.title("优化映射")
    
    plt.suptitle(f"{name}色映射对比", fontsize=14)
    plt.tight_layout()
    plt.show()

# 色度图
plt.figure(figsize=(6, 5))
# sRGB和AdobeRGB色域边界
sRGB_coords = np.array([[0.64,0.33], [0.30,0.60], [0.15,0.06], [0.64,0.33]])
adobeRGB_coords = np.array([[0.64,0.33], [0.21,0.71], [0.15,0.06], [0.64,0.33]])

plt.plot(sRGB_coords[:,0], sRGB_coords[:,1], 'r-', label='sRGB色域')
plt.plot(adobeRGB_coords[:,0], adobeRGB_coords[:,1], 'g-', label='AdobeRGB色域')
plt.scatter(xy_rgbv[:,0], xy_rgbv[:,1], c='blue', alpha=0.3, label='输入')
plt.scatter(xy_rgbcx[:,0], xy_rgbcx[:,1], c='red', alpha=0.3, label='目标')
plt.scatter(xy_aligned[:,0], xy_aligned[:,1], c='green', alpha=0.3, label='对齐后')
plt.xlabel('x'); plt.ylabel('y')
plt.title('色度空间分布')
plt.legend()
plt.grid(True)
plt.show()
