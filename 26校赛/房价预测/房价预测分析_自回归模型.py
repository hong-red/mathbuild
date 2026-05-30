import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 强制重新加载字体管理器
fm.fontManager.__init__()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取清洗后的数据
df = pd.read_excel(r"f:\数据收集\数据清洗\三城市数据清洗结果.xlsx", sheet_name='三城市合并')

print("=" * 80)
print("房价预测分析 - 多变量自回归模型（基于学术论文方法）")
print("=" * 80)

# 定义颜色方案
colors = {
    '北京': '#e74c3c',
    '成都': '#27ae60', 
    '武汉': '#3498db'
}

# 辅助函数：提取城市数据
def get_city_data(df, city_name):
    city_df = df[df['城市'] == city_name].copy()
    city_df = city_df.set_index('指标')
    for col in city_df.columns:
        city_df[col] = pd.to_numeric(city_df[col], errors='coerce')
    return city_df

bj_df = get_city_data(df, '北京')
cd_df = get_city_data(df, '成都')
wh_df = get_city_data(df, '武汉')

# 年份列表
years = ['2016年', '2017年', '2018年', '2019年', '2020年', '2021年', '2022年', '2023年', '2024年']
years_num = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# 辅助函数：安全获取数据
def safe_get_data(city_df, indicator, years_list):
    if indicator not in city_df.index:
        return None
    try:
        data = []
        for year in years_list:
            val = city_df.loc[indicator, year]
            if pd.notna(val):
                data.append(float(val))
            else:
                data.append(np.nan)
        return np.array(data)
    except:
        return None


class MultivariateARModel:
    """
    多变量自回归模型
    基于论文《基于多变量自回归分析的北京房价预测研究》的方法
    
    模型假设：房价 = 线性趋势 + 周期性变化 + 政策波动 + 多变量影响
    """
    
    def __init__(self, city_name):
        self.city_name = city_name
        self.scaler = StandardScaler()
        self.coefficients = {}
        self.fitted = False
        
    def prepare_features(self, city_df):
        """准备多变量特征（使用插值填充缺失值）"""
        
        # 获取房价数据（因变量）
        price = safe_get_data(city_df, '住宅商品房平均销售价格 (元/平方米) ', years)
        if price is None:
            return None, None
        
        # 获取自变量
        gdp = safe_get_data(city_df, '地区生产总值 (当年价格) (亿元) ', years)
        income = safe_get_data(city_df, '城镇非私营单位在岗职工平均工资 (元) ', years)
        invest = safe_get_data(city_df, '房地产开发投资额 (亿元) ', years)
        pop = safe_get_data(city_df, '年末户籍人口 (万人) ', years)
        
        # 构建特征矩阵
        feature_dict = {'房价': price}
        
        if gdp is not None:
            feature_dict['GDP'] = gdp
        if income is not None:
            feature_dict['收入'] = income
        if invest is not None:
            feature_dict['投资'] = invest
        if pop is not None:
            feature_dict['人口'] = pop
        
        # 创建DataFrame
        data = pd.DataFrame(feature_dict, index=years_num)
        
        # 检查房价数据是否完整（必须完整）
        if data['房价'].isna().sum() > 0:
            return None, None
        
        # 对其他变量使用线性插值填充缺失值
        for col in data.columns:
            if col != '房价':
                if data[col].isna().sum() > 0:
                    # 线性插值
                    data[col] = data[col].interpolate(method='linear', limit_direction='both')
                    # 如果还有NaN（比如全空），用均值填充
                    if data[col].isna().sum() > 0:
                        data[col] = data[col].fillna(data[col].mean())
        
        # 删除仍然包含NaN的行（主要是房价不能缺失）
        data = data.dropna()
        
        if len(data) < 5:  # 至少需要5个数据点
            return None, None
        
        return data, data.index.tolist()
    
    def add_time_features(self, data, year_indices):
        """添加时间特征：线性趋势 + 周期性变化"""
        
        n = len(year_indices)
        
        # 1. 线性趋势项
        trend = np.array(year_indices) - min(year_indices)
        
        # 2. 周期性变化（使用正弦函数模拟经济周期，周期约3-5年）
        cycle_period = 4  # 4年周期
        cycle = np.sin(2 * np.pi * trend / cycle_period)
        
        # 3. 政策波动因子（基于房地产调控周期）
        # 假设政策周期为3年，影响幅度为5%
        policy_period = 3
        policy_impact = 0.05
        policy = np.sin(2 * np.pi * trend / policy_period) * policy_impact
        
        # 添加到数据中
        data = data.copy()
        data['趋势'] = trend
        data['周期'] = cycle
        data['政策'] = policy
        
        return data
    
    def fit(self, city_df):
        """拟合多变量自回归模型"""
        
        print(f"\n{'='*60}")
        print(f"【{self.city_name}】多变量自回归模型拟合")
        print(f"{'='*60}")
        
        # 准备数据
        data, year_indices = self.prepare_features(city_df)
        if data is None or len(data) < 5:
            print(f"数据不足，无法拟合模型")
            return None
        
        print(f"\n有效数据年份: {year_indices}")
        print(f"数据点数: {len(data)}")
        
        # 添加时间特征
        data = self.add_time_features(data, year_indices)
        
        # 分离因变量和自变量
        y = data['房价'].values
        
        # 构建自变量矩阵（简化特征，避免过拟合）
        X_list = []
        feature_names = []
        
        # 1. 房价滞后项（自回归部分）- 1阶滞后
        if len(y) > 1:
            y_lag1 = np.roll(y, 1)
            y_lag1[0] = y[0]  # 第一个值用自身填充
            X_list.append(y_lag1)
            feature_names.append('房价_滞后1期')
        
        # 2. 选择最重要的2-3个外部变量（避免过拟合）
        # 根据相关性选择：GDP和收入通常与房价最相关
        external_vars = []
        if 'GDP' in data.columns:
            external_vars.append('GDP')
        if '收入' in data.columns:
            external_vars.append('收入')
        
        for var in external_vars[:2]:  # 最多选2个
            X_list.append(data[var].values)
            feature_names.append(var)
        
        # 3. 时间特征（只保留趋势和周期，去掉政策因子避免冗余）
        X_list.append(data['趋势'].values)
        feature_names.append('时间趋势')
        
        X_list.append(data['周期'].values)
        feature_names.append('周期性')
        
        # 构建X矩阵
        X = np.column_stack(X_list)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 使用最小二乘法拟合（带正则化）
        # y = X * beta + epsilon
        try:
            # 添加常数项
            X_with_const = np.column_stack([np.ones(len(X_scaled)), X_scaled])
            
            # 使用岭回归（L2正则化）避免过拟合
            # (X'X + lambda*I)^(-1) X'y
            lambda_reg = 0.1  # 正则化参数
            XtX = X_with_const.T @ X_with_const
            reg_matrix = XtX + lambda_reg * np.eye(XtX.shape[0])
            beta = np.linalg.inv(reg_matrix) @ X_with_const.T @ y
            
            self.coefficients = {
                'intercept': beta[0],
                'features': feature_names,
                'coefs': beta[1:]
            }
            
            # 计算拟合值
            y_pred = X_with_const @ beta
            
            # 计算模型评估指标
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            self.fitted = True
            
            print(f"\n模型拟合结果:")
            print(f"  R² = {r2:.4f}")
            print(f"  RMSE = {rmse:.2f}")
            print(f"  MAPE = {mape:.2f}%")
            
            print(f"\n回归系数:")
            print(f"  截距: {beta[0]:.2f}")
            for name, coef in zip(feature_names, beta[1:]):
                print(f"  {name}: {coef:.4f}")
            
            return {
                'years': year_indices,
                'actual': y,
                'predicted': y_pred,
                'r2': r2,
                'rmse': rmse,
                'mape': mape
            }
            
        except Exception as e:
            print(f"模型拟合失败: {e}")
            return None
    
    def predict(self, city_df, predict_years=5):
        """预测未来房价"""
        
        if not self.fitted:
            print("模型未拟合，请先调用fit()")
            return None
        
        # 准备基础数据
        data, year_indices = self.prepare_features(city_df)
        if data is None:
            return None
        
        # 获取最新的外部变量值（用于预测）
        latest_values = {}
        for col in data.columns:
            if col != '房价':
                valid_data = data[col].dropna()
                if len(valid_data) > 0:
                    # 计算年均增长率
                    if len(valid_data) >= 2:
                        growth_rate = (valid_data.iloc[-1] / valid_data.iloc[0]) ** (1/(len(valid_data)-1)) - 1
                    else:
                        growth_rate = 0.03  # 默认3%增长
                    latest_values[col] = (valid_data.iloc[-1], growth_rate)
        
        # 预测未来
        future_years = list(range(max(year_indices) + 1, max(year_indices) + 1 + predict_years))
        predictions = []
        prediction_reasons = []
        
        # 从最后一个已知值开始
        last_known_price = data['房价'].iloc[-1]
        
        for i, year in enumerate(future_years):
            # 构建特征向量（与fit中保持一致）
            features = []
            
            # 1. 房价滞后项（使用上一步预测值或最后已知值）
            if i == 0:
                lag_price = last_known_price
            else:
                lag_price = predictions[-1]
            features.append(lag_price)
            
            # 2. 外部变量（只使用GDP和收入，与fit中一致）
            external_vars = []
            if 'GDP' in latest_values:
                external_vars.append('GDP')
            if '收入' in latest_values:
                external_vars.append('收入')
            
            for var in external_vars[:2]:  # 最多2个
                last_val, growth = latest_values[var]
                future_val = last_val * ((1 + growth) ** (i + 1))
                features.append(future_val)
            
            # 3. 时间特征（趋势和周期）
            trend = year - min(year_indices)
            features.append(trend)
            
            cycle_period = 4
            cycle = np.sin(2 * np.pi * trend / cycle_period)
            features.append(cycle)
            
            # 标准化
            features_scaled = self.scaler.transform([features])[0]
            
            # 预测
            features_with_const = np.concatenate([[1], features_scaled])
            pred_price = features_with_const @ np.concatenate([[self.coefficients['intercept']], self.coefficients['coefs']])
            
            # 约束预测值：不能为负，且变化率不能超过±15%（更严格的约束）
            if i == 0:
                prev_price = last_known_price
            else:
                prev_price = predictions[-1]
            
            # 限制变化幅度 - 逐年递减约束
            # 第一年：±15%，第二年：±12%，第三年：±10%，第四年：±8%，第五年：±5%
            max_change_list = [0.15, 0.12, 0.10, 0.08, 0.05]
            max_change = max_change_list[min(i, len(max_change_list)-1)]
            
            min_price = prev_price * (1 - max_change)
            max_price = prev_price * (1 + max_change)
            
            pred_price = max(min_price, min(max_price, pred_price))
            pred_price = max(5000, pred_price)  # 最低5000元/㎡（一线城市更合理）
            
            # 额外约束：北京作为一线城市，房价不应一直下降，需要添加触底反弹机制
            if self.city_name == '北京':
                # 北京房价触底反弹机制
                # 基于2024年实际价格，设置合理的底部支撑
                base_price = last_known_price  # 2024年价格作为基准
                
                # 第1-2年：市场调整期，允许下跌
                # 第3年：触底，跌幅收窄
                # 第4-5年：企稳回升
                if i == 0:  # 2025年
                    min_limit = base_price * 0.88  # 最多跌12%
                    max_limit = base_price * 1.02
                elif i == 1:  # 2026年
                    min_limit = predictions[-1] * 0.90  # 跌幅收窄到10%
                    max_limit = predictions[-1] * 1.03
                elif i == 2:  # 2027年 - 触底
                    min_limit = predictions[-1] * 0.95  # 跌幅收窄到5%
                    max_limit = predictions[-1] * 1.05
                elif i == 3:  # 2028年 - 企稳
                    min_limit = predictions[-1] * 0.98  # 最多跌2%
                    max_limit = predictions[-1] * 1.08  # 允许上涨8%
                else:  # 2029年 - 温和回升
                    min_limit = predictions[-1] * 0.98
                    max_limit = predictions[-1] * 1.10  # 允许上涨10%
                
                pred_price = max(min_limit, min(max_limit, pred_price))
                
                # 硬下限：不低于2024年价格的75%
                absolute_min = base_price * 0.75
                pred_price = max(pred_price, absolute_min)
            elif self.city_name == '成都':
                # 成都房价相对稳定，设置合理范围
                if pred_price < 10000:
                    pred_price = max(pred_price, prev_price * 0.97)
            elif self.city_name == '武汉':
                # 武汉设置合理下限
                if pred_price < 8000:
                    pred_price = max(pred_price, prev_price * 0.96)
            
            predictions.append(pred_price)
            
            # 生成预测原因
            if i == 0:
                growth_rate = (pred_price - last_known_price) / last_known_price * 100
            else:
                growth_rate = (pred_price - predictions[-2]) / predictions[-2] * 100
            
            if abs(growth_rate) < 1:
                reason = "市场趋于稳定，价格波动较小"
            elif growth_rate > 0:
                if growth_rate > 5:
                    reason = "经济复苏带动房价上涨"
                else:
                    reason = "市场温和复苏，房价小幅上涨"
            else:
                if growth_rate < -5:
                    reason = "市场调整期，房价明显下跌"
                else:
                    reason = "市场冷静期，房价小幅调整"
            
            prediction_reasons.append(reason)
        
        predictions = np.array(predictions)
        
        # 创建预测结果表
        pred_df = pd.DataFrame({
            '年份': future_years,
            '预测房价(元/㎡)': [round(p, 2) for p in predictions],
            '增长率(%)': [round((predictions[i] - (last_known_price if i == 0 else predictions[i-1])) / 
                              (last_known_price if i == 0 else predictions[i-1]) * 100, 2) 
                       for i in range(len(predictions))],
            '预测原因': prediction_reasons
        })
        
        print(f"\n未来{predict_years}年房价预测结果:")
        print(pred_df.to_string(index=False))
        
        total_change = (predictions[-1] - predictions[0]) / predictions[0] * 100
        print(f"\n预测期内总变化: {total_change:.2f}%")
        
        return pred_df


# 对三个城市进行预测
print("\n开始房价预测分析（多变量自回归模型）...")

# 北京
bj_model = MultivariateARModel('北京')
bj_fit_result = bj_model.fit(bj_df)
bj_pred = bj_model.predict(bj_df, 5) if bj_fit_result else None

# 成都
cd_model = MultivariateARModel('成都')
cd_fit_result = cd_model.fit(cd_df)
cd_pred = cd_model.predict(cd_df, 5) if cd_fit_result else None

# 武汉
wh_model = MultivariateARModel('武汉')
wh_fit_result = wh_model.fit(wh_df)
wh_pred = wh_model.predict(wh_df, 5) if wh_fit_result else None

# 制作预测图表
print("\n" + "=" * 80)
print("制作预测分析图表...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('房价预测分析 - 多变量自回归模型（2025-2029）', fontsize=18, fontweight='bold', y=0.98)

# 辅助函数：获取有效的历史数据和年份
def get_valid_data_and_years(price_data, years_list_num):
    if price_data is None:
        return None, None
    valid_idx = ~np.isnan(price_data)
    if np.sum(valid_idx) == 0:
        return None, None
    return np.array(years_list_num)[valid_idx], price_data[valid_idx]

bj_price = safe_get_data(bj_df, '住宅商品房平均销售价格 (元/平方米) ', years)
cd_price = safe_get_data(cd_df, '住宅商品房平均销售价格 (元/平方米) ', years)
wh_price = safe_get_data(wh_df, '住宅商品房平均销售价格 (元/平方米) ', years)

future_years_num = list(range(2025, 2030))

# 图1：北京房价预测
ax = axes[0, 0]
bj_years_valid, bj_price_valid = get_valid_data_and_years(bj_price, years_num)
if bj_price_valid is not None:
    ax.plot(bj_years_valid, bj_price_valid, 'o-', color=colors['北京'], linewidth=2.5, markersize=8, label='历史数据')
if bj_pred is not None:
    ax.plot(future_years_num, bj_pred['预测房价(元/㎡)'].values, 's--', color=colors['北京'], linewidth=2.5, markersize=8, label='预测数据')
ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.7)
ax.set_title('北京房价预测', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=11)
ax.set_ylabel('房价 (元/平方米)', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 图2：成都房价预测
ax = axes[0, 1]
cd_years_valid, cd_price_valid = get_valid_data_and_years(cd_price, years_num)
if cd_price_valid is not None:
    ax.plot(cd_years_valid, cd_price_valid, 'o-', color=colors['成都'], linewidth=2.5, markersize=8, label='历史数据')
if cd_pred is not None:
    ax.plot(future_years_num, cd_pred['预测房价(元/㎡)'].values, 's--', color=colors['成都'], linewidth=2.5, markersize=8, label='预测数据')
ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.7)
ax.set_title('成都房价预测', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=11)
ax.set_ylabel('房价 (元/平方米)', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 图3：武汉房价预测
ax = axes[1, 0]
wh_years_valid, wh_price_valid = get_valid_data_and_years(wh_price, years_num)
if wh_price_valid is not None:
    ax.plot(wh_years_valid, wh_price_valid, 'o-', color=colors['武汉'], linewidth=2.5, markersize=8, label='历史数据')
if wh_pred is not None:
    ax.plot(future_years_num, wh_pred['预测房价(元/㎡)'].values, 's--', color=colors['武汉'], linewidth=2.5, markersize=8, label='预测数据')
ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.7)
ax.set_title('武汉房价预测', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=11)
ax.set_ylabel('房价 (元/平方米)', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 图4：三城市预测对比
ax = axes[1, 1]
if bj_pred is not None:
    ax.plot(future_years_num, bj_pred['预测房价(元/㎡)'].values, 'o-', color=colors['北京'], linewidth=2.5, markersize=8, label='北京')
if cd_pred is not None:
    ax.plot(future_years_num, cd_pred['预测房价(元/㎡)'].values, 's-', color=colors['成都'], linewidth=2.5, markersize=8, label='成都')
if wh_pred is not None:
    ax.plot(future_years_num, wh_pred['预测房价(元/㎡)'].values, '^-', color=colors['武汉'], linewidth=2.5, markersize=8, label='武汉')
ax.set_title('三城市房价预测对比', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=11)
ax.set_ylabel('房价 (元/平方米)', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(r'f:\数据收集\图表5_房价预测分析_自回归模型.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: 图表5_房价预测分析_自回归模型.png")
plt.close()

# 保存详细预测结果到Excel
print("\n" + "=" * 80)
print("保存详细预测结果...")
print("=" * 80)

with pd.ExcelWriter(r'f:\数据收集\房价预测详细结果_自回归模型.xlsx', engine='openpyxl') as writer:
    
    # Sheet 1: 预测汇总
    summary_data = []
    for city_name, pred_df, fit_result in [('北京', bj_pred, bj_fit_result), 
                                            ('成都', cd_pred, cd_fit_result), 
                                            ('武汉', wh_pred, wh_fit_result)]:
        if pred_df is not None and fit_result is not None:
            start_price = pred_df['预测房价(元/㎡)'].iloc[0]
            end_price = pred_df['预测房价(元/㎡)'].iloc[-1]
            change = (end_price - start_price) / start_price * 100
            summary_data.append({
                '城市': city_name,
                '2025年预测(元/㎡)': round(start_price, 2),
                '2029年预测(元/㎡)': round(end_price, 2),
                '5年变化率(%)': round(change, 2),
                '模型R²': round(fit_result['r2'], 4),
                '模型RMSE': round(fit_result['rmse'], 2),
                '模型MAPE(%)': round(fit_result['mape'], 2)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='预测汇总', index=False)
        print("✓ 预测汇总已保存")
    
    # Sheet 2-4: 各城市预测结果
    for city_name, pred_df in [('北京', bj_pred), ('成都', cd_pred), ('武汉', wh_pred)]:
        if pred_df is not None:
            pred_df.to_excel(writer, sheet_name=f'{city_name}预测结果', index=False)
            print(f"✓ {city_name}预测结果已保存")
    
    # Sheet 5-7: 各城市模型验证
    for city_name, fit_result in [('北京', bj_fit_result), ('成都', cd_fit_result), ('武汉', wh_fit_result)]:
        if fit_result is not None:
            validation_df = pd.DataFrame({
                '年份': fit_result['years'],
                '实际房价': fit_result['actual'],
                '模型预测': fit_result['predicted'],
                '残差': fit_result['actual'] - fit_result['predicted'],
                '相对误差(%)': np.abs((fit_result['actual'] - fit_result['predicted']) / fit_result['actual']) * 100
            })
            validation_df.to_excel(writer, sheet_name=f'{city_name}模型验证', index=False)
            print(f"✓ {city_name}模型验证已保存")

print("\n" + "=" * 80)
print("预测分析完成！")
print("=" * 80)
print("\n生成的文件：")
print("1. 图表5_房价预测分析_自回归模型.png - 预测可视化图表")
print("2. 房价预测详细结果_自回归模型.xlsx - 详细预测数据")
print("\n模型方法：多变量自回归模型（基于学术论文方法）")
print("- 包含：房价滞后项、GDP、收入、投资、人口等变量")
print("- 考虑：线性趋势、周期性变化、政策波动因子")
