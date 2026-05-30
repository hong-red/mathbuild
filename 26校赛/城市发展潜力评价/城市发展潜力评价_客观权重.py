import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 强制重新加载字体管理器
fm.fontManager.__init__()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("第三问：城市发展潜力与稳定性评价 - 基于客观影响占比")
print("方法：相关系数权重 + TOPSIS综合评价模型")
print("=" * 80)

# 读取清洗后的数据
df = pd.read_excel(r"f:\数据收集\数据清洗\三城市数据清洗结果.xlsx", sheet_name='三城市合并')

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

# ==================== 基于相关系数计算客观权重 ====================
print("\n" + "=" * 80)
print("第一步：计算各因素与房价的相关系数")
print("=" * 80)

cities = ['北京', '成都', '武汉']
city_dfs = [bj_df, cd_df, wh_df]

# 定义各因素包含的指标
factor_indicators = {
    '房地产因素': [
        '房地产开发投资额 (亿元) ',
        '商品房销售面积 (万平方米) ',
        '房地产开发企业土地购置面积 (万平方米) '
    ],
    '经济因素': [
        '地区生产总值 (当年价格) (亿元) ',
        '城镇非私营单位在岗职工平均工资 (元) ',
        '地方财政一般预算收入 (亿元) '
    ],
    '人口社会因素': [
        '年末户籍人口 (万人) ',
        '社会消费品零售总额 (亿元) ',
        '城乡居民储蓄年末余额 (亿元) '
    ],
    '政策与周期因素': [
        '地区生产总值 (当年价格) (亿元) ',  # 用GDP波动代表政策经济周期
        '住宅商品房平均销售价格 (元/平方米) '  # 房价自身的周期性
    ]
}

# 计算各城市各因素与房价的相关系数
city_factor_corr = {}

for city_name, city_df in zip(cities, city_dfs):
    city_factor_corr[city_name] = {}
    price = safe_get_data(city_df, '住宅商品房平均销售价格 (元/平方米) ', years)
    
    if price is None:
        continue
    
    for factor_name, indicators in factor_indicators.items():
        correlations = []
        for indicator in indicators:
            data = safe_get_data(city_df, indicator, years)
            if data is not None:
                valid_mask = ~np.isnan(price) & ~np.isnan(data)
                if np.sum(valid_mask) > 2:
                    corr = np.corrcoef(price[valid_mask], data[valid_mask])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        # 计算该因素的平均相关系数
        if correlations:
            city_factor_corr[city_name][factor_name] = np.mean(correlations)
        else:
            city_factor_corr[city_name][factor_name] = 0

# 打印各城市相关系数
print("\n各城市各因素与房价的相关系数（绝对值）：")
for city in cities:
    print(f"\n{city}:")
    for factor, corr in sorted(city_factor_corr[city].items(), key=lambda x: x[1], reverse=True):
        print(f"  {factor}: {corr:.4f}")

# 计算三城市平均相关系数
print("\n" + "=" * 80)
print("第二步：计算三城市平均相关系数并归一化为权重")
print("=" * 80)

all_factors = list(factor_indicators.keys())
avg_correlations = {}

for factor in all_factors:
    corrs = [city_factor_corr[city].get(factor, 0) for city in cities]
    avg_correlations[factor] = np.mean(corrs)

print("\n三城市平均相关系数：")
for factor, corr in sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True):
    print(f"  {factor}: {corr:.4f}")

# 归一化得到权重
total_corr = sum(avg_correlations.values())
factor_weights = {factor: corr / total_corr for factor, corr in avg_correlations.items()}

print("\n客观权重分配（基于相关系数）：")
for factor, weight in sorted(factor_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {factor}: {weight*100:.2f}%")

# ==================== 构建评价指标体系 ====================
print("\n" + "=" * 80)
print("第三步：构建评价指标体系并计算得分")
print("=" * 80)

# 计算评价指标数据
def calculate_growth_rate(data):
    """计算年均增长率"""
    if data is None or len(data) < 2:
        return None
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < 2:
        return None
    start_val = valid_data[0]
    end_val = valid_data[-1]
    if start_val <= 0:
        return None
    years_count = len(valid_data) - 1
    growth_rate = (end_val / start_val) ** (1/years_count) - 1
    return growth_rate * 100

def calculate_volatility(data):
    """计算波动率（标准差/均值）"""
    if data is None or len(data) < 2:
        return None
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < 2 or np.mean(valid_data) == 0:
        return None
    volatility = np.std(valid_data) / np.mean(valid_data) * 100
    return volatility

def calculate_price_income_ratio(city_df):
    """计算房价收入比"""
    price = safe_get_data(city_df, '住宅商品房平均销售价格 (元/平方米) ', years)
    income = safe_get_data(city_df, '城镇非私营单位在岗职工平均工资 (元) ', years)
    if price is None or income is None:
        return None
    housing_area = 90
    ratio = price * housing_area / income
    valid_ratio = ratio[~np.isnan(ratio)]
    if len(valid_ratio) == 0:
        return None
    return np.mean(valid_ratio)

# 计算各因素得分
factor_scores = {city: {} for city in cities}

for city_name, city_df in zip(cities, city_dfs):
    # 房地产因素得分（房价增长、投资增长、销售增长的平均值）
    price_growth = calculate_growth_rate(safe_get_data(city_df, '住宅商品房平均销售价格 (元/平方米) ', years))
    inv_growth = calculate_growth_rate(safe_get_data(city_df, '房地产开发投资额 (亿元) ', years))
    sales_growth = calculate_growth_rate(safe_get_data(city_df, '商品房销售面积 (万平方米) ', years))
    
    real_estate_scores = [s for s in [price_growth, inv_growth, sales_growth] if s is not None]
    factor_scores[city_name]['房地产因素'] = np.mean(real_estate_scores) if real_estate_scores else 0
    
    # 经济因素得分（GDP增长、收入增长）
    gdp_growth = calculate_growth_rate(safe_get_data(city_df, '地区生产总值 (当年价格) (亿元) ', years))
    income_growth = calculate_growth_rate(safe_get_data(city_df, '城镇非私营单位在岗职工平均工资 (元) ', years))
    
    economy_scores = [s for s in [gdp_growth, income_growth] if s is not None]
    factor_scores[city_name]['经济因素'] = np.mean(economy_scores) if economy_scores else 0
    
    # 人口社会因素得分（人口增长、消费增长、房价收入比友好度）
    pop_growth = calculate_growth_rate(safe_get_data(city_df, '年末户籍人口 (万人) ', years))
    cons_growth = calculate_growth_rate(safe_get_data(city_df, '社会消费品零售总额 (亿元) ', years))
    price_income = calculate_price_income_ratio(city_df)
    
    # 房价收入比转换为得分（越低越好，10左右为最佳）
    if price_income is not None:
        price_income_score = max(0, 100 - abs(price_income - 10) * 5)
    else:
        price_income_score = 50  # 默认值
    
    society_scores = [s for s in [pop_growth, cons_growth, price_income_score] if s is not None]
    factor_scores[city_name]['人口社会因素'] = np.mean(society_scores) if society_scores else 0
    
    # 政策与周期因素得分（市场稳定性）
    price_volatility = calculate_volatility(safe_get_data(city_df, '住宅商品房平均销售价格 (元/平方米) ', years))
    gdp_volatility = calculate_volatility(safe_get_data(city_df, '地区生产总值 (当年价格) (亿元) ', years))
    
    # 波动率转换为稳定性得分（越低越稳定）
    stability_scores_list = []
    if price_volatility is not None:
        stability_scores_list.append(max(0, 100 - price_volatility))
    if gdp_volatility is not None:
        stability_scores_list.append(max(0, 100 - gdp_volatility))
    
    factor_scores[city_name]['政策与周期因素'] = np.mean(stability_scores_list) if stability_scores_list else 50

# 打印各因素得分
print("\n各因素原始得分：")
for city in cities:
    print(f"\n{city}:")
    for factor, score in factor_scores[city].items():
        print(f"  {factor}: {score:.2f}")

# ==================== TOPSIS综合评价 ====================
print("\n" + "=" * 80)
print("第四步：TOPSIS综合评价")
print("=" * 80)

def topsis_evaluation(data_df, weights_dict):
    """TOPSIS综合评价"""
    # 数据标准化（向量归一化）
    normalized_df = data_df.copy()
    for col in data_df.columns:
        col_data = data_df[col].values
        norm = np.sqrt(np.sum(col_data ** 2))
        if norm > 0:
            normalized_df[col] = col_data / norm
        else:
            normalized_df[col] = 0
    
    # 构建加权标准化矩阵
    weights_array = np.array([weights_dict[col] for col in data_df.columns])
    weighted_df = normalized_df * weights_array
    
    # 确定正理想解和负理想解
    positive_ideal = weighted_df.max().values
    negative_ideal = weighted_df.min().values
    
    # 计算到正理想解和负理想解的距离
    d_positive = np.sqrt(np.sum((weighted_df.values - positive_ideal) ** 2, axis=1))
    d_negative = np.sqrt(np.sum((weighted_df.values - negative_ideal) ** 2, axis=1))
    
    # 计算综合评价指数
    scores = d_negative / (d_positive + d_negative)
    
    return scores

# 构建评价矩阵
factor_df = pd.DataFrame(factor_scores).T
print("\n评价矩阵：")
print(factor_df.to_string())

# 计算TOPSIS得分
topsis_scores = topsis_evaluation(factor_df, factor_weights)

print("\n各城市TOPSIS得分：")
for i, city in enumerate(cities):
    print(f"  {city}: {topsis_scores[i]:.4f}")

# 排名
ranking = np.argsort(-topsis_scores)
print("\n综合排名：")
for rank, idx in enumerate(ranking, 1):
    print(f"  第{rank}名: {cities[idx]} (得分: {topsis_scores[idx]:.4f})")

# ==================== 生成评价结果图表 ====================
print("\n" + "=" * 80)
print("第五步：生成评价结果图表")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('城市发展潜力与稳定性评价 - 基于客观相关系数权重', fontsize=18, fontweight='bold', y=0.98)

# 图1：客观权重饼图
ax = axes[0, 0]
weights_values = [factor_weights[f] * 100 for f in all_factors]
colors_pie = ['#e74c3c', '#3498db', '#27ae60', '#f39c12']
wedges, texts, autotexts = ax.pie(weights_values, labels=all_factors, autopct='%1.1f%%', 
                                   colors=colors_pie, startangle=90, textprops={'fontsize': 10})
ax.set_title('基于相关系数的客观权重', fontsize=14, fontweight='bold')

# 图2：各因素得分对比
ax = axes[0, 1]
x = np.arange(len(cities))
width = 0.2
for i, factor in enumerate(all_factors):
    scores = [factor_scores[city][factor] for city in cities]
    ax.bar(x + i*width, scores, width, label=factor, alpha=0.8, color=colors_pie[i])

ax.set_title('各因素原始得分对比', fontsize=14, fontweight='bold')
ax.set_ylabel('得分', fontsize=11)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(cities)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 图3：综合得分
ax = axes[1, 0]
bars = ax.bar(cities, topsis_scores, color=[colors[c] for c in cities], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title('综合得分排名（TOPSIS）', fontsize=14, fontweight='bold')
ax.set_ylabel('得分', fontsize=11)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, topsis_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 图4：雷达图
ax = axes[1, 1]
angles = np.linspace(0, 2 * np.pi, len(all_factors), endpoint=False).tolist()
angles += angles[:1]

ax = plt.subplot(2, 2, 4, projection='polar')
for city in cities:
    values = [factor_scores[city][factor] for factor in all_factors]
    # 归一化到0-1
    max_val = max(values) if max(values) > 0 else 1
    min_val = min(values)
    if max_val > min_val:
        values = [(v - min_val) / (max_val - min_val) for v in values]
    else:
        values = [0.5] * len(values)
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=city, color=colors[city])
    ax.fill(angles, values, alpha=0.15, color=colors[city])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([f.replace('因素', '') for f in all_factors], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('四大因素雷达图', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(r'f:\数据收集\城市发展潜力评价\图表8_城市发展潜力评价_客观权重_v2.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: 图表8_城市发展潜力评价_客观权重_v2.png")
plt.close()

# ==================== 保存评价结果到Excel ====================
print("\n" + "=" * 80)
print("第六步：保存评价结果")
print("=" * 80)

with pd.ExcelWriter(r'f:\数据收集\城市发展潜力评价\城市发展潜力评价结果_客观权重_v2.xlsx', engine='openpyxl') as writer:
    
    # Sheet 1: 各城市相关系数
    corr_df = pd.DataFrame(city_factor_corr).T
    corr_df.to_excel(writer, sheet_name='各城市相关系数')
    print("✓ 各城市相关系数已保存")
    
    # Sheet 2: 客观权重
    weight_data = pd.DataFrame({
        '因素名称': all_factors,
        '平均相关系数': [avg_correlations[f] for f in all_factors],
        '客观权重': [factor_weights[f] for f in all_factors],
        '权重百分比': [f"{factor_weights[f]*100:.2f}%" for f in all_factors]
    })
    weight_data = weight_data.sort_values('客观权重', ascending=False)
    weight_data.to_excel(writer, sheet_name='客观权重', index=False)
    print("✓ 客观权重已保存")
    
    # Sheet 3: 各因素得分
    factor_df.to_excel(writer, sheet_name='各因素得分')
    print("✓ 各因素得分已保存")
    
    # Sheet 4: 综合评价结果
    result_data = pd.DataFrame({
        '城市': cities,
        '房地产因素得分': [factor_scores[c]['房地产因素'] for c in cities],
        '经济因素得分': [factor_scores[c]['经济因素'] for c in cities],
        '人口社会因素得分': [factor_scores[c]['人口社会因素'] for c in cities],
        '政策与周期因素得分': [factor_scores[c]['政策与周期因素'] for c in cities],
        'TOPSIS综合得分': topsis_scores,
        '排名': [list(ranking).index(i) + 1 for i in range(len(cities))]
    })
    result_data = result_data.sort_values('排名')
    result_data.to_excel(writer, sheet_name='综合评价结果', index=False)
    print("✓ 综合评价结果已保存")

print("\n" + "=" * 80)
print("第三问完成！基于客观相关系数的综合评价")
print("=" * 80)
print("\n生成的文件：")
print("1. 图表8_城市发展潜力评价_客观权重_v2.png - 评价结果可视化")
print("2. 城市发展潜力评价结果_客观权重_v2.xlsx - 详细评价数据")
print("\n客观权重分配（基于相关系数）：")
for factor, weight in sorted(factor_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {factor}: {weight*100:.2f}%")
print("\n评价结果：")
for rank, idx in enumerate(ranking, 1):
    print(f"  第{rank}名: {cities[idx]} (得分: {topsis_scores[idx]:.4f})")
