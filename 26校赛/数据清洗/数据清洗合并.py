import pandas as pd
import numpy as np

def clean_city_data(file_path, city_name):
    """
    清洗单个城市数据：
    1. 删除全为空的行和列
    2. 正确处理表头
    3. 保留有效数据
    """
    print(f"\n{'='*60}")
    print(f"正在清洗 {city_name} 数据...")
    print(f"{'='*60}")
    
    # 读取原始数据（不使用默认表头）
    df = pd.read_excel(file_path, header=None)
    
    print(f"原始数据形状: {df.shape}")
    
    # 删除全为NaN的列
    df = df.dropna(axis=1, how='all')
    print(f"删除空列后形状: {df.shape}")
    
    # 删除全为NaN的行
    df = df.dropna(axis=0, how='all')
    print(f"删除空行后形状: {df.shape}")
    
    # 找到实际的表头行（第2行，索引为2，内容是"指标 2025年 2024年..."）
    header_row_idx = 2
    
    # 提取列名：第一列是"指标"，后面是年份
    columns = ['指标'] + [f'{2025-i}年' for i in range(df.shape[1]-1)]
    
    # 提取数据部分（从第3行开始，索引3及以后）
    data_df = df.iloc[header_row_idx+1:].copy()
    data_df.columns = columns
    data_df = data_df.reset_index(drop=True)
    
    # 删除指标为空的行
    data_df = data_df.dropna(subset=['指标'])
    
    # 删除指标列全为空的行（指标列有值但其他列全空）
    data_df = data_df[data_df.iloc[:, 1:].notna().any(axis=1)]
    
    # 重置索引
    data_df = data_df.reset_index(drop=True)
    
    # 添加城市标识列
    data_df.insert(0, '城市', city_name)
    
    print(f"清洗后数据形状: {data_df.shape}")
    print(f"\n前10行数据预览:")
    print(data_df.head(10).to_string())
    
    return data_df

# 清洗三个城市的数据
print("开始数据清洗...")

beijing_clean = clean_city_data(r"f:\数据收集\主要城市年度数据_北京.xlsx", "北京")
chengdu_clean = clean_city_data(r"f:\数据收集\主要城市年度数据 _成都.xlsx", "成都")
wuhan_clean = clean_city_data(r"f:\数据收集\主要城市年度数据 _武汉.xlsx", "武汉")

# 合并三个城市的数据
print(f"\n{'='*60}")
print("合并三个城市数据...")
print(f"{'='*60}")

# 确保三个数据框的列一致
all_columns = list(beijing_clean.columns)
print(f"列名: {all_columns}")

# 合并数据
merged_df = pd.concat([beijing_clean, chengdu_clean, wuhan_clean], ignore_index=True)

print(f"合并后总数据形状: {merged_df.shape}")
print(f"\n各城市数据量:")
print(merged_df['城市'].value_counts())

# 保存到Excel（三个sheet + 合并sheet）
output_file = r"f:\数据收集\三城市数据清洗结果.xlsx"
print(f"\n{'='*60}")
print(f"保存结果到: {output_file}")
print(f"{'='*60}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet1: 北京数据
    beijing_clean.to_excel(writer, sheet_name='北京', index=False)
    print("✓ Sheet1: 北京数据已保存")
    
    # Sheet2: 成都数据
    chengdu_clean.to_excel(writer, sheet_name='成都', index=False)
    print("✓ Sheet2: 成都数据已保存")
    
    # Sheet3: 武汉数据
    wuhan_clean.to_excel(writer, sheet_name='武汉', index=False)
    print("✓ Sheet3: 武汉数据已保存")
    
    # Sheet4: 合并数据
    merged_df.to_excel(writer, sheet_name='三城市合并', index=False)
    print("✓ Sheet4: 三城市合并数据已保存")

print(f"\n{'='*60}")
print("数据清洗完成！")
print(f"{'='*60}")
print(f"输出文件: {output_file}")
print(f"包含4个工作表：北京、成都、武汉、三城市合并")
print(f"每个城市有效数据: {len(beijing_clean)} 行")
