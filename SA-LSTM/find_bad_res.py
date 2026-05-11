import pandas as pd
import os

# ===================== 配置区域 =====================
CSV_PATH = "/data1/xyh/projects/nnUNet/SA-LSTM/airway_landmark_601_results_LCC.csv" 
OUTPUT_EXCEL_PATH = "all_poor_results.xlsx"

# 设定过滤标准 (单位: mm)
CLINICAL_REJECT_MM = 4.0  # 超过这个值就算“结果不好”            临床不合格
SEVERE_FAIL_MM = 6.0      # 超过这个值算“严重偏离”              超过设定的半径阈值，说明 pred 和 gt 没有交集

LANDMARKS = ['M5', 'M5L', 'M5R', 'M7', 'M7R', 'M7L', 'ANS', 'CM', 'ZR', 'PNS', 'ZL', 'N']

def categorize_error(val):
    """根据定义好的标准，对误差进行定性分级"""
    if pd.isna(val) or val == '---':
        return 'Valid (No GT expected)'
    if val == 'Miss':
        return 'Level 1: Miss (漏检)'
    if val == 'FP':
        return 'Level 1: FP (误检)'
    
    try:
        num_val = float(val)
        if num_val > SEVERE_FAIL_MM:
            return f'Level 2: 严重偏离 (> {SEVERE_FAIL_MM}mm)'
        elif num_val > CLINICAL_REJECT_MM:
            return f'Level 3: 临床不合格 (> {CLINICAL_REJECT_MM}mm)'
        else:
            return 'Pass (及格)'
    except:
        return 'Unknown'

def main():
    print(f"📥 读取数据: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到文件: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 强制清除表头可能存在的隐藏空格，防止意外报错
    df.columns = df.columns.str.strip()
    
    # 1. 将宽表“融化”成长表，把所有病例的所有点拉平，逐一审查
    # 【修改点】：将 id_vars 修改为真实表头 'Patient_Name'
    melted_df = df.melt(id_vars=['Patient_Name'], value_vars=LANDMARKS, 
                        var_name='Landmark_Name', value_name='Error_Value')
    
    # 2. 为每一个点打上“严重程度”标签
    melted_df['Severity_Level'] = melted_df['Error_Value'].apply(categorize_error)
    
    # 3. 过滤出所有“不合格”的点 (排除 Pass 和 Valid 的点)
    all_bad_points = melted_df[~melted_df['Severity_Level'].isin(['Pass (及格)', 'Valid (No GT expected)'])]
    
    # 按照严重程度和误差值排序，把最惨的排在最前面
    all_bad_points['Numeric_Error'] = pd.to_numeric(all_bad_points['Error_Value'], errors='coerce').fillna(999)
    all_bad_points = all_bad_points.sort_values(by=['Severity_Level', 'Numeric_Error'], ascending=[True, False])
    all_bad_points = all_bad_points.drop(columns=['Numeric_Error'])

    # 4. 汇总“不良病例”档案
    # 【修改点】：按照 'Patient_Name' 进行 groupby
    bad_cases_summary = all_bad_points.groupby('Patient_Name').agg(
        Total_Bad_Points=('Landmark_Name', 'count'),
        Bad_Point_Details=('Landmark_Name', lambda x: ', '.join(x)),
        Severity_Details=('Severity_Level', lambda x: ' | '.join(x))
    ).reset_index().sort_values(by='Total_Bad_Points', ascending=False)

    # 5. 保存到 Excel
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH) as writer:
        # Sheet 1: 所有出问题的点位清单
        all_bad_points.to_excel(writer, sheet_name="所有不良点位清单", index=False)
        
        # Sheet 2: 不良病例黑名单
        bad_cases_summary.to_excel(writer, sheet_name="不良病例黑名单", index=False)
        
        # Sheet 3: 宏观统计
        point_stats = all_bad_points.groupby(['Landmark_Name', 'Severity_Level']).size().unstack(fill_value=0).reset_index()
        point_stats.to_excel(writer, sheet_name="各关键点失败统计", index=False)

    print(f"\n✅ 筛查完成！结果已保存至: {OUTPUT_EXCEL_PATH}")
    print(f"📊 发现的不良点位总数: {len(all_bad_points)}")
    print(f"📋 涉及的不良病例总数: {len(bad_cases_summary)}")
    # =================================================================
    # 6. 提取最差的 10 个样本，方便直接复制去可视化
    # =================================================================
    print("\n" + "="*70)
    print("🚨 建议优先可视化的【最差 10 个样本】名单")
    print("="*70)

    # 视角 A：按“不良点位数量”排名的 Top 10
    top10_by_count = bad_cases_summary.head(10)
    print("\n🔻 视角 1：按【不良点位数量 (含漏检/误检/大误差)】排名的 Top 10:")
    for i, row in top10_by_count.reset_index().iterrows():
        print(f"  {i+1:>2d}. {row['Patient_Name']:<15} | 坏点数量: {row['Total_Bad_Points']} 个")
        # 如果你想在终端看具体是哪些点坏了，可以取消下面这行的注释
        # print(f"      -> 涉及点位: {row['Bad_Point_Details']}")

    # 视角 B：按“纯数字平均误差”排名的 Top 10 (剔除掉 Miss/FP 的非数字项后计算平均值)
    numeric_df = df.copy()
    for col in LANDMARKS:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    # 计算每个病人的平均误差
    numeric_df['Patient_Mean_Error'] = numeric_df[LANDMARKS].mean(axis=1)
    
    top10_by_error = numeric_df.sort_values(by='Patient_Mean_Error', ascending=False).head(10)
    print("\n🔻 视角 2：按【平均空间定位误差】排名的 Top 10 (仅统计找出来的点):")
    for i, row in top10_by_error.reset_index().iterrows():
        print(f"  {i+1:>2d}. {row['Patient_Name']:<15} | 平均误差: {row['Patient_Mean_Error']:.2f} mm")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()