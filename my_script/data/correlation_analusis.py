import os
import pandas as pd
from scipy.stats import pearsonr

# ================= 配置区域 =================
# 1. 输入文件路径 (确保这三个文件都在脚本同级目录下)
LANDMARK_CSV = "/data1/xyh/projects/nnUNet/my_script/results/csv/airway_landmark_511_results_LCC.csv"   # 定点结果
SEGMENTATION_CSV = "/data1/xyh/projects/nnUNet/my_script/results/csv/airway_segmentation_510_results_LCC.csv"    # 分割结果
RESOLUTION_CSV = "/data1/xyh/projects/nnUNet/my_script/data/image_resolutions_info_511.csv"    # 分辨率与Size信息

# 2. 输出文件路径
OUTPUT_MERGED_CSV = os.path.join("/data1/xyh/projects/nnUNet/my_script/results/csv", "multimodal_analysis_merged.csv")
OUTPUT_REPORT_TXT = os.path.join("/data1/xyh/projects/nnUNet/my_script/results/csv", "full_spatial_correlation_report.txt")

# 3. 定点标签列表 (用于计算平均误差)
LABELS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================

def calculate_pearson(df, col1, col2):
    """计算两个变量的皮尔逊相关系数及 P 值，自动剔除空值"""
    valid_data = df[[col1, col2]].dropna()
    if len(valid_data) < 2:
        return float('nan'), float('nan')
    r, p_value = pearsonr(valid_data[col1], valid_data[col2])
    return r, p_value

def main():
    print("🚀 开始进行全维度空间属性与模型性能的相关性分析...\n")

    # ================= 1. 数据读取与清洗 =================
    if not os.path.exists(RESOLUTION_CSV):
        print(f"❌ 找不到文件: {RESOLUTION_CSV}") 
        return
        
    df_res = pd.read_csv(RESOLUTION_CSV)

    # 读取定点数据
    df_lm = pd.read_csv(LANDMARK_CSV)
    for label in LABELS:
        df_lm[label] = pd.to_numeric(df_lm[label], errors='coerce')
    df_lm['Landmark_Avg_Error'] = df_lm[LABELS].mean(axis=1)

    # 读取分割数据
    df_seg = pd.read_csv(SEGMENTATION_CSV)
    if 'Case Name' in df_seg.columns:
        df_seg['Patient_Name'] = df_seg['Case Name'].str.replace('.nii.gz', '', regex=False)
        df_seg = df_seg.drop(columns=['Case Name'])

    # ================= 2. 数据融合 =================
    df_merged = pd.merge(df_res, df_lm, on='Patient_Name', how='outer')
    df_merged = pd.merge(df_merged, df_seg, on='Patient_Name', how='outer')
    df_merged.to_csv(OUTPUT_MERGED_CSV, index=False, float_format='%.4f', encoding='utf-8-sig')

    # ================= 3. 统计分析：皮尔逊相关性 =================
    # 全维度的自变量 (Spacing = 物理间距, Size = 像素个数)
    independent_vars = [
        'Spacing_X', 'Spacing_Y', 'Spacing_Z', 
        'Size_X', 'Size_Y', 'Size_Z'
    ] 
    
    # 你指定的两个核心目标变量
    dependent_vars = [
        ('Landmark_Avg_Error', '定点平均误差 (mm)', 'positive'), 
        ('Dice', '分割 Dice 相似系数', 'negative')
    ]

    report_lines = []
    report_lines.append("=====================================================================")
    report_lines.append("          【全维度】图像空间属性对 定点与分割 性能的影响报告")
    report_lines.append("=====================================================================")
    report_lines.append("指标说明:")
    report_lines.append(" - Spacing: 像素的物理间距 (代表图像清晰度/层厚)。")
    report_lines.append(" - Size: 图像的像素矩阵大小 (代表视野大小/FOV)。")
    report_lines.append(" - r (相关系数): -1 到 1 之间。绝对值 > 0.4 为中度相关，> 0.6 为强相关。")
    report_lines.append(" - p (显著性): < 0.05 代表统计学显著(***)，不是巧合。")
    report_lines.append("=====================================================================\n")

    for indep_var in independent_vars:
        report_lines.append(f"🔍 探究自变量: 【{indep_var}】")
        report_lines.append("-" * 69)
        
        for dep_col, dep_name, expected_trend in dependent_vars:
            if dep_col not in df_merged.columns:
                continue
                
            r, p = calculate_pearson(df_merged, indep_var, dep_col)
            
            # 解释相关性强度
            strength = "极弱或无"
            if abs(r) >= 0.6: strength = "强相关"
            elif abs(r) >= 0.4: strength = "中度相关"
            elif abs(r) >= 0.2: strength = "弱相关"

            # 解释物理意义
            if p >= 0.05:
                conclusion = "无统计学显著影响"
            elif r > 0 and expected_trend == 'positive':
                conclusion = f"此数值越大，{dep_name} 越大 (表现变差)"
            elif r < 0 and expected_trend == 'negative':
                conclusion = f"此数值越大，{dep_name} 越小 (表现变差)"
            elif r < 0 and expected_trend == 'positive':
                conclusion = f"此数值越大，{dep_name} 越小 (表现变好)"
            elif r > 0 and expected_trend == 'negative':
                conclusion = f"此数值越大，{dep_name} 越大 (表现变好)"
            else:
                conclusion = "趋势复杂"

            # 判断显著性
            sig = "显著! ***" if p < 0.05 else "不显著 ns"

            report_lines.append(f"  🎯 受影响指标: {dep_name:<18}")
            report_lines.append(f"     * r 值: {r:>8.4f}  ({strength})")
            report_lines.append(f"     * p 值: {p:>8.4e}  [{sig}]")
            report_lines.append(f"     * 结论: {conclusion}")
            report_lines.append("")
        report_lines.append("=====================================================================\n")

    report_text = "\n".join(report_lines)
    print(report_text)

    # 保存报告
    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ 全维度相关性分析报告已保存至: {OUTPUT_REPORT_TXT}")

if __name__ == "__main__":
    main()