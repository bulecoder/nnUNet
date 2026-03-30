import pandas as pd
import numpy as np
import os
"""
    根据gt和validation的最小横截面积，计算平均误差
"""

TASK_ID = 510

# ================= 配置区域 =================
# 指向你刚才用 GT-nii 转出的 STL 计算得到的 CSV (列名应包含 Case Name, GT_mCSA_mm2)
STL_CSV_PATH = f'/data1/xyh/projects/nnUNet/my_script/results/csv/5fold_mcsa_gt_standardized_results_{TASK_ID}.csv'
# 指向你刚才用 Pred-nii 转出的 STL 计算得到的 CSV (列名应包含 Fold, Case Name, Pred_mCSA_mm2)
VAL_CSV_PATH = f'/data1/xyh/projects/nnUNet/my_script/results/csv/5fold_mcsa_pred_standardized_results_{TASK_ID}.csv'
# 最终生成的对比报告路径
OUTPUT_FILE = f'/data1/xyh/projects/nnUNet/my_script/results/csv/5fold_mcsa_final_evaluation_report_{TASK_ID}.csv'
# ===========================================

def main():
    if not os.path.exists(STL_CSV_PATH) or not os.path.exists(VAL_CSV_PATH):
        print(f"❌ 错误: 找不到输入文件。")
        return

    try:
        # 1. 读取数据
        df_gt = pd.read_csv(STL_CSV_PATH)
        df_pred = pd.read_csv(VAL_CSV_PATH)

        # 2. 清洗数据：去除 Case Name 的前后空格
        df_gt['Case Name'] = df_gt['Case Name'].astype(str).str.strip()
        df_pred['Case Name'] = df_pred['Case Name'].astype(str).str.strip()
        
        # 3. 合并数据 (以 Case Name 为主键)
        df_merged = pd.merge(df_pred, df_gt, on='Case Name', how='inner')

        if df_merged.empty:
            print("⚠️ 警告: 合并后的数据为空，请检查两个 CSV 中的 Case Name 是否真的匹配。")
            return

        # 4. 计算误差指标
        df_merged['Abs_Error_mm2'] = (df_merged['Pred_mCSA_mm2'] - df_merged['GT_mCSA_mm2']).abs()
        df_merged['Relative_Error_Pct'] = (df_merged['Abs_Error_mm2'] / df_merged['GT_mCSA_mm2']) * 100

        # 5. 排序
        df_merged = df_merged.sort_values(by=['Fold', 'Case Name'])

        # 6. 构建统计平均行 (修复 Length mismatch 的关键)
        # 我们先创建一个全空的行，然后只填充数值列
        avg_series = pd.Series(index=df_merged.columns, dtype=object)
        avg_series['Fold'] = 'OVERALL'
        avg_series['Case Name'] = 'TOTAL_AVERAGE'
        
        # 自动计算所有数值类型的平均值
        numeric_cols = ['Pred_mCSA_mm2', 'GT_mCSA_mm2', 'Abs_Error_mm2', 'Relative_Error_Pct']
        for col in numeric_cols:
            if col in df_merged.columns:
                avg_series[col] = df_merged[col].mean()

        # 7. 使用 pd.concat 合并平均行
        df_final = pd.concat([df_merged, avg_series.to_frame().T], ignore_index=True)

        # 8. 四舍五入保留两位小数 (只针对存在的列)
        for col in numeric_cols:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col]).round(2)

        # 9. 保存结果
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        # 10. 终端简报输出
        print("=" * 60)
        print(f"✅ 评估报告已生成: {os.path.basename(OUTPUT_FILE)}")
        print(f"📊 样本总数: {len(df_merged)}")
        print("-" * 60)
        print(f"📈 平均绝对误差 (MAE): {avg_series['Abs_Error_mm2']:.2f} mm²")
        print(f"📉 平均相对误差 (MAPE): {avg_series['Relative_Error_Pct']:.2f} %")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 处理过程中出现异常: {e}")

if __name__ == "__main__":
    main()