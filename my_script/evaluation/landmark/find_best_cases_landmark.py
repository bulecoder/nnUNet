import os
import pandas as pd

"""
    在汇总的结果表中找出每个样本的平均误差、最大误差
"""

TASK_ID = 511

# ================= 配置区域 =================
# 1. 你的汇总 CSV 文件路径 (自动获取同级目录)
SCRIPT_DIR = os.path.dirname("/data1/xyh/projects/nnUNet/my_script/results/csv/")
CSV_PATH = os.path.join(SCRIPT_DIR, f"airway_landmark_{TASK_ID}_results_LCC.csv")

# 2. 标签列表
LABELS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误: 找不到文件 {CSV_PATH}")
        return

    print(f"📥 正在读取汇总数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 1. 数据清洗：将文本转为数值类型。
    # errors='coerce' 会把 'FP', 'Miss', '---' 全部变成 NaN
    for label in LABELS:
        df[label] = pd.to_numeric(df[label], errors='coerce')

    # 2. 核心筛选：剔除任何包含 NaN 的行
    # 这保证了留下来的样本，7 个点全部都成功预测到了！
    df_complete = df.dropna(subset=LABELS).copy()
    
    total_cases = len(df)
    complete_cases = len(df_complete)
    print(f"📊 数据库概况: 总样本数 {total_cases} 例 | 7个点全部成功预测的样本有 {complete_cases} 例\n")

    if complete_cases == 0:
        print("⚠️ 没有找到7个点全预测成功的样本！")
        return

    # 3. 计算这 7 个点的平均误差 (重新计算，保证精度)
    df_complete['Average_Error'] = df_complete[LABELS].mean(axis=1)
    
    # 计算这 7 个点中的最大误差 (木桶效应，最大误差越小说明越稳定)
    df_complete['Max_Error'] = df_complete[LABELS].max(axis=1)

    # 4. 按照“平均误差”从小到大排序 (也可以改成 by='Max_Error' 找最稳定的)
    df_best = df_complete.sort_values(by='Average_Error', ascending=True)

    # 5. 打印 Top 10 最完美的样本
    print("🏆 预测最完美的 Top 10 样本 (7点全中，平均误差极小):")
    print("=" * 90)
    print(f"{'Rank':<5} | {'Patient Name':<30} | {'Avg Error (mm)':<16} | {'Max Error (mm)':<16}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(df_best.head(10).iterrows(), 1):
        patient = str(row['Patient_Name'])
        avg_err = row['Average_Error']
        max_err = row['Max_Error']
        print(f"#{i:<4} | {patient:<30} | {avg_err:<16.3f} | {max_err:<16.3f}")
    print("=" * 90)

    # 6. 保存排序后的完整榜单，方便你后续在电脑上慢慢看
    output_csv = os.path.join(SCRIPT_DIR, f"best_cases_ranked_landmark_{TASK_ID}.csv")
    
    # 调整列顺序，把 Avg 和 Max 放在名字后面，方便查阅
    cols = ['Patient_Name', 'Average_Error', 'Max_Error'] + LABELS
    df_best = df_best[cols]
    
    df_best.to_csv(output_csv, index=False, float_format='%.3f', encoding='utf-8-sig')
    print(f"\n💾 完整的学霸排名表已保存至: \n   {output_csv}")

if __name__ == "__main__":
    main()