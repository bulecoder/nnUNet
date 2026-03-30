import os
import re
from collections import OrderedDict
import numpy as np
"""
    评估5个fold的分割结果
"""

TASK_ID = 510   # 任务编号

# ================= 配置区域 =================
# 1. 设置每个 Fold 评估结果 TXT 的路径
# 你可以根据实际情况修改这里的路径
# FOLD_PATHS = {
#     "Fold 0": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/fold0_eval.txt",
#     "Fold 1": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/fold1_eval.txt",
#     "Fold 2": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/fold2_eval.txt",
#     "Fold 3": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval.txt",
#     "Fold 4": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval.txt",
# }

FOLD_PATHS = {
    "Fold 0": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/fold0_eval_LCC.txt",
    "Fold 1": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/fold1_eval_LCC.txt",
    "Fold 2": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/fold2_eval_LCC.txt",
    "Fold 3": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval_LCC.txt",
    "Fold 4": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval_LCC.txt",
}

# 2. 最终汇总报告保存路径
# FINAL_REPORT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/5_fold_cv_eval.txt"
FINAL_REPORT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/5_fold_cv_eval_LCC.txt"
# ===========================================

def parse_all_cases_from_txt(file_path):
    """
    从 TXT 文件中解析所有病例的原始数据
    返回: list of (dice, iou, hd95, assd)
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        return []
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过表头，从第4行左右开始读取数据
            # 逻辑：寻找包含 "|" 且不包含 "Case Name", "===", "---", "AVERAGE" 的行
            for line in lines:
                if "|" in line and "Dice" not in line and "=" not in line and "-" not in line and "AVERAGE" not in line:
                    parts = line.split('|')
                    if len(parts) < 5: continue
                    try:
                        dice = float(parts[1].strip())
                        iou = float(parts[2].strip())
                        # 处理可能存在的 NaN
                        hd95_val = parts[3].strip()
                        assd_val = parts[4].strip()
                        hd95 = float(hd95_val) if hd95_val.lower() != 'nan' else np.nan
                        assd = float(assd_val) if assd_val.lower() != 'nan' else np.nan
                        data.append([dice, iou, hd95, assd])
                    except ValueError:
                        continue
    except Exception as e:
        print(f"❌ 解析 {file_path} 失败: {e}")
    return data

def format_mean_std(data_list, decimals=4):
    """计算均值和标准差并格式化为字符串"""
    arr = np.array(data_list)
    # 过滤掉 NaN
    valid_data = arr[~np.isnan(arr)]
    if len(valid_data) == 0:
        return "NaN"
    m = np.mean(valid_data)
    s = np.std(valid_data)
    if decimals == 4:
        return f"{m:.4f}±{s:.4f}"
    else:
        return f"{m:.3f}±{s:.3f}"

def main():
    print("🚀 开始读取并解析各 Fold 详细病例数据...")
    
    all_fold_raw_data = OrderedDict()
    global_dice, global_iou, global_hd95, global_assd = [], [], [], []

    for fold_name, path in FOLD_PATHS.items():
        case_data = parse_all_cases_from_txt(path)
        if case_data:
            all_fold_raw_data[fold_name] = case_data
            # 汇总到全局列表用于计算整体均值标准差
            for d in case_data:
                global_dice.append(d[0])
                global_iou.append(d[1])
                global_hd95.append(d[2])
                global_assd.append(d[3])
            print(f"✅ 已读取 {fold_name}: 共 {len(case_data)} 个病例")

    if not all_fold_raw_data:
        print("❌ 未能读取到任何有效的评估结果。")
        return

    # 生成汇总报告内容
    report = []
    # 增加列宽以容纳 ±std 字符串
    sep_line = "=" * 115
    report.append(sep_line)
    header = f"{'Fold':<15} | {'Dice (mean±std)':<22} | {'IoU (mean±std)':<22} | {'HD95 (mean±std)':<22} | {'ASSD (mean±std)':<22}"
    report.append(header)
    report.append("-" * 115)

    # 逐个 Fold 输出
    for fold_name, data in all_fold_raw_data.items():
        data_np = np.array(data)
        d_str = format_mean_std(data_np[:, 0], 4)
        i_str = format_mean_std(data_np[:, 1], 4)
        h_str = format_mean_std(data_np[:, 2], 3)
        a_str = format_mean_std(data_np[:, 3], 3)
        
        line = f"{fold_name:<15} | {d_str:<22} | {i_str:<22} | {h_str:<22} | {a_str:<22}"
        report.append(line)

    report.append("-" * 115)
    
    # 计算所有 Fold 汇总后的平均值 ± 标准差 (Overall CV Performance)
    overall_d_str = format_mean_std(global_dice, 4)
    overall_i_str = format_mean_std(global_iou, 4)
    overall_h_str = format_mean_std(global_hd95, 3)
    overall_a_str = format_mean_std(global_assd, 3)
    
    summary_line = f"{'OVERALL MEAN':<15} | {overall_d_str:<22} | {overall_i_str:<22} | {overall_h_str:<22} | {overall_a_str:<22}"
    report.append(summary_line)
    report.append(sep_line)

    final_report = "\n".join(report)
    
    # 终端输出
    print("\n" + final_report)

    # 保存文件
    try:
        with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(final_report + "\n")
        print(f"\n✅ 5-Fold 汇总报告(Mean±Std)已保存至: {FINAL_REPORT_PATH}")
    except Exception as e:
        print(f"❌ 保存汇总文件失败: {e}")

if __name__ == "__main__":
    main()