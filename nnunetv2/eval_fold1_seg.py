import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
from pathlib import Path

# ================= 配置区域 (请修改路径) =================
# 1. 预处理后的真实标签文件夹 (labelsTr)
GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_preprocessed/Dataset502_AirwaySegmentation/gt_segmentations"

# 2. Fold 0 验证集预测结果文件夹
PRED_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation"

# 3. 评估结果保存的 TXT 文件路径
OUTPUT_TXT_PATH = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval.txt"
# =========================================================

def calculate_metrics(gt_arr, pred_arr, spacing):
    """计算二值化分割指标"""
    # 确保是二值图像
    gt_arr = (gt_arr > 0).astype(np.uint8)
    pred_arr = (pred_arr > 0).astype(np.uint8)
    
    if np.sum(pred_arr) == 0:
        return 0.0, float('nan')
        
    # 计算 Dice
    dsc = metric.binary.dc(pred_arr, gt_arr)
    
    # 计算 HD95 (如果无法计算则返回 nan)
    try:
        # medpy 需要 (z, y, x) 的顺序，与 spacing 的 (x, y, z) 对应
        hd95 = metric.binary.hd95(pred_arr, gt_arr, voxelspacing=spacing[::-1])
    except:
        hd95 = float('nan')
    
    return dsc, hd95

def main():
    if not os.path.exists(PRED_DIR):
        print(f"错误: 找不到预测结果目录 {PRED_DIR}")
        return

    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')])
    
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append(f"{'Case Name':<40} | {'Dice':<8} | {'HD95 (mm)':<8}")
    report_lines.append("-" * 65)

    all_dsc = []
    all_hd95 = []

    total_files = len(pred_files)
    print(f"开始评估 {total_files} 个病例...\n")

    for idx, f_name in enumerate(pred_files, 1):
        gt_path = os.path.join(GT_DIR, f_name)
        pred_path = os.path.join(PRED_DIR, f_name)

        if not os.path.exists(gt_path):
            print(f"[{idx:03d}] 找不到对应 GT: {f_name}，跳过！")
            continue

        # 读取图像
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        
        spacing = np.array(gt_img.GetSpacing())
        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        # 计算指标
        dsc, hd95 = calculate_metrics(gt_arr, pred_arr, spacing)
        
        all_dsc.append(dsc)
        if not np.isnan(hd95):
            all_hd95.append(hd95)

        # 格式化当前行
        hd95_str = f"{hd95:<8.3f}" if not np.isnan(hd95) else f"{'NaN':<8}"
        line = f"{f_name:<40} | {dsc:<8.4f} | {hd95_str}"
        report_lines.append(line)
        
        print(f"[{idx:03d}/{total_files}] {f_name:<35} | Dice: {dsc:.4f}")

    # 计算平均值
    report_lines.append("-" * 65)
    mean_dsc = np.mean(all_dsc) if all_dsc else 0
    mean_hd95 = np.mean(all_hd95) if all_hd95 else 0
    
    summary_line = f"{'AVERAGE':<40} | {mean_dsc:<8.4f} | {mean_hd95:<8.3f}"
    report_lines.append(summary_line)
    report_lines.append("=" * 65)

    # 1. 终端输出
    final_report = "\n".join(report_lines)
    print("\n" + final_report)

    # 2. 保存到 TXT 文件
    try:
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(final_report + "\n")
        print(f"\n✅ 评估结果已成功保存至: {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"\n❌ 保存失败: {e}")

if __name__ == "__main__":
    main()