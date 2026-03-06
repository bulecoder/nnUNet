import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
from pathlib import Path
from collections import OrderedDict

# ================= 配置区域 =================
# 1. 真实标签文件夹
GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_preprocessed/Dataset502_AirwaySegmentation/gt_segmentations"

# 2. 结果根目录 (自动遍历下方的 fold_0 到 fold_4)
RESULTS_ROOT = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres"

# 3. 汇总报告保存路径
FINAL_REPORT_PATH = os.path.join(RESULTS_ROOT, "5_fold_cross_validation_eval.txt")
# ===========================================

def calculate_metrics(gt_arr, pred_arr, spacing):
    gt_arr = (gt_arr > 0).astype(np.uint8)
    pred_arr = (pred_arr > 0).astype(np.uint8)
    
    if np.sum(pred_arr) == 0:
        return 0.0, float('nan')
        
    dsc = metric.binary.dc(pred_arr, gt_arr)
    try:
        hd95 = metric.binary.hd95(pred_arr, gt_arr, voxelspacing=spacing[::-1])
    except:
        hd95 = float('nan')
    
    return dsc, hd95

def main():
    fold_summaries = OrderedDict()
    all_folds_dsc = []
    all_folds_hd95 = []

    # 遍历 5 个 Fold
    for fold in range(5):
        fold_dir = os.path.join(RESULTS_ROOT, f"fold_{fold}", "validation")
        if not os.path.exists(fold_dir):
            print(f"⚠️ 跳过: 找不到 {fold_dir}")
            continue

        print(f"正在评估 Fold {fold}...")
        pred_files = sorted([f for f in os.listdir(fold_dir) if f.endswith('.nii.gz')])
        
        fold_dsc = []
        fold_hd95 = []

        for f_name in pred_files:
            gt_path = os.path.join(GT_DIR, f_name)
            pred_path = os.path.join(fold_dir, f_name)

            if not os.path.exists(gt_path): continue

            gt_img = sitk.ReadImage(gt_path)
            pred_img = sitk.ReadImage(pred_path)
            
            spacing = np.array(gt_img.GetSpacing())
            gt_arr = sitk.GetArrayFromImage(gt_img)
            pred_arr = sitk.GetArrayFromImage(pred_img)

            dsc, hd95 = calculate_metrics(gt_arr, pred_arr, spacing)
            
            fold_dsc.append(dsc)
            if not np.isnan(hd95): fold_hd95.append(hd95)

        # 记录该 Fold 的平均值
        f_mean_dsc = np.mean(fold_dsc) if fold_dsc else 0
        f_mean_hd95 = np.mean(fold_hd95) if fold_hd95 else 0
        fold_summaries[f"Fold {fold}"] = (f_mean_dsc, f_mean_hd95)
        
        all_folds_dsc.extend(fold_dsc)
        all_folds_hd95.extend(fold_hd95)

    # 生成最终报告
    report = []
    report.append("=" * 50)
    report.append(f"{'Fold':<15} | {'Mean Dice':<12} | {'Mean HD95 (mm)':<12}")
    report.append("-" * 50)

    for fold_name, (m_dsc, m_hd95) in fold_summaries.items():
        report.append(f"{fold_name:<15} | {m_dsc:<12.4f} | {m_hd95:<12.3f}")

    report.append("-" * 50)
    final_mean_dsc = np.mean(all_folds_dsc) if all_folds_dsc else 0
    final_mean_hd95 = np.mean(all_folds_hd95) if all_folds_hd95 else 0
    report.append(f"{'OVERALL MEAN':<15} | {final_mean_dsc:<12.4f} | {final_mean_hd95:<12.3f}")
    report.append("=" * 50)

    final_report = "\n".join(report)
    print("\n" + final_report)

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(final_report + "\n")
    print(f"\n✅ 全 Fold 汇总报告已保存至: {FINAL_REPORT_PATH}")

if __name__ == "__main__":
    main()