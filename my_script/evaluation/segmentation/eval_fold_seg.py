import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import multiprocessing as mp
from tqdm import tqdm
"""
    评估单个fold的分割结果
"""

TASK_ID = 510   # 任务编号
TASK_FOLD = 4   # 目标 fold

# ================= 配置区域 =================
# 1. 真实标签文件夹
GT_DIR = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset{TASK_ID}_AirwaySegmentation/labelsTr"
# 2. 预测结果文件夹
# PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation"
PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation_postprocessed_LCC"

# 3. 结果保存路径
# OUTPUT_TXT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/fold{TASK_FOLD}_eval.txt"
OUTPUT_TXT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/fold{TASK_FOLD}_eval_LCC.txt"

# 4. 设置并行进程数量 (根据服务器负载调整)
NUM_PROCESSES = 8
# ===========================================

def evaluate_single_case(f_name):
    """单个病例的评估逻辑，用于多进程调用"""
    gt_path = os.path.join(GT_DIR, f_name)
    pred_path = os.path.join(PRED_DIR, f_name)

    if not os.path.exists(gt_path):
        return f_name, None, None, None, None

    try:
        # 读取图像
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        
        spacing = gt_img.GetSpacing()
        # 仅在计算指标前转换为 Numpy
        gt_arr = sitk.GetArrayFromImage(gt_img).astype(np.uint8)
        pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.uint8)

        # 确保二值化
        gt_arr = (gt_arr > 0).astype(np.uint8)
        pred_arr = (pred_arr > 0).astype(np.uint8)

        if np.sum(pred_arr) == 0:
            return f_name, 0.0, 0.0, float('nan'), float('nan')

        # 计算 Dice 和 IoU (Jaccard)
        dsc = metric.binary.dc(pred_arr, gt_arr)
        iou = metric.binary.jc(pred_arr, gt_arr)
        
        # 计算 HD95 和 ASSD
        try:
            # 传入切片间距 spacing[::-1] (z, y, x)
            hd95 = metric.binary.hd95(pred_arr, gt_arr, voxelspacing=spacing[::-1])
            assd = metric.binary.assd(pred_arr, gt_arr, voxelspacing=spacing[::-1])
        except:
            hd95 = float('nan')
            assd = float('nan')

        return f_name, dsc, iou, hd95, assd
    except Exception as e:
        print(f"Error processing {f_name}: {e}")
        return f_name, None, None, None, None

def main():
    if not os.path.exists(PRED_DIR):
        print(f"错误: 找不到预测结果目录 {PRED_DIR}")
        return

    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')])
    total_files = len(pred_files)
    
    # 仅在控制台告知当前进程配置
    print(f"🚀 开始并行评估 {total_files} 个病例 (使用 {NUM_PROCESSES} 个进程)...")

    # 使用多进程池
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(evaluate_single_case, pred_files), total=total_files))

    # 整理结果 (增加 IoU 和 ASSD 列)
    report_lines = []
    report_lines.append("=" * 95)
    report_lines.append(f"{'Case Name':<40} | {'Dice':<8} | {'IoU':<8} | {'HD95(mm)':<10} | {'ASSD(mm)':<10}")
    report_lines.append("-" * 95)

    all_dsc = []
    all_iou = []
    all_hd95 = []
    all_assd = []

    for f_name, dsc, iou, hd95, assd in results:
        if dsc is None:
            continue
            
        all_dsc.append(dsc)
        all_iou.append(iou)
        if not np.isnan(hd95):
            all_hd95.append(hd95)
        if not np.isnan(assd):
            all_assd.append(assd)

        hd95_str = f"{hd95:<10.3f}" if not np.isnan(hd95) else f"{'NaN':<10}"
        assd_str = f"{assd:<10.3f}" if not np.isnan(assd) else f"{'NaN':<10}"
        line = f"{f_name:<40} | {dsc:<8.4f} | {iou:<8.4f} | {hd95_str} | {assd_str}"
        report_lines.append(line)

    # 计算平均值
    report_lines.append("-" * 95)
    mean_dsc = np.mean(all_dsc) if all_dsc else 0
    mean_iou = np.mean(all_iou) if all_iou else 0
    mean_hd95 = np.mean(all_hd95) if all_hd95 else 0
    mean_assd = np.mean(all_assd) if all_assd else 0
    
    summary_line = f"{'AVERAGE':<40} | {mean_dsc:<8.4f} | {mean_iou:<8.4f} | {mean_hd95:<10.3f} | {mean_assd:<10.3f}"
    report_lines.append(summary_line)
    report_lines.append("=" * 95)

    # 输出与保存
    final_report = "\n".join(report_lines)
    print("\n" + final_report)

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(final_report + "\n")
    print(f"\n✅ 评估完成！报告已保存至: {OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    main()