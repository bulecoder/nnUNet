import os
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
from collections import OrderedDict
"""
    计算5个fold分割结果的
        "Sens": recall,
        "Spec": specificity,
        "Acc": accuracy,
        "F1": f1,
        "PPV": ppv,
        "NPV": npv,
        "MCC": mcc
    只使用了 label 和 validation 来计算
"""

# ================= 配置区域 =================
GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset506_AirwaySegmentation/labelsTr"
VAL_ROOT = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres"
USE_LCC = True 
SUB_DIR = "validation_postprocessed_LCC" if USE_LCC else "validation"

OUTPUT_REPORT = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/5fold_comprehensive_metrics_final.txt"
NUM_PROCESSES = 8 
# ===========================================

def calculate_confusion_matrix_3d(gt, pred):
    """高效计算3D二值图像的混淆矩阵元素"""
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    # 强制转为 Python 原生 int 避免 numpy 溢出
    return int(tp), int(tn), int(fp), int(fn)

def calculate_metrics_from_cm(tp, tn, fp, fn):
    """基于混淆矩阵计算各项指标，并解决 MCC 溢出问题"""
    # 基础指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    ppv = precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # MCC 计算: 转换为 float64 分步计算以防止溢出
    num = float(tp) * float(tn) - float(fp) * float(fn)
    # 分开求开方再相乘，防止四项连乘溢出
    den = np.sqrt(float(tp + fp)) * np.sqrt(float(tp + fn)) * \
          np.sqrt(float(tn + fp)) * np.sqrt(float(tn + fn))
    
    mcc = num / den if den > 0 else 0
    
    return {
        "Sens": recall,
        "Spec": specificity,
        "Acc": accuracy,
        "F1": f1,
        "PPV": ppv,
        "NPV": npv,
        "MCC": mcc
    }

def evaluate_case(args):
    f_name, gt_path, pred_path, fold_name = args
    try:
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        gt_arr = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
        pred_arr = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)
        
        tp, tn, fp, fn = calculate_confusion_matrix_3d(gt_arr, pred_arr)
        metrics = calculate_metrics_from_cm(tp, tn, fp, fn)
        metrics["Case"] = f_name
        metrics["Fold"] = fold_name
        return metrics
    except Exception:
        return None

def format_stats(val_list):
    if not val_list: return "N/A"
    return f"{np.mean(val_list):.4f} ± {np.std(val_list):.4f}"

def main():
    tasks = []
    for fold in range(5):
        fold_name = f"Fold {fold}"
        pred_dir = os.path.join(VAL_ROOT, f"fold_{fold}", SUB_DIR)
        if not os.path.exists(pred_dir): continue
        for f in os.listdir(pred_dir):
            if f.endswith(".nii.gz"):
                tasks.append((f, os.path.join(GT_DIR, f), os.path.join(pred_dir, f), fold_name))

    total = len(tasks)
    print(f"🚀 任务启动: 共有 {total} 个样本...")
    
    results = []
    # 使用 imap_unordered 提高多进程效率
    with mp.Pool(NUM_PROCESSES) as pool:
        for i, res in enumerate(pool.imap_unordered(evaluate_case, tasks)):
            if res:
                results.append(res)
            # 简单的文本进度报告，替代 tqdm
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"进度: [{i+1}/{total}] 处理中...")
    
    # 按 Fold 组织数据
    fold_data = OrderedDict()
    for r in results:
        f_name = r["Fold"]
        if f_name not in fold_data: fold_data[f_name] = []
        fold_data[f_name].append(r)

    # 写入报告
    with open(OUTPUT_REPORT, "w", encoding='utf-8') as f:
        f.write("COMPREHENSIVE SEGMENTATION METRICS REPORT\n")
        metric_keys = ["Sens", "Spec", "Acc", "F1", "PPV", "NPV", "MCC"]
        
        f.write("\n" + "="*145 + "\n")
        f.write(f"{'Fold':<10} | " + " | ".join([f"{k:<18}" for k in metric_keys]) + "\n")
        f.write("-" * 145 + "\n")
        
        global_metrics = {k: [] for k in metric_keys}
        for fold_idx in range(5):
            f_name = f"Fold {fold_idx}"
            if f_name in fold_data:
                cases = fold_data[f_name]
                fold_metrics = {k: [c[k] for c in cases] for k in metric_keys}
                row = f"{f_name:<10} | " + " | ".join([f"{format_stats(fold_metrics[k]):<18}" for k in metric_keys])
                f.write(row + "\n")
                for k in metric_keys: global_metrics[k].extend(fold_metrics[k])
            
        f.write("-" * 145 + "\n")
        overall_row = f"{'OVERALL':<10} | " + " | ".join([f"{format_stats(global_metrics[k]):<18}" for k in metric_keys])
        f.write(overall_row + "\n")
        f.write("="*145 + "\n")

    print(f"\n✅ 评估完成！结果保存至: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()