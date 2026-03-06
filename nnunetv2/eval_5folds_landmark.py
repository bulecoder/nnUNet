import os
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass

# ================= 配置区域 (请确认路径) =================
# 1. 真实标签所在的文件夹 (labelsTr)
GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset504_AirwayLandmarks/labelsTr"

# 2. 你的 3d_fullres 主目录 (包含 fold_0, fold_1... 的那一层)
BASE_MODEL_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset504_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres"

# 3. dataset.json 的路径
DATASET_JSON_PATH = "/data1/xyh/data/nnUNet/nnUNet_preprocessed/Dataset504_AirwayLandmarks/dataset.json"

# 4. 最终汇总报告的保存路径
OUTPUT_TXT_PATH = os.path.join(BASE_MODEL_DIR, "5_fold_cross_validation_eval.txt")

# 5. 需要遍历的 Fold 列表和标签列表
FOLDS = [0, 1, 2, 3, 4]
# LABELS = [1, 2, 3, 4, 5, 6, 7]
LABELS = [1, 2, 3]
# =========================================================

def get_centroid(mask_array, label_id):
    if np.sum(mask_array == label_id) == 0:
        return None
    return center_of_mass(mask_array == label_id)

def main():
    with open(DATASET_JSON_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    label_dict = dataset_info.get("labels", {})
    id_to_name = {int(v): str(k) for k, v in label_dict.items() if int(v) > 0}

    # 全局统计容器
    global_matched_distances = {L: [] for L in LABELS}
    global_stats = {L: {"matched": 0, "gt_only_missed": 0, "pred_only_hallucinated": 0} for L in LABELS}
    total_processed_files = 0

    print(f"开始执行 5 折交叉验证综合评估 (极度宽松模式)...\n")
    print("=" * 60)

    for fold in FOLDS:
        pred_dir = os.path.join(BASE_MODEL_DIR, f"fold_{fold}", "validation")
        
        if not os.path.exists(pred_dir):
            print(f"⚠️ 警告: 找不到 Fold {fold} 的 validation 文件夹，已跳过！路径: {pred_dir}")
            continue

        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
        print(f"正在处理 Fold {fold} ... 共找到 {len(pred_files)} 个预测文件。")

        for pred_file in pred_files:
            gt_path = os.path.join(GT_DIR, pred_file)
            pred_path = os.path.join(pred_dir, pred_file)

            if not os.path.exists(gt_path):
                continue

            gt_img = sitk.ReadImage(gt_path)
            pred_img = sitk.ReadImage(pred_path)
            spacing = np.array(gt_img.GetSpacing()[::-1])

            gt_arr = sitk.GetArrayFromImage(gt_img)
            pred_arr = sitk.GetArrayFromImage(pred_img)

            for L in LABELS:
                c_gt = get_centroid(gt_arr, L)
                c_pred = get_centroid(pred_arr, L)

                if c_gt is not None and c_pred is not None:
                    diff_voxels = np.array(c_gt) - np.array(c_pred)
                    dist = np.sqrt(np.sum((diff_voxels * spacing)**2))
                    global_matched_distances[L].append(dist)
                    global_stats[L]["matched"] += 1
                elif c_gt is not None and c_pred is None:
                    global_stats[L]["gt_only_missed"] += 1
                elif c_gt is None and c_pred is not None:
                    global_stats[L]["pred_only_hallucinated"] += 1
            
            total_processed_files += 1

    print("=" * 60)
    print(f"所有 Fold 处理完毕！共评估了 {total_processed_files} 个有效病例。\n")

    # ================= 汇总计算并生成报告 =================
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append(f"          5-Fold Cross Validation Overall Results (Relaxed Mode)")
    report_lines.append("=" * 90)
    report_lines.append(f"{'Label Name':<15} | {'有效匹配总数':<12} | {'MRE (mm)':<10} | {'SDR@2mm':<8} | {'SDR@3mm':<8} | {'SDR@4mm':<8}")
    report_lines.append("-" * 90)

    all_valid_dists = []

    for L in LABELS:
        name = id_to_name.get(L, f"L{L}")
        dists = np.array(global_matched_distances[L])
        num_matched = len(dists)
        
        if num_matched > 0:
            mre = np.mean(dists)
            sdr_2 = np.mean(dists <= 2.0) * 100
            sdr_3 = np.mean(dists <= 3.0) * 100
            sdr_4 = np.mean(dists <= 4.0) * 100
            all_valid_dists.extend(dists)
        else:
            mre = sdr_2 = sdr_3 = sdr_4 = float('nan')

        line = f"{name:<15} | {num_matched:<12} | {mre:<10.3f} | {sdr_2:<7.1f}% | {sdr_3:<7.1f}% | {sdr_4:<7.1f}%"
        report_lines.append(line)

    report_lines.append("-" * 90)
    
    if len(all_valid_dists) > 0:
        overall_dists = np.array(all_valid_dists)
        o_mre = np.mean(overall_dists)
        o_sdr_2 = np.mean(overall_dists <= 2.0) * 100
        o_sdr_3 = np.mean(overall_dists <= 3.0) * 100
        o_sdr_4 = np.mean(overall_dists <= 4.0) * 100
        total_matches = len(overall_dists)
        total_line = f"{'Total Summary':<15} | {total_matches:<12} | {o_mre:<10.3f} | {o_sdr_2:<7.1f}% | {o_sdr_3:<7.1f}% | {o_sdr_4:<7.1f}%"
        report_lines.append(total_line)
        
    report_lines.append("=" * 90)

    report_lines.append("\n--- 全局附加信息 (250 个病例的被忽略情况总计) ---")
    for L in LABELS:
        name = id_to_name.get(L, f"L{L}")
        missed = global_stats[L]["gt_only_missed"]
        hallu = global_stats[L]["pred_only_hallucinated"]
        if missed > 0 or hallu > 0:
            report_lines.append(f"{name:<15}: 总漏检(GT有预测无) = {missed} 例, 总误检(GT无预测有) = {hallu} 例")

    # 打印并保存
    print("\n".join(report_lines))

    try:
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\n✅ 最终综合评估报告已成功保存至:\n   {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()