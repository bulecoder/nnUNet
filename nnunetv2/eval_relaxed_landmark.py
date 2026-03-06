import os
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass

# ================= 配置区域 (请修改路径) =================
# 1. 真实标签所在的文件夹 (labelsTr)
GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset505_AirwayLandmarks/labelsTr"

# 2. 模型预测结果所在的文件夹 (你刚刚挑出来的 Fold 0 验证集)
PRED_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset505_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation"

# 3. dataset.json 的路径 (用于自动读取 Landmark 名字)
DATASET_JSON_PATH = "/data1/xyh/data/nnUNet/nnUNet_preprocessed/Dataset505_AirwayLandmarks/dataset.json"

# 4. 评估结果保存的 TXT 文件路径 (会自动创建)
OUTPUT_TXT_PATH = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset505_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval.txt"

# 5. 你的 Landmark 标签列表
LABELS = [1, 2, 3, 4, 5, 6, 7]
# LABELS = [1, 2, 3]
# =========================================================

def get_centroid(mask_array, label_id):
    """计算特定标签的质心坐标，如果不存在返回 None"""
    if np.sum(mask_array == label_id) == 0:
        return None
    return center_of_mass(mask_array == label_id)

def main():
    with open(DATASET_JSON_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    label_dict = dataset_info.get("labels", {})
    id_to_name = {int(v): str(k) for k, v in label_dict.items() if int(v) > 0}

    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')]
    
    matched_distances = {L: [] for L in LABELS}
    stats = {L: {"matched": 0, "gt_only_missed": 0, "pred_only_hallucinated": 0} for L in LABELS}

    total_files = len(pred_files)
    print(f"开始评估 {total_files} 个病例 (极度宽松模式)...\n")
    print("-" * 60)

    for idx, pred_file in enumerate(pred_files, 1):
        print(f"[{idx:02d}/{total_files}] 正在处理 {pred_file:<15} ... ", end="", flush=True)

        gt_path = os.path.join(GT_DIR, pred_file)
        pred_path = os.path.join(PRED_DIR, pred_file)

        if not os.path.exists(gt_path):
            print("找不到对应的 GT 文件，跳过！")
            continue

        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        spacing = np.array(gt_img.GetSpacing()[::-1])

        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        case_matched = 0
        case_missed = 0
        case_hallu = 0

        for L in LABELS:
            c_gt = get_centroid(gt_arr, L)
            c_pred = get_centroid(pred_arr, L)

            if c_gt is not None and c_pred is not None:
                diff_voxels = np.array(c_gt) - np.array(c_pred)
                dist = np.sqrt(np.sum((diff_voxels * spacing)**2))
                matched_distances[L].append(dist)
                stats[L]["matched"] += 1
                case_matched += 1
            elif c_gt is not None and c_pred is None:
                stats[L]["gt_only_missed"] += 1
                case_missed += 1
            elif c_gt is None and c_pred is not None:
                stats[L]["pred_only_hallucinated"] += 1
                case_hallu += 1

        print(f"完成! [成功匹配: {case_matched}, 漏检: {case_missed}, 误检: {case_hallu}]")

    # ================= 收集并打印结果 =================
    
    # 我们用一个列表来专门存储要写入 TXT 文件的所有文本行
    report_lines = []
    
    report_lines.append("=" * 90)
    report_lines.append(f"{'Label Name':<15} | {'有效匹配数':<10} | {'MRE (mm)':<10} | {'SDR@2mm':<8} | {'SDR@3mm':<8} | {'SDR@4mm':<8}")
    report_lines.append("-" * 90)

    all_valid_dists = []

    for L in LABELS:
        name = id_to_name.get(L, f"L{L}")
        dists = np.array(matched_distances[L])
        num_matched = len(dists)
        
        if num_matched > 0:
            mre = np.mean(dists)
            sdr_2 = np.mean(dists <= 2.0) * 100
            sdr_3 = np.mean(dists <= 3.0) * 100
            sdr_4 = np.mean(dists <= 4.0) * 100
            all_valid_dists.extend(dists)
        else:
            mre = sdr_2 = sdr_3 = sdr_4 = float('nan')

        line = f"{name:<15} | {num_matched:<10} | {mre:<10.3f} | {sdr_2:<7.1f}% | {sdr_3:<7.1f}% | {sdr_4:<7.1f}%"
        report_lines.append(line)

    report_lines.append("-" * 90)
    
    if len(all_valid_dists) > 0:
        overall_dists = np.array(all_valid_dists)
        o_mre = np.mean(overall_dists)
        o_sdr_2 = np.mean(overall_dists <= 2.0) * 100
        o_sdr_3 = np.mean(overall_dists <= 3.0) * 100
        o_sdr_4 = np.mean(overall_dists <= 4.0) * 100
        total_matches = len(overall_dists)
        total_line = f"{'Total':<15} | {total_matches:<10} | {o_mre:<10.3f} | {o_sdr_2:<7.1f}% | {o_sdr_3:<7.1f}% | {o_sdr_4:<7.1f}%"
        report_lines.append(total_line)
        
    report_lines.append("=" * 90)

    report_lines.append("\n--- 附加信息 (整个验证集被忽略的情况统计) ---")
    for L in LABELS:
        name = id_to_name.get(L, f"L{L}")
        missed = stats[L]["gt_only_missed"]
        hallu = stats[L]["pred_only_hallucinated"]
        if missed > 0 or hallu > 0:
            report_lines.append(f"{name:<15}: 漏检(GT有预测无) = {missed} 例, 误检(GT无预测有) = {hallu} 例")

    # 1. 在终端打印结果
    print("\n" + "\n".join(report_lines))

    # 2. 将结果写入到 TXT 文件
    try:
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\n✅ 评估报告已成功保存至: {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()