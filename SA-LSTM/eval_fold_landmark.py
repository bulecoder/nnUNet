import os
import json
import numpy as np
import SimpleITK as sitk
"""
    评估单个fold的定点结果 (适配 Task 601 CBCT_Landmark)
"""

# ================= 配置区域 (请根据实际情况调整) =================
TASK_ID = 601   # 修改为新的任务编号
DATASET_NAME = "CBCT_Landmark" # 修改为新的数据集名称
TASK_FOLD = 0   # 目标 fold，请根据你实际训练的 fold 修改 (例如 0, 1, 2, 3, 4)

# 1. 真实标签所在的文件夹 (labelsTr)
GT_DIR = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset{TASK_ID}_{DATASET_NAME}/labelsTr"

# 2. 模型预测结果所在的文件夹 (验证集推理结果)
# PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation"
# PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation_postprocessed_LCC"
# 没有数据镜像的版本
# PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation"
PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation_postprocessed_LCC"


# 3. dataset.json 的路径 (用于自动读取 Landmark 名字)
DATASET_JSON_PATH = f"/data1/xyh/data/nnUNet/nnUNet_preprocessed/Dataset{TASK_ID}_{DATASET_NAME}/dataset.json"

# 4. 评估结果保存的 TXT 文件路径
# OUTPUT_TXT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/fold{TASK_FOLD}_eval.txt"
# OUTPUT_TXT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/fold{TASK_FOLD}_eval_LCC.txt"
# 没有数据镜像的版本
# OUTPUT_TXT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/fold{TASK_FOLD}_eval.txt"
OUTPUT_TXT_PATH = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_{DATASET_NAME}/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/fold{TASK_FOLD}_eval_LCC.txt"


# 5. Task 601 的 Landmark 标签列表 (1 到 12)
LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# =========================================================

def main():
    # 读取 dataset.json 获取标签名字映射
    with open(DATASET_JSON_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    label_dict = dataset_info.get("labels", {})
    id_to_name = {int(v): str(k) for k, v in label_dict.items() if int(v) > 0}

    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')]
    
    matched_distances = {L: [] for L in LABELS}
    stats = {L: {"matched": 0, "gt_only_missed": 0, "pred_only_hallucinated": 0} for L in LABELS}

    # 用于保存每个病例具体信息的字典 (TXT文件专用)
    case_details = {}

    total_files = len(pred_files)
    print(f"开始评估 {total_files} 个病例 (极度宽松模式)...\n")
    print("-" * 60)

    # 在循环外实例化滤波器，避免重复开销
    shape_filter_gt = sitk.LabelShapeStatisticsImageFilter()
    shape_filter_pred = sitk.LabelShapeStatisticsImageFilter()

    for idx, pred_file in enumerate(pred_files, 1):
        print(f"[{idx:02d}/{total_files}] 正在处理 {pred_file:<20} ... ", end="", flush=True)

        gt_path = os.path.join(GT_DIR, pred_file)
        pred_path = os.path.join(PRED_DIR, pred_file)

        if not os.path.exists(gt_path):
            print("找不到对应的 GT 文件，跳过！")
            case_details[pred_file] = {"status": "No GT"}
            continue

        # 仅读取 SimpleITK 对象
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)

        # 计算所有标签的形状和物理属性
        shape_filter_gt.Execute(gt_img)
        shape_filter_pred.Execute(pred_img)

        case_matched = 0
        case_missed = 0
        case_hallu = 0
        current_case_dists = []
        case_info = {}

        for L in LABELS:
            # 判断 GT 和 Pred 中是否存在该点
            has_gt = shape_filter_gt.HasLabel(L)
            has_pred = shape_filter_pred.HasLabel(L)

            if has_gt and has_pred:
                # 获取真实的物理空间坐标 (x_mm, y_mm, z_mm)
                c_gt_world = np.array(shape_filter_gt.GetCentroid(L))
                c_pred_world = np.array(shape_filter_pred.GetCentroid(L))
                
                # 计算欧氏距离
                dist = np.linalg.norm(c_gt_world - c_pred_world)
                
                matched_distances[L].append(dist)
                current_case_dists.append(dist)
                
                stats[L]["matched"] += 1
                case_matched += 1
                case_info[L] = f"{dist:.3f}"
            elif has_gt and not has_pred:
                stats[L]["gt_only_missed"] += 1
                case_missed += 1
                case_info[L] = "Miss"
            elif not has_gt and has_pred:
                stats[L]["pred_only_hallucinated"] += 1
                case_hallu += 1
                case_info[L] = "FP"
            else:
                case_info[L] = "---"

        # 计算并保存该病例的平均误差
        case_mre = np.mean(current_case_dists) if len(current_case_dists) > 0 else float('nan')
        case_info['avg'] = f"{case_mre:.3f}" if not np.isnan(case_mre) else "N/A"
        case_details[pred_file] = case_info

        print(f"完成! [成功匹配: {case_matched}, 漏检: {case_missed}, 误检: {case_hallu}]")

    # ================= 收集并打印结果 (终端用) =================
    
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

    # ================= 拼接并保存 TXT 文件 (含详细矩阵) =================
    txt_content = []
    txt_content.append("==========================================================================================================")
    txt_content.append("                              详细样本评估矩阵 (每行代表一个病例)")
    txt_content.append("==========================================================================================================")
    
    # 动态构建表头
    header_col1 = "Case Name"
    label_headers = [f"{id_to_name.get(L, f'L{L}'):<8}" for L in LABELS]
    header = f"{header_col1:<20} | " + " | ".join(label_headers) + " | Case_Avg (mm)"
    txt_content.append(header)
    txt_content.append("-" * 150) # 延长分隔线以适应更多标签

    for pred_file in sorted(pred_files):
        info = case_details.get(pred_file, {})
        if info.get("status") == "No GT":
            txt_content.append(f"{pred_file.replace('.nii.gz',''):<20} | 找不到对应的真实标签 (GT Missing)")
            continue
        
        row_str = f"{pred_file.replace('.nii.gz',''):<20} | "
        row_str += " | ".join([f"{info.get(L, ''):<8}" for L in LABELS])
        row_str += f" | {info.get('avg', ''):<10}"
        txt_content.append(row_str)

    txt_content.append("\n\n")
    txt_content.append("==========================================================================================================")
    txt_content.append("                                        总体评价结果汇总")
    
    txt_content.extend(report_lines)

    try:
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_content) + "\n")
        print(f"\n✅ 评估报告已成功保存至: {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()