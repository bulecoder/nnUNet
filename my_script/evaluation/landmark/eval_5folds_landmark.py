import os
import json
import numpy as np
import SimpleITK as sitk
"""
    评估5个fold的定点结果
"""

TASK_ID = 511   # 任务编号

# ================= 配置区域 (请确认路径) =================
# 1. 真实标签所在的文件夹 (labelsTr)
GT_DIR = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset{TASK_ID}_AirwayLandmarks/labelsTr"
# GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset508_AirwayHardLandmarks/labelsTr"


# 2. 你的 3d_fullres 主目录 (包含 fold_0, fold_1... 的那一层)
BASE_MODEL_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres"
# BASE_MODEL_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset508_AirwayHardLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres"


# 3. dataset.json 的路径
DATASET_JSON_PATH = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset{TASK_ID}_AirwayLandmarks/dataset.json"
# DATASET_JSON_PATH = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset508_AirwayHardLandmarks/dataset.json"


# 4. 【新功能：切换开关】你要评估哪个预测文件夹？
# 如果看原始结果，填 "validation"
# 如果看 LCC 重绘后的纯净结果，填 "validation_postprocessed_LCC"
# PRED_FOLDER_NAME = "validation"
PRED_FOLDER_NAME = "validation_postprocessed_LCC"


# 5. 最终汇总报告的保存路径 (自动加上后缀以区分不同版本的评估)
OUTPUT_TXT_PATH = os.path.join(BASE_MODEL_DIR, f"5_fold_CV_eval_{PRED_FOLDER_NAME}.txt")

# 6. 需要遍历的 Fold 列表和标签列表
FOLDS = [0, 1, 2, 3, 4]
LABELS = [1, 2, 3, 4, 5, 6, 7]
# =========================================================

def main():
    with open(DATASET_JSON_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    label_dict = dataset_info.get("labels", {})
    id_to_name = {int(v): str(k) for k, v in label_dict.items() if int(v) > 0}

    # 全局统计容器
    global_matched_distances = {L: [] for L in LABELS}
    global_stats = {L: {"matched": 0, "gt_only_missed": 0, "pred_only_hallucinated": 0} for L in LABELS}
    total_processed_files = 0

    print(f"🚀 开始执行 5 折交叉验证综合评估 (底层 C++ 极速模式)...\n")
    print(f"📁 目标预测目录: {PRED_FOLDER_NAME}")
    print("=" * 60)

    # 【核心极速引擎】复用滤波器对象，避免在循环中重复实例化
    shape_filter_gt = sitk.LabelShapeStatisticsImageFilter()
    shape_filter_pred = sitk.LabelShapeStatisticsImageFilter()

    for fold in FOLDS:
        pred_dir = os.path.join(BASE_MODEL_DIR, f"fold_{fold}", PRED_FOLDER_NAME)
        
        if not os.path.exists(pred_dir):
            print(f"⚠️ 警告: 找不到 Fold {fold} 的 {PRED_FOLDER_NAME} 文件夹，已跳过！")
            continue

        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
        print(f"正在处理 Fold {fold} ... 共找到 {len(pred_files)} 个预测文件。")

        for pred_file in pred_files:
            gt_path = os.path.join(GT_DIR, pred_file)
            pred_path = os.path.join(pred_dir, pred_file)

            if not os.path.exists(gt_path):
                continue

            # 仅读取对象，绝不转换为 numpy 数组
            gt_img = sitk.ReadImage(gt_path)
            pred_img = sitk.ReadImage(pred_path)

            # 在 C++ 底层瞬间计算所有标签的形状和物理属性
            shape_filter_gt.Execute(gt_img)
            shape_filter_pred.Execute(pred_img)

            for L in LABELS:
                has_gt = shape_filter_gt.HasLabel(L)
                has_pred = shape_filter_pred.HasLabel(L)

                if has_gt and has_pred:
                    # 直接获取真实的物理空间坐标 (x_mm, y_mm, z_mm)
                    c_gt_world = np.array(shape_filter_gt.GetCentroid(L))
                    c_pred_world = np.array(shape_filter_pred.GetCentroid(L))
                    
                    # 直接计算欧氏距离，自带物理单位，无需乘 spacing！
                    dist = np.linalg.norm(c_gt_world - c_pred_world)
                    
                    global_matched_distances[L].append(dist)
                    global_stats[L]["matched"] += 1
                    
                elif has_gt and not has_pred:
                    global_stats[L]["gt_only_missed"] += 1
                elif not has_gt and has_pred:
                    global_stats[L]["pred_only_hallucinated"] += 1
            
            total_processed_files += 1

    print("=" * 60)
    print(f"✅ 所有 Fold 处理完毕！共极速评估了 {total_processed_files} 个有效病例。\n")

    # ================= 汇总计算并生成报告 (排版微调以容纳 ± std) =================
    report_lines = []
    report_lines.append("=" * 96)
    report_lines.append(f"          5-Fold Cross Validation Overall Results ({PRED_FOLDER_NAME})")
    report_lines.append("=" * 96)
    # 将 MRE 列宽度由 10 提升至 16，以完美放下 "12.345 ± 6.789"
    report_lines.append(f"{'Label Name':<15} | {'有效匹配总数':<12} | {'MRE (mm)':<16} | {'SDR@2mm':<8} | {'SDR@3mm':<8} | {'SDR@4mm':<8}")
    report_lines.append("-" * 96)

    all_valid_dists = []

    for L in LABELS:
        name = id_to_name.get(L, f"L{L}")
        dists = np.array(global_matched_distances[L])
        num_matched = len(dists)
        
        if num_matched > 0:
            mre_mean = np.mean(dists)
            mre_std = np.std(dists) # 新增：计算标准差
            mre_str = f"{mre_mean:.3f} ± {mre_std:.3f}" # 新增：组合字符串
            sdr_2 = np.mean(dists <= 2.0) * 100
            sdr_3 = np.mean(dists <= 3.0) * 100
            sdr_4 = np.mean(dists <= 4.0) * 100
            all_valid_dists.extend(dists)
        else:
            mre_str = "nan"
            sdr_2 = sdr_3 = sdr_4 = float('nan')

        line = f"{name:<15} | {num_matched:<12} | {mre_str:<16} | {sdr_2:<7.1f}% | {sdr_3:<7.1f}% | {sdr_4:<7.1f}%"
        report_lines.append(line)

    report_lines.append("-" * 96)
    
    if len(all_valid_dists) > 0:
        overall_dists = np.array(all_valid_dists)
        o_mre_mean = np.mean(overall_dists)
        o_mre_std = np.std(overall_dists) # 新增：计算总体标准差
        o_mre_str = f"{o_mre_mean:.3f} ± {o_mre_std:.3f}" # 新增：组合总体字符串
        o_sdr_2 = np.mean(overall_dists <= 2.0) * 100
        o_sdr_3 = np.mean(overall_dists <= 3.0) * 100
        o_sdr_4 = np.mean(overall_dists <= 4.0) * 100
        total_matches = len(overall_dists)
        total_line = f"{'Total Summary':<15} | {total_matches:<12} | {o_mre_str:<16} | {o_sdr_2:<7.1f}% | {o_sdr_3:<7.1f}% | {o_sdr_4:<7.1f}%"
        report_lines.append(total_line)
        
    report_lines.append("=" * 96)

    report_lines.append(f"\n--- 全局附加信息 ({total_processed_files} 个病例的被忽略情况总计) ---")
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
        print(f"\n🎉 最终极速综合评估报告已成功保存至:\n   {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()