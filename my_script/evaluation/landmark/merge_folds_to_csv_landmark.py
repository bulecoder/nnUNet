import os
import csv
"""
    合并5个fold的定点评价结果到csv中
    (无需映射表版本：文件已直接使用患者真实姓名命名)
"""

TASK_ID = 511   # 任务编号

# ================= 配置区域 =================
# 1. 你的 5 个 Fold 的 txt 评估结果路径列表
EVAL_TXT_PATHS = [
    f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/fold0_eval_LCC.txt",
    f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/fold1_eval_LCC.txt",
    f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/fold2_eval_LCC.txt",
    f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval_LCC.txt",
    f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval_LCC.txt"
]

# 2. 最终汇总输出的 CSV 文件路径 (自动保存在结果csv文件夹中)
OUTPUT_MERGED_CSV = os.path.join("/data1/xyh/projects/nnUNet/my_script/results/csv", f"airway_landmark_{TASK_ID}_results_LCC.csv")

# 3. 你的 Landmark 标签列表
LABELS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================


def parse_eval_txt(txt_path):
    """解析单个 fold 的评估 txt 文件，提取详细样本评估矩阵"""
    case_data = {}
    if not os.path.exists(txt_path):
        print(f"⚠️ 警告: 找不到评估文件 {txt_path}，跳过该文件。")
        return case_data

    with open(txt_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    # 寻找数据表格的起始点
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("Case Name") and "|" in line:
            start_idx = i + 2 # 跳过表头和分割线
            break

    if start_idx == -1:
         print(f"❌ 错误: 在 {txt_path} 中未找到数据表格标记。")
         return case_data

    # 逐行读取数据直到遇到空行或汇总表
    for line in lines[start_idx:]:
        line = line.strip()
        if not line or line.startswith("=="):
            break
        
        parts = [p.strip() for p in line.split("|")]
        # 确保这行数据包含足够的列 (Case Name + N个Label + Avg)
        if len(parts) >= len(LABELS) + 2:
            patient_name = parts[0] # 现在的第一列直接就是真实的患者姓名
            # 提取各个标签的误差，忽略最后一列的 Case_Avg
            metrics = {label: val for label, val in zip(LABELS, parts[1:len(LABELS)+1])}
            case_data[patient_name] = metrics
            
    return case_data

def main():
    print(f"🚀 开始合并任务 {TASK_ID} 的 5-Fold 评估结果...")
    
    # 1. 收集所有 Fold 的数据
    all_case_results = {}
    for fold_idx, txt_path in enumerate(EVAL_TXT_PATHS):
        print(f"正在解析 Fold {fold_idx}: {os.path.basename(txt_path)}")
        fold_data = parse_eval_txt(txt_path)
        all_case_results.update(fold_data)
        
    if not all_case_results:
        print("❌ 未从任何 txt 文件中提取到数据，请检查文件路径和内容。")
        return

    # 2. 准备写入 CSV
    print(f"\n📝 准备生成最终的合并表格...")
    
    # 按患者姓名进行字母顺序排序 (由于现在名字是真实的字符串)
    sorted_patient_names = sorted(all_case_results.keys())
    
    # CSV 表头
    headers = ["Patient_Name"] + LABELS

    try:
        with open(OUTPUT_MERGED_CSV, mode='w', encoding='utf-8-sig', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for patient_name in sorted_patient_names:
                row_data = {
                    "Patient_Name": patient_name 
                }
                # 更新各个 Label 的误差数据
                row_data.update(all_case_results[patient_name])
                writer.writerow(row_data)
                
        print(f"🎉 合并完成！共处理了 {len(sorted_patient_names)} 个病例。")
        print(f"📂 文件已保存至: \n   {OUTPUT_MERGED_CSV}")
        
    except Exception as e:
        print(f"❌ 写入 CSV 失败: {e}")

if __name__ == "__main__":
    main()