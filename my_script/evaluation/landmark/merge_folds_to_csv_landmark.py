import os
import csv
import re
"""
    合并5个fold的定点评价结果到csv中
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

# 2. 存放病例 ID 与真实姓名对应关系的 CSV 文件路径
NAME_MAPPING_CSV = "/data1/xyh/projects/nnUNet/nnunetv2/case_mapping_all.csv"

# 3. 最终汇总输出的 CSV 文件路径 (自动保存在当前脚本同级目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MERGED_CSV = os.path.join(SCRIPT_DIR, "airway_landmark_results_LCC.csv")

# 4. 你的 Landmark 标签列表
LABELS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================

def load_name_mapping(csv_path):
    """读取 Case_ID 到患者姓名的映射表"""
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"⚠️ 警告: 找不到姓名映射文件 {csv_path}。")
        return mapping
        
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        # 读取 CSV 时跳过空行
        reader = csv.DictReader(line for line in f if line.strip())
        for row in reader:
            case_id = row.get("Case_ID", "").strip()
            name = row.get("Original_Patient_Name", "").strip()
            if case_id:
                mapping[case_id] = name if name else case_id
    return mapping

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
            case_id = parts[0]
            # 提取各个标签的误差，忽略最后一列的 Case_Avg
            metrics = {label: val for label, val in zip(LABELS, parts[1:len(LABELS)+1])}
            case_data[case_id] = metrics
            
    return case_data

def main():
    print("🚀 开始合并 5-Fold 评估结果并替换为真实患者姓名...")
    
    # 1. 加载映射字典
    name_map = load_name_mapping(NAME_MAPPING_CSV)
    
    # 2. 收集所有 Fold 的数据
    all_case_results = {}
    for fold_idx, txt_path in enumerate(EVAL_TXT_PATHS):
        print(f"正在解析 Fold {fold_idx}: {os.path.basename(txt_path)}")
        fold_data = parse_eval_txt(txt_path)
        all_case_results.update(fold_data)
        
    if not all_case_results:
        print("❌ 未从任何 txt 文件中提取到数据，请检查文件路径和内容。")
        return

    # 3. 准备写入 CSV
    print(f"\n📝 准备生成最终的合并表格...")
    
    # 按 Case_ID 排序 (提取出数字部分进行数值排序)
    def sort_key(case_str):
        match = re.search(r'\d+', case_str)
        return int(match.group()) if match else float('inf'), case_str

    sorted_case_ids = sorted(all_case_results.keys(), key=sort_key)
    
    # 【改动点】CSV 表头不再包含 Case_ID，直接以 Patient_Name 开头
    headers = ["Patient_Name"] + LABELS

    try:
        with open(OUTPUT_MERGED_CSV, mode='w', encoding='utf-8-sig', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for case_id in sorted_case_ids:
                row_data = {
                    # 【改动点】将映射后的真实姓名直接赋给 Patient_Name
                    "Patient_Name": name_map.get(case_id, case_id) 
                }
                # 更新各个 Label 的误差数据
                row_data.update(all_case_results[case_id])
                writer.writerow(row_data)
                
        print(f"🎉 合并完成！共处理了 {len(sorted_case_ids)} 个病例。")
        print(f"📂 文件已保存至脚本同级目录: \n   {OUTPUT_MERGED_CSV}")
        
    except Exception as e:
        print(f"❌ 写入 CSV 失败: {e}")

if __name__ == "__main__":
    main()