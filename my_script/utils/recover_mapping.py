import os
import csv
from tqdm import tqdm
"""
    507任务将命名修改为了数字，这里构建一个csv文件来还原这个映射
"""

# ================= 配置区域 =================
BASE_DIR = "/data1/xyh/data/software_analysis2"
CBCT_DIR = os.path.join(BASE_DIR, "CBCT")
OUTPUT_CSV = "case_mapping_all.csv"
# ===========================================

def main():
    print("⚡ 启动带有进度指示的极速映射还原...\n")
    
    if not os.path.exists(CBCT_DIR):
        print(f"❌ 找不到目录: {CBCT_DIR}")
        return

    # 1. 纯粹依赖操作系统的底层读取顺序
    patient_folders = os.listdir(CBCT_DIR)
    patient_folders = [p for p in patient_folders if os.path.isdir(os.path.join(CBCT_DIR, p))]
    
    valid_cases = 0
    mapping_data = []

    # 2. 加上 tqdm 进度条进行遍历
    for patient_name in tqdm(patient_folders, desc="提取映射关系", unit="case"):
        case_identifier = f"Case_{valid_cases:03d}"
        
        # 将数据存为列表，方便后续写入 csv
        mapping_data.append([case_identifier, patient_name])
        valid_cases += 1

    # 3. 极速写入标准的 CSV 文件
    try:
        # 使用 utf-8-sig 编码，确保用 Excel 打开时绝对不会乱码
        with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(["Case_ID", "Original_Patient_Name"])
            # 批量写入所有行
            writer.writerows(mapping_data)
            
        print(f"\n✅ 还原成功！共提取 {valid_cases} 个病例的映射关系。")
        print(f"完整的对应表已保存至: {OUTPUT_CSV}")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()