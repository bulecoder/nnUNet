import os
import glob
import re
from collections import defaultdict
"""
    检查统计原始数据（DIAN文件夹中）标注的分布（缺失）情况
"""

# ================= 配置区域 =================
DIAN_DIR = "/data1/xyh/data/software_analysis2/DIAN"
LANDMARKS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================

def is_landmark_in_filename(landmark, filename):
    """
    【终极防弹墙：精准正则匹配】
    防止类似 Hermans 误判为 ANS，或者 STEPHANIS 误判为 TEP 的子串陷阱。
    """
    pattern = re.compile(rf"(^|[^a-zA-Z]){landmark}([^a-zA-Z]|$)", re.IGNORECASE)
    return bool(pattern.search(filename))

def main():
    points_distribution = defaultdict(int)
    missing_counts = {lm: 0 for lm in LANDMARKS}

    patient_folders = [f for f in os.listdir(DIAN_DIR) if os.path.isdir(os.path.join(DIAN_DIR, f))]
    print(f"正在扫描 {len(patient_folders)} 个患者目录...\n")

    for folder in patient_folders:
        folder_path = os.path.join(DIAN_DIR, folder)
        
        # 融合 1：使用 glob 继承强大的底层搜索能力，绝不漏掉特殊后缀
        stl_files = glob.glob(os.path.join(folder_path, "*.stl"))

        found_landmarks = set()
        for stl_path in stl_files:
            fname = os.path.basename(stl_path) # 正则引擎自带大小写忽略，这里不用提前 lower()
            
            for lm in LANDMARKS:
                # 融合 2：使用正则匹配替换掉危险的 if lm.lower() in fname
                if is_landmark_in_filename(lm, fname):
                    # 融合 3：使用 set 继承去重能力，绝不重复计数
                    found_landmarks.add(lm)
                    break 

        num_found = len(found_landmarks)
        points_distribution[num_found] += 1

        for lm in LANDMARKS:
            if lm not in found_landmarks:
                missing_counts[lm] += 1

    # 打印精美结果
    print("==============================")
    print(f"{'点数(有效STL)':<15} | {'病例个数'}")
    print("------------------------------")
    for i in sorted(points_distribution.keys()):
        print(f"{i:<19} | {points_distribution[i]}")
    print("------------------------------\n")

    print("==============================")
    print(" 📉 各关键点缺失详情")
    print("==============================")
    print(f"{'关键点名称':<15} | {'缺失病例数'}")
    print("------------------------------")
    for lm in LANDMARKS:
        print(f"{lm:<19} | {missing_counts[lm]}")
    print("------------------------------")

if __name__ == "__main__":
    main()