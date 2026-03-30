import os
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

"""
    检查507任务中生成的label和DIAN文件夹中的stl标注，关键点的分布是否一样
    结论：label和stl标注一致，也就是说预处理脚本逻辑正确
"""

# ================= 配置区域 =================
# 填写你生成的 labelsTr 绝对路径 (请确保 TASK_ID 对应你刚才生成的文件夹)
LABELS_TR_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset507_AirwayLandmarks/labelsTr"

LANDMARK_MAP = {
    1: "AICV", 2: "ANS", 3: "BEP",
    4: "PNS",  5: "TEE", 6: "TEP", 7: "TUV"
}
LANDMARKS = list(LANDMARK_MAP.values())
# ============================================

def main():
    if not os.path.exists(LABELS_TR_DIR):
        print(f"❌ 找不到目录: {LABELS_TR_DIR}")
        return

    points_distribution = defaultdict(int)
    missing_counts = {lm: 0 for lm in LANDMARKS}

    label_files = [f for f in os.listdir(LABELS_TR_DIR) if f.endswith('.nii.gz')]
    print(f"正在扫描 {len(label_files)} 个生成的 NIfTI 标签文件...\n")

    for label_file in label_files:
        file_path = os.path.join(LABELS_TR_DIR, label_file)

        # 1. 读取 NIfTI 文件
        try:
            sitk_img = sitk.ReadImage(file_path)
            img_array = sitk.GetArrayFromImage(sitk_img)
        except Exception as e:
            print(f"❌ 无法读取文件 {label_file}: {e}")
            continue

        # 2. 核心操作：获取图像中所有存在的唯一像素值
        # 例如，如果里面画了点，就会返回类似 {0, 1, 3, 5}
        unique_vals = set(np.unique(img_array))
        unique_vals.discard(0) # 移除背景值 0

        # 3. 将像素值映射回标签名称
        found_landmarks = set()
        for val in unique_vals:
            if val in LANDMARK_MAP:
                found_landmarks.add(LANDMARK_MAP[val])

        # 4. 统计分布
        num_found = len(found_landmarks)
        points_distribution[num_found] += 1

        # 5. 统计缺失
        for lm in LANDMARKS:
            if lm not in found_landmarks:
                missing_counts[lm] += 1

    # ================= 打印质检报告 =================
    print("==============================")
    print(f"{'点数(NIfTI标签)':<15} | {'病例个数'}")
    print("------------------------------")
    for i in sorted(points_distribution.keys()):
        print(f"{i:<19} | {points_distribution[i]}")
    print("------------------------------\n")

    print("==============================")
    print(" 📉 各关键点缺失详情 (NIfTI逆向检查)")
    print("==============================")
    print(f"{'关键点名称':<15} | {'缺失病例数'}")
    print("------------------------------")
    for lm in LANDMARKS:
        print(f"{lm:<19} | {missing_counts[lm]}")
    print("------------------------------")

if __name__ == "__main__":
    main()