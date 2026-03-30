import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
"""
    读取目标任务的ANS和PNS的点，旋转label使其连线水平，记录这个修正的参数到txt文档中
    (直接读取真实姓名命名的文件，已彻底移除 CSV 映射逻辑)
"""

TASK_ID = 511

# ================= 配置区域 =================
GT_LMARK_DIR = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset{TASK_ID}_AirwayLandmarks/labelsTr"
PARAMS_OUTPUT = f"/data1/xyh/projects/nnUNet/my_script/results/csv/rotation_params_{TASK_ID}.txt"

# 根据 dataset.json 更新 ID
ANS_ID = 2
PNS_ID = 4
# ===========================================

def get_landmark_centroid(sitk_img, label_id):
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(sitk_img)
    if shape_filter.HasLabel(label_id):
        # 获取物理质心 (x, y, z)
        return np.array(shape_filter.GetCentroid(label_id))
    return None

def main():
    rotation_dict = {}

    # 1. 自动获取目录下所有的标签文件
    if not os.path.exists(GT_LMARK_DIR):
        print(f"❌ 错误: 找不到目录 {GT_LMARK_DIR}")
        return
        
    label_files = [f for f in os.listdir(GT_LMARK_DIR) if f.endswith('.nii.gz')]
    
    if not label_files:
        print(f"⚠️ 警告: 目录 {GT_LMARK_DIR} 下没有找到任何 .nii.gz 文件！")
        return

    print(f"🚀 开始遍历 {len(label_files)} 个病例，提取 ANS(ID:2) 和 PNS(ID:4) 计算旋转参数...")
    
    # 2. 直接遍历文件，提取真实姓名
    for filename in tqdm(label_files):
        # 剥离后缀，留下纯净的患者姓名
        patient_name = filename.replace('.nii.gz', '')
        
        lm_path = os.path.join(GT_LMARK_DIR, filename)
        lm_img = sitk.ReadImage(lm_path)
        
        p_ans = get_landmark_centroid(lm_img, ANS_ID)
        p_pns = get_landmark_centroid(lm_img, PNS_ID)

        # 检查是否缺失关键点
        if p_ans is not None and p_pns is not None:
            vector = p_ans - p_pns
            angle_rad = np.arctan2(vector[2], vector[1]) # 矢状位倾角
            center = (p_ans + p_pns) / 2
            rotation_dict[patient_name] = f"{angle_rad},{center[0]},{center[1]},{center[2]}"
        else:
            # 记录缺失状态
            rotation_dict[patient_name] = "MISSING"

    # 3. 按患者姓名排序写入文件，方便人类查阅
    with open(PARAMS_OUTPUT, "w", encoding="utf-8") as f:
        for name in sorted(rotation_dict.keys()):
            f.write(f"{name}|{rotation_dict[name]}\n")
    
    print(f"✅ 参数计算完成！已按患者姓名排序保存至: \n {PARAMS_OUTPUT} (含缺失值处理)")

if __name__ == "__main__":
    main()