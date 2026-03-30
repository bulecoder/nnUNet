import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
"""
    读取目标任务的ANS和PNS的点 旋转label使其连线水平，记录这个修正的参数到txt文档中
"""

# ================= 配置区域 =================
GT_LMARK_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset507_AirwayLandmarks/labelsTr"
MAPPING_CSV = "case_mapping_all.csv"
PARAMS_OUTPUT = "rotation_params.txt"

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
    mapping_df = pd.read_csv(MAPPING_CSV)
    rotation_dict = {}

    print(f"🚀 开始从 507 任务提取 ANS(ID:2) 和 PNS(ID:4) 并计算旋转参数...")
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df)):
        case_507 = row['Case_ID']
        case_506 = row['Original_Patient_Name']
        
        lm_path = os.path.join(GT_LMARK_DIR, f"{case_507}.nii.gz")
        if not os.path.exists(lm_path):
            continue

        lm_img = sitk.ReadImage(lm_path)
        p_ans = get_landmark_centroid(lm_img, ANS_ID)
        p_pns = get_landmark_centroid(lm_img, PNS_ID)

        # 检查是否缺失关键点
        if p_ans is not None and p_pns is not None:
            vector = p_ans - p_pns
            angle_rad = np.arctan2(vector[2], vector[1]) # 矢状位倾角
            center = (p_ans + p_pns) / 2
            rotation_dict[case_506] = f"{angle_rad},{center[0]},{center[1]},{center[2]}"
        else:
            # 记录缺失状态
            rotation_dict[case_506] = "MISSING"

    with open(PARAMS_OUTPUT, "w") as f:
        for name, params in rotation_dict.items():
            f.write(f"{name}|{params}\n")
    
    print(f"✅ 参数保存至: {PARAMS_OUTPUT} (含缺失值处理)")

if __name__ == "__main__":
    main()