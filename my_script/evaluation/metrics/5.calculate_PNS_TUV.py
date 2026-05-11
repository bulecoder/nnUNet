import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
"""
    计算目标目录下 PNS-TUV 的距离 单位mm
    计算软腭高度，在医学头影测量学中 软腭高度定义为 悬雍垂尖端（TUV）到硬腭平面（Palatal Plane, 即 ANS-PNS 连线所在的平面）的垂直距离。
"""

# ================= 配置区域 =================
# 1. 标签所在的文件夹列表 (List 格式)
LABEL_DIRS = [
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation_postprocessed_LCC"
]

# 2. 结果保存的 CSV 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "5.pns_tuv_and_height.csv")

# 3. 目标点的 ID (ANS:2, PNS:4, TUV:7)
ANS_ID = 2
PNS_ID = 4
TUV_ID = 7
# ============================================

def get_landmark_centroid(shape_filter, label_id):
    """从形状滤波器中安全地获取物理质心坐标 (x, y, z)"""
    if shape_filter.HasLabel(label_id):
        return np.array(shape_filter.GetCentroid(label_id))
    return None

def main():
    results = []
    
    # 实例化 C++ 底层的形状分析滤波器
    shape_filter = sitk.LabelShapeStatisticsImageFilter()

    print(f"🚀 开始计算软腭形态: PNS-TUV 长度 & TUV 到 ANS-PNS 平面的垂直高度...")
    print("-" * 60)

    total_processed_files = 0

    for current_dir in LABEL_DIRS:
        if not os.path.exists(current_dir):
            continue

        label_files = [f for f in os.listdir(current_dir) if f.endswith('.nii.gz')]
        folder_name = os.path.basename(os.path.dirname(current_dir)) + "/" + os.path.basename(current_dir)
        
        if not label_files:
            continue

        print(f"📁 正在处理: [{folder_name}] (共 {len(label_files)} 个文件)")
        
        for filename in tqdm(label_files, leave=False):
            patient_name = filename.replace('.nii.gz', '')
            label_path = os.path.join(current_dir, filename)
            
            try:
                label_img = sitk.ReadImage(label_path)
                shape_filter.Execute(label_img)
                
                # 获取关键点的物理坐标
                p_ans = get_landmark_centroid(shape_filter, ANS_ID)
                p_pns = get_landmark_centroid(shape_filter, PNS_ID)
                p_tuv = get_landmark_centroid(shape_filter, TUV_ID)

                length_mm = None
                height_mm = None
                status = "Success"

                # 1. 计算软腭长度 (只需要 PNS 和 TUV)
                if p_pns is not None and p_tuv is not None:
                    length_mm = np.linalg.norm(p_pns - p_tuv)
                else:
                    status = "Missing PNS or TUV"

                # 2. 计算软腭垂直下垂高度 (需要 ANS, PNS, TUV 三个点齐备)
                if p_ans is not None and p_pns is not None and p_tuv is not None:
                    # 向量 A: 硬腭连线 (PNS -> ANS)
                    vec_A = p_ans - p_pns
                    # 向量 B: 软腭斜线 (PNS -> TUV)
                    vec_B = p_tuv - p_pns
                    
                    # 向量叉乘的模，除以硬腭连线的模，得到 TUV 到直线的垂直距离
                    area = np.linalg.norm(np.cross(vec_A, vec_B))
                    base_length = np.linalg.norm(vec_A)
                    
                    if base_length > 0:
                        height_mm = area / base_length
                    else:
                        status = "Error: ANS and PNS are identical"
                elif status == "Success":
                    # 如果长度算出来了，但高度没算出来，说明缺了 ANS
                    status = "Missing ANS for Height"
                
                results.append({
                    "Patient_Name": patient_name,
                    "PNS_TUV_Length_mm": length_mm,
                    "Soft_Palate_Height_mm": height_mm,
                    "Status": status
                })
                total_processed_files += 1
                
            except Exception as e:
                print(f"\n❌ 读取文件失败 {filename}: {e}")

    if not results:
        print("❌ 未从任何目录中提取到有效数据，程序退出。")
        return

    # ================= 整理并保存结果 =================
    df = pd.DataFrame(results).sort_values(by='Patient_Name')
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.3f', encoding='utf-8-sig')
    
    # 统计信息打印
    valid_length = df['PNS_TUV_Length_mm'].notna().sum()
    valid_height = df['Soft_Palate_Height_mm'].notna().sum()
    
    print("-" * 60)
    print(f"🎉 测量任务完成！共扫描了 {total_processed_files} 个文件。")
    if valid_length > 0:
        print(f"📏 总体 PNS-TUV 长度: {df['PNS_TUV_Length_mm'].mean():.3f} ± {df['PNS_TUV_Length_mm'].std():.3f} mm (样本数: {valid_length})")
    if valid_height > 0:
        print(f"📐 总体 软腭垂直高度: {df['Soft_Palate_Height_mm'].mean():.3f} ± {df['Soft_Palate_Height_mm'].std():.3f} mm (样本数: {valid_height})")
    print(f"📂 详细测量报告已保存至: \n   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()