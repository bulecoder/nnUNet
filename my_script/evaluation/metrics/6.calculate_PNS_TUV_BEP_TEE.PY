import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 标签所在的文件夹列表
LABEL_DIRS = [
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation_postprocessed_LCC"
]

# 2. 结果保存的 CSV 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "6.pns_tuv_bep_tee_tissue_area.csv")

# 3. 目标点的 ID
PNS_ID = 4
TUV_ID = 7
BEP_ID = 3
TEE_ID = 5
# ============================================

def get_landmark_centroid(shape_filter, label_id):
    """从形状滤波器中安全地获取物理质心坐标 (x, y, z)"""
    if shape_filter.HasLabel(label_id):
        return np.array(shape_filter.GetCentroid(label_id))
    return None

def main():
    results = []
    
    shape_filter = sitk.LabelShapeStatisticsImageFilter()

    print(f"🚀 开始计算矢状面软组织占位面积 (PNS-TUV-BEP-TEE)...")
    print("-" * 60)

    total_processed_files = 0

    for current_dir in LABEL_DIRS:
        if not os.path.exists(current_dir):
            print(f"⚠️ 警告: 找不到目录，已跳过 -> {current_dir}")
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
                
                # 获取 4 个关键点的物理坐标
                p_pns = get_landmark_centroid(shape_filter, PNS_ID)
                p_tuv = get_landmark_centroid(shape_filter, TUV_ID)
                p_bep = get_landmark_centroid(shape_filter, BEP_ID)
                p_tee = get_landmark_centroid(shape_filter, TEE_ID)

                area_mm2 = None
                status = "Success"

                if all(p is not None for p in [p_pns, p_tuv, p_bep, p_tee]):
                    # 选取 PNS 作为公共顶点，构建两个三角形的向量
                    # 三角形 1: PNS, TUV, BEP
                    vec_A1 = p_tuv - p_pns
                    vec_B1 = p_bep - p_pns
                    area1 = 0.5 * np.linalg.norm(np.cross(vec_A1, vec_B1))
                    
                    # 三角形 2: PNS, BEP, TEE (注意这里复用了 vec_B1)
                    vec_A2 = p_bep - p_pns 
                    vec_B2 = p_tee - p_pns
                    area2 = 0.5 * np.linalg.norm(np.cross(vec_A2, vec_B2))
                    
                    # 总面积
                    area_mm2 = area1 + area2
                else:
                    status = "Missing Landmark(s)"
                    # 具体记录缺失了哪些点
                    missing = []
                    if p_pns is None: missing.append("PNS")
                    if p_tuv is None: missing.append("TUV")
                    if p_bep is None: missing.append("BEP")
                    if p_tee is None: missing.append("TEE")
                    status = f"Missing: {','.join(missing)}"
                
                results.append({
                    "Patient_Name": patient_name,
                    "Pharyngeal_Area_mm2": area_mm2,
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
    
    valid_count = df['Pharyngeal_Area_mm2'].notna().sum()
    print("-" * 60)
    print(f"🎉 测量任务完成！共扫描了 {total_processed_files} 个文件。")
    print(f"📊 成功测量面积的样本数: {valid_count} / {len(df)}")
    if valid_count > 0:
        mean_area = df['Pharyngeal_Area_mm2'].mean()
        std_area = df['Pharyngeal_Area_mm2'].std()
        print(f"📐 总体矢状面软组织面积: {mean_area:.3f} ± {std_area:.3f} mm²")
    print(f"📂 详细报告已保存至: \n   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()