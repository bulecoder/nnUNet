import numpy as np
import SimpleITK as sitk
import trimesh
import os
"""
    检查 image 和 label 的几何一致性
"""

def check_alignment(image_path, label_path, stl_path):
    # 1. 加载数据
    img = sitk.ReadImage(image_path)
    lbl = sitk.ReadImage(label_path)
    mesh = trimesh.load(stl_path)
    
    # 2. 检查 Image 与 Label 的几何一致性 (这是训练成功的核心)
    print("=== [1] 图像 (Image) 与 标签 (Label) 对齐检查 ===")
    same_origin = np.allclose(img.GetOrigin(), lbl.GetOrigin())
    same_spacing = np.allclose(img.GetSpacing(), lbl.GetSpacing())
    same_direction = np.allclose(img.GetDirection(), lbl.GetDirection())
    
    print(f"原点 (Origin) 是否一致: {same_origin}")
    print(f"间距 (Spacing) 是否一致: {same_spacing}")
    print(f"方向 (Direction) 是否一致: {same_direction}")
    
    if not (same_origin and same_spacing and same_direction):
        print("❌ 警告：Image 与 Label 物理空间不一致！训练会失败。")
    else:
        print("✅ 成功：Image 与 Label 完美对齐。")

    # 3. 检查 STL 与 Label 的位移关系
    print("\n=== [2] STL 与 Label 空间位置检查 ===")
    
    # 计算 STL 的物理中心 (世界坐标)
    stl_centroid = mesh.centroid
    print(f"STL 物理中心 (World): {stl_centroid}")
    
    # 计算 Label 中气道的物理中心
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lbl)
    if stats.HasLabel(1):
        lbl_centroid = np.array(stats.GetCentroid(1))
        print(f"Label 气道物理中心 (World): {lbl_centroid}")
        
        # 计算欧几里得距离 (位移向量)
        displacement = lbl_centroid - stl_centroid
        distance = np.linalg.norm(displacement)
        
        print(f"物理空间位移向量 (LPS): {displacement}")
        print(f"总位移距离: {distance:.2f} mm")
        
        if distance < 1.0:
            print("✅ STL 与 Label 在物理空间基本重合。")
        else:
            print(f"ℹ️ 注意：STL 与 Label 存在 {distance:.2f}mm 的位移。")
            print("   这通常是由于 STL 导出时坐标系参考点不同导致的。")
    else:
        print("❌ 错误：Label 文件中没有找到气道标签 (值1)。")

# --- 请修改以下路径进行测试 ---
# CASE_NAME = "A.C._Roos_1979_11_18"
# CASE_NAME = "ZXM 22"
CASE_NAME = "CRAWFORD_Matthew_1963_5_21"
img_p = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset502_AirwaySegmentation/imagesTr/{CASE_NAME}_0000.nii.gz"
lbl_p = f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset502_AirwaySegmentation/labelsTr/{CASE_NAME}.nii.gz"
stl_p = f"/data1/xyh/data/software_analysis2/stl/{CASE_NAME}.stl"

check_alignment(img_p, lbl_p, stl_p)