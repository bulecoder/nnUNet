import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
"""
    对预测结果进行后处理，最大连通域（LCC）计算，计算出主区域的质心，进行半径为6的重绘
"""

TASK_ID = 511   # 任务编号
TASK_FOLD = 4   # 目标 fold

# ================= 配置区域 =================
# 1. 原始预测结果目录 (nnUNet 刚刚输出的文件夹，注意修改为你的 fold 目录)
RAW_PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation"
# RAW_PRED_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset508_AirwayHardLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation"

# 2. 重绘后的最终输出目录
CLEAN_PRED_DIR = RAW_PRED_DIR + "_postprocessed_LCC"

# 3. 标签列表与物理半径
# LABELS = [1, 2, 3]
LABELS = [1, 2, 3, 4, 5, 6, 7]
SPHERE_RADIUS_MM = 6.0
# SPHERE_RADIUS_MM = 8.0
# ===========================================

def get_lcc_centroid_voxel(sitk_img, label_id):
    """
    【极速优化】全量使用 SimpleITK 底层 C++ 执行，不涉及 Numpy 转换。
    返回图像体素坐标系下的质心 (x, y, z)
    """
    # 1. 二值化提取 (底层 C++，极快)
    binary_thresh = sitk.BinaryThresholdImageFilter()
    binary_thresh.SetLowerThreshold(label_id)
    binary_thresh.SetUpperThreshold(label_id)
    binary_thresh.SetInsideValue(1)
    binary_thresh.SetOutsideValue(0)
    binary_img = binary_thresh.Execute(sitk_img)
    
    # 如果没有预测出这个点，直接返回 None
    stat_filter = sitk.StatisticsImageFilter()
    stat_filter.Execute(binary_img)
    if stat_filter.GetMaximum() == 0:
        return None
    
    # 2. 连通域分析
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_img = cc_filter.Execute(binary_img)
    
    # 3. 提取连通域形状属性 (质心在这里就被顺便算出来了)
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(labeled_img)
    
    if shape_filter.GetNumberOfLabels() == 0:
        return None
        
    # 4. 找到物理体积最大的碎片 ID
    largest_label = max(shape_filter.GetLabels(), key=lambda l: shape_filter.GetPhysicalSize(l))
    
    # 5. 直接获取金标准物理质心 (x_mm, y_mm, z_mm)
    physical_centroid = shape_filter.GetCentroid(largest_label)
    
    # 6. 将物理坐标转为体素(像素)坐标系 (x, y, z)，供后续 numpy 画图用
    voxel_centroid = sitk_img.TransformPhysicalPointToContinuousIndex(physical_centroid)
    
    return voxel_centroid

def draw_sphere_xyz(mask_array, center_xyz, radius_px, label_id):
    """在 numpy 数组的指定中心 (x, y, z) 处绘制球体"""
    shape = mask_array.shape
    r = max(1, int(radius_px))
    
    # 注意：center_xyz 是 (x, y, z)，而 Numpy 数组形状是 (z, y, x)
    cx, cy, cz = center_xyz
    
    z_min = max(0, int(cz - r - 1))
    z_max = min(shape[0], int(cz + r + 2))
    y_min = max(0, int(cy - r - 1))
    y_max = min(shape[1], int(cy + r + 2))
    x_min = max(0, int(cx - r - 1))
    x_max = min(shape[2], int(cx + r + 2))

    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    
    mask_roi = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_roi[dist_sq <= r**2] = label_id

def main():
    os.makedirs(CLEAN_PRED_DIR, exist_ok=True)
    
    pred_files = sorted([f for f in os.listdir(RAW_PRED_DIR) if f.endswith('.nii.gz')])
    print(f"✨ 开始执行【极速版：质心提取 + 完美重绘】... 共计 {len(pred_files)} 例\n")
    
    for filename in tqdm(pred_files, desc="重绘进度"):
        raw_path = os.path.join(RAW_PRED_DIR, filename)
        clean_path = os.path.join(CLEAN_PRED_DIR, filename)
        
        # 仅读取一次图像，且保持为 SimpleITK 对象
        raw_img = sitk.ReadImage(raw_path)
        
        # 创建一个全黑的空白 numpy 画布
        clean_arr = np.zeros(raw_img.GetSize()[::-1], dtype=np.uint8)
        
        avg_spacing = np.mean(raw_img.GetSpacing())
        radius_pixels = SPHERE_RADIUS_MM / avg_spacing
        
        for L in LABELS:
            # 1. 极速提取 LCC 体素坐标
            voxel_centroid = get_lcc_centroid_voxel(raw_img, L)
            
            # 2. 在空白画布上局部重绘 (局部赋值，极快)
            if voxel_centroid is not None:
                draw_sphere_xyz(clean_arr, voxel_centroid, radius_pixels, L)
                
        # 3. 仅在保存时，将 Numpy 转回 SimpleITK
        clean_img = sitk.GetImageFromArray(clean_arr)
        clean_img.CopyInformation(raw_img)
        sitk.WriteImage(clean_img, clean_path)

    print(f"\n✅ 极速重绘完成！像 GT 一样完美的预测结果已保存至:\n{CLEAN_PRED_DIR}")

if __name__ == "__main__":
    main()