import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
"""
    对分割结果进行LCC后处理
"""

TASK_ID = 510   # 任务编号
TASK_FOLD = 4   # 目标 fold

# ================= 配置区域 =================
# 1. 原始预测结果目录 (任务 506)
RAW_PRED_DIR = f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{TASK_FOLD}/validation"
# 2. LCC 后的输出目录
CLEAN_PRED_DIR = RAW_PRED_DIR + "_postprocessed_LCC"
# ===========================================

def apply_lcc_airway(sitk_img):
    """
    针对气道分割的 LCC：只保留最大体积的连通域，保持原始形状不变
    """
    # 1. 确保是二值图像 (0/1)
    binary_img = sitk.BinaryThreshold(sitk_img, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    
    # 2. 连通域分析 (C++ 底层)
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_img = cc_filter.Execute(binary_img)
    
    # 3. 统计形状属性
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(labeled_img)
    
    if shape_filter.GetNumberOfLabels() == 0:
        return sitk_img # 如果预测为空，原样返回

    # 4. 找到物理体积最大的标签 ID
    largest_label = max(shape_filter.GetLabels(), key=lambda l: shape_filter.GetPhysicalSize(l))
    
    # 5. 二值化提取该最大块 (仅保留 largest_label)
    lcc_mask = sitk.BinaryThreshold(labeled_img, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
    
    return lcc_mask

def main():
    os.makedirs(CLEAN_PRED_DIR, exist_ok=True)
    pred_files = sorted([f for f in os.listdir(RAW_PRED_DIR) if f.endswith('.nii.gz')])
    
    print(f"✨ 开始执行 {TASK_ID} 气道 LCC 后处理... 共计 {len(pred_files)} 例\n")
    
    for filename in tqdm(pred_files):
        raw_path = os.path.join(RAW_PRED_DIR, filename)
        clean_path = os.path.join(CLEAN_PRED_DIR, filename)
        
        # 读取
        raw_img = sitk.ReadImage(raw_path)
        
        # 执行 LCC 过滤
        clean_img = apply_lcc_airway(raw_img)
        
        # 复制元数据并保存
        clean_img.CopyInformation(raw_img)
        sitk.WriteImage(clean_img, clean_path)

    print(f"\n✅ LCC 处理完成！已保存至:\n{CLEAN_PRED_DIR}")

if __name__ == "__main__":
    main()