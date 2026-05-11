import os
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
"""
    统计原始图像的分辨率 判断原始图像分辨率是否会影响到最后的定点和分割结果
    原始DICOM数据读取起来太慢了 很多冗余信息
    preprocessed里面的数据是nnUnet重采样以后的数据 分辨率已经统一
"""

# ================= 配置区域 =================
# 指向你任务的原始 imagesTr 目录 (请修改为你的实际路径)
RAW_IMAGES_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset511_AirwayLandmarks/imagesTr"
OUTPUT_CSV = "image_resolutions_info_511.csv"
# ============================================

def main():
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ 找不到目录: {RAW_IMAGES_DIR}")
        return

    image_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith('.nii.gz')]
    results = []

    print(f"🚀 开始提取 {len(image_files)} 个原始图像的分辨率信息...")

    for filename in tqdm(image_files):
        # 提取患者姓名 (去除 _0000.nii.gz)
        patient_name = filename.replace('_0000.nii.gz', '')
        
        img_path = os.path.join(RAW_IMAGES_DIR, filename)
        
        # 只读取文件头信息，不加载整个庞大的矩阵，速度极快！
        reader = sitk.ImageFileReader()
        reader.SetFileName(img_path)
        reader.ReadImageInformation()
        
        spacing = reader.GetSpacing() # 物理分辨率 (x, y, z) 单位: mm
        size = reader.GetSize()       # 像素矩阵大小 (x, y, z)
        
        results.append({
            "Patient_Name": patient_name,
            "Spacing_X": spacing[0],
            "Spacing_Y": spacing[1],
            "Spacing_Z": spacing[2], # 重点关注！这是层厚
            "Size_X": size[0],
            "Size_Y": size[1],
            "Size_Z": size[2]
        })

    # 保存为 CSV
    df = pd.DataFrame(results)
    # 按患者姓名排序
    df = df.sort_values(by='Patient_Name')
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')
    
    print(f"✅ 提取完成！分辨率信息已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()