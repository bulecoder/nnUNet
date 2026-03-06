import SimpleITK as sitk
import numpy as np
"""
    检查label占前景的比例有多大来判断是否处理正确
"""

# 请替换为你刚才在 Slicer 中查看的那个文件路径
# file_path = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset502_AirwaySegmentation/labelsTr/A.C._Roos_1979_11_18.nii.gz"
file_path = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset502_AirwaySegmentation/labelsTr/ZXM 22.nii.gz"


img = sitk.ReadImage(file_path)
data = sitk.GetArrayFromImage(img)

total_pixels = data.size
foreground_pixels = np.sum(data == 1)
background_pixels = np.sum(data == 0)

print(f"文件名: {file_path}")
print(f"总像素数: {total_pixels}")
print(f"前景 (Label 1) 像素数: {foreground_pixels}")
print(f"背景 (Label 0) 像素数: {background_pixels}")
print(f"占比: {(foreground_pixels/total_pixels)*100:.2f}%")

if foreground_pixels > background_pixels:
    print("\n❌ 确认错误：标签已反转！背景变成了 1，气道变成了 0。")
else:
    print("\n✅ 标签逻辑正常（如果 Slicer 看起来还是方块，请检查 3D 显示设置）。")