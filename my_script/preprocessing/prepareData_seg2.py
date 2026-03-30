import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
"""
    对于prepareData_seg.py中的数据预处理脚本，将几乎所有的stl转换为nii.gz数据的时候，能够将所有的位置对齐
    MALOUIN_JACQUELINE_1932_11_10 这个样本反而会产生偏差，单独写一个prepareData_seg2.py脚本来处理这个样本
    结果：使用该脚本单独处理这个样本，可以将stl转换为nii.gz但不产生位移
"""

# ================= 1. 输入配置（替换为你的出错样本路径） =================
# 该样本的 CBCT DICOM 文件夹路径
TEST_CBCT_DIR = "/data1/xyh/data/software_analysis2/CBCT/MALOUIN_JACQUELINE_1932_11_10" 
# 该样本的 STL 文件路径
TEST_STL_PATH = "/data1/xyh/data/software_analysis2/stl/MALOUIN_JACQUELINE_1932_11_10.stl"

# 测试输出目录
TEST_OUT_DIR = "/data1/xyh/data/test_alignment"
os.makedirs(TEST_OUT_DIR, exist_ok=True)

OUT_IMG_PATH = os.path.join(TEST_OUT_DIR, "test_image_0000.nii.gz")
OUT_LBL_PATH = os.path.join(TEST_OUT_DIR, "test_label.nii.gz")
# =====================================================================

def vtk_stl_to_mask_corrected(ref_img, stl_path):
    """
    修复版 VTK 转换函数：考虑 DICOM 的 Direction 矩阵
    """
    origin = ref_img.GetOrigin()
    spacing = ref_img.GetSpacing()
    dims = ref_img.GetSize()
    direction = ref_img.GetDirection() 
    
    # 打印方向矩阵，看看它是不是非单位矩阵
    print(f"[*] 原始图像 Origin: {origin}")
    print(f"[*] 原始图像 Spacing: {spacing}")
    print(f"[*] 原始图像 Direction (3x3): \n{np.array(direction).reshape(3,3)}")

    # 1. 读取 STL
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    polydata = reader.GetOutput()

    # 2. 核心数学修复：将 STL 从真实物理空间逆变换到 VTK 的轴对齐局部空间
    R_lps = np.array(direction).reshape((3, 3))
    R_lps_inv = R_lps.T # 正交矩阵的逆等于转置

    vtk_matrix = vtk.vtkMatrix4x4()
    vtk_matrix.Identity()
    
    # 填充 3x3 旋转部分
    for row in range(3):
        for col in range(3):
            vtk_matrix.SetElement(row, col, R_lps_inv[row, col])

    # 填充平移部分: Translation = Origin - R_inv * Origin
    orig_np = np.array(origin)
    trans_np = orig_np - np.dot(R_lps_inv, orig_np)
    for row in range(3):
        vtk_matrix.SetElement(row, 3, trans_np[row])

    # 3. 应用变换到 STL
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    
    transform_pd = vtk.vtkTransformPolyDataFilter()
    transform_pd.SetInputData(polydata)
    transform_pd.SetTransform(transform)
    transform_pd.Update()
    transformed_polydata = transform_pd.GetOutput()

    # 4. 生成 Stencil 和 Mask
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(transformed_polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
    pol2stenc.Update()

    stencil_to_mask = vtk.vtkImageStencilToImage()
    stencil_to_mask.SetInputConnection(pol2stenc.GetOutputPort())
    stencil_to_mask.SetOutsideValue(0) 
    stencil_to_mask.SetInsideValue(1)  
    stencil_to_mask.SetOutputScalarTypeToUnsignedChar()
    stencil_to_mask.Update()

    # 5. 转回 Numpy
    out_vtk_image = stencil_to_mask.GetOutput()
    sc = out_vtk_image.GetPointData().GetScalars()
    mask_np = numpy_support.vtk_to_numpy(sc)
    
    return mask_np.reshape(dims[::-1])

def test_single_case():
    if not os.path.exists(TEST_CBCT_DIR) or not os.path.exists(TEST_STL_PATH):
        print("❌ 错误: 找不到指定的 CBCT 文件夹或 STL 文件，请检查路径。")
        return

    print("🚀 开始处理单例数据...")
    
    # 1. 读取 DICOM
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(TEST_CBCT_DIR)
    reader.SetFileNames(dicom_names)
    reader.GlobalWarningDisplayOff()
    image_sitk = reader.Execute()
    
    # 2. 转换 STL
    print(f"[*] 正在转换 STL 并应用 Direction 修复...")
    mask_np = vtk_stl_to_mask_corrected(image_sitk, TEST_STL_PATH)
    
    # 3. 保存图像和标签
    sitk.WriteImage(image_sitk, OUT_IMG_PATH)
    print(f"✅ 原图已保存至: {OUT_IMG_PATH}")
    
    label_sitk = sitk.GetImageFromArray(mask_np)
    # 关键：将修正后的标签重新贴上原图的元数据标签
    label_sitk.CopyInformation(image_sitk) 
    sitk.WriteImage(label_sitk, OUT_LBL_PATH)
    print(f"✅ 修正后的标签已保存至: {OUT_LBL_PATH}")
    print("\n🎉 处理完成！请在 3D Slicer 中同时打开这两个 nii.gz 文件检查是否对齐。")

if __name__ == "__main__":
    test_single_case()