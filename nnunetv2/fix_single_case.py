import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

def fix_and_convert_single_case(dicom_dir, stl_path, out_img_path, out_lbl_path):
    # 1. 读取 DICOM
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image_sitk = reader.Execute()
    
    # 2. 自动定位图像中的气道质心 (作为对齐目标)
    img_arr = sitk.GetArrayFromImage(image_sitk)
    # 阈值选在 -500 以下以确保只选取空气
    air_mask = (img_arr < -500).astype(np.uint8)
    air_img = sitk.GetImageFromArray(air_mask)
    air_img.CopyInformation(image_sitk)
    
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(air_img)
    if not stats.HasLabel(1):
        print("❌ 错误：在图像中找不到明显的空气腔！")
        return
    target_centroid = np.array(stats.GetCentroid(1))
    
    # 3. 读取 STL 并计算其当前质心
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_path)
    stl_reader.Update()
    polydata = stl_reader.GetOutput()
    
    # 获取 STL 中心 (VTK 方式)
    center_filter = vtk.vtkCenterOfMass()
    center_filter.SetInputData(polydata)
    center_filter.SetUseScalarsAsWeights(False)
    center_filter.Update()
    current_centroid = np.array(center_filter.GetCenter())
    
    # 4. 执行物理平移 (修复核心)
    translation_vec = target_centroid - current_centroid
    print(f"检测到解剖偏移向量: {translation_vec}")
    
    transform = vtk.vtkTransform()
    transform.Translate(translation_vec)
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    aligned_polydata = transform_filter.GetOutput()

    # 5. 使用你之前的 VTK Stencil 逻辑生成 Mask
    origin = image_sitk.GetOrigin()
    spacing = image_sitk.GetSpacing()
    dims = image_sitk.GetSize()

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(aligned_polydata)
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

    # 6. 保存结果
    mask_np = numpy_support.vtk_to_numpy(stencil_to_mask.GetOutput().GetPointData().GetScalars())
    mask_np = mask_np.reshape(dims[::-1])
    
    label_sitk = sitk.GetImageFromArray(mask_np)
    label_sitk.CopyInformation(image_sitk)
    
    sitk.WriteImage(image_sitk, out_img_path)
    sitk.WriteImage(label_sitk, out_lbl_path)
    print(f"✅ 修复完成！请在 Slicer 中检查新的 Label。")

# --- 测试运行 ---
CASE = "COLEMAN_Felicity_1974_6_6" # 填入你名单中的一个
fix_and_convert_single_case(
    f"/data1/xyh/data/software_analysis2/CBCT/{CASE}",
    f"/data1/xyh/data/software_analysis2/stl/{CASE}.stl",
    f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset506_AirwaySegmentation/imagesTr/{CASE}_0000.nii.gz",
    f"/data1/xyh/data/nnUNet/nnUNet_raw/Dataset506_AirwaySegmentation/labelsTr/{CASE}.nii.gz"
)