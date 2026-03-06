import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from collections import OrderedDict
import vtk
from vtk.util import numpy_support
"""
    数据预处理脚本，将原始DICOM数据和stl数据转换为nnUNet_raw中的nii.gz
    stl数据绝大多数都是和image不对齐的，经过本脚本以后，大多数label可以和image对齐
    但是还有46个样本经过本脚本以后label还是和stl对齐重叠，没有和image对齐
"""

# ================= 配置区域 =================
BASE_DIR = "/data1/xyh/data/software_analysis2"
CBCT_DIR = os.path.join(BASE_DIR, "CBCT")
STL_DIR = os.path.join(BASE_DIR, "stl") 

NNUNET_RAW = "/data1/xyh/data/nnUNet/nnUNet_raw"
TASK_ID = 502  
TASK_NAME = f"Dataset{TASK_ID}_AirwaySegmentation"

OUT_IMAGES_TR = os.path.join(NNUNET_RAW, TASK_NAME, "imagesTr")
OUT_LABELS_TR = os.path.join(NNUNET_RAW, TASK_NAME, "labelsTr")

LABEL_MAP = {"background": 0, "airway": 1}
# ===========================================

def vtk_stl_to_mask(ref_img, stl_path):
    """最终修复版：使用 vtkImageStencilToImage 明确正向填充逻辑"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    polydata = reader.GetOutput()

    origin = ref_img.GetOrigin()
    spacing = ref_img.GetSpacing()
    dims = ref_img.GetSize()

    # 1. 生成 Stencil (STL 占据的空间形状)
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    # 必须设置 Extent 以匹配参考图像
    pol2stenc.SetOutputWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
    pol2stenc.Update()

    # 2. 核心修复：直接将 Stencil 转换为二值图像
    # SetInsideValue(1) 确保 STL 内部为 1，OutsideValue(0) 确保外部为 0
    stencil_to_mask = vtk.vtkImageStencilToImage()
    stencil_to_mask.SetInputConnection(pol2stenc.GetOutputPort())
    stencil_to_mask.SetOutsideValue(0) 
    stencil_to_mask.SetInsideValue(1)  
    stencil_to_mask.SetOutputScalarTypeToUnsignedChar()
    stencil_to_mask.Update()

    # 3. 转回 Numpy
    out_vtk_image = stencil_to_mask.GetOutput()
    sc = out_vtk_image.GetPointData().GetScalars()
    mask_np = numpy_support.vtk_to_numpy(sc)
    
    return mask_np.reshape(dims[::-1])

def process_data():
    # 自动创建目录
    os.makedirs(OUT_IMAGES_TR, exist_ok=True)
    os.makedirs(OUT_LABELS_TR, exist_ok=True)
    
    all_stls = sorted([f for f in os.listdir(STL_DIR) if f.lower().endswith('.stl')])
    print(f"开始执行修正后的 VTK 转换... 总计: {len(all_stls)} 例")

    for stl_file in tqdm(all_stls):
        case_name = os.path.splitext(stl_file)[0]
        patient_cbct_path = os.path.join(CBCT_DIR, case_name)
        stl_path = os.path.join(STL_DIR, stl_file)
        
        if not os.path.isdir(patient_cbct_path):
            continue
            
        try:
            # 读取 DICOM 序列
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(patient_cbct_path)
            reader.SetFileNames(dicom_names)
            reader.GlobalWarningDisplayOff() # 屏蔽采样不均警告
            image_sitk = reader.Execute()
            
            # 转换 STL
            mask_np = vtk_stl_to_mask(image_sitk, stl_path)
            
            # 保存原图 (_0000.nii.gz)
            sitk.WriteImage(image_sitk, os.path.join(OUT_IMAGES_TR, f"{case_name}_0000.nii.gz"))
            
            # 保存标签并强制复制元数据（确保 Slicer 对齐）
            label_sitk = sitk.GetImageFromArray(mask_np)
            label_sitk.CopyInformation(image_sitk) 
            sitk.WriteImage(label_sitk, os.path.join(OUT_LABELS_TR, f"{case_name}.nii.gz"))
            
        except Exception as e:
            print(f"\n处理 {case_name} 时出错: {e}")

    # 生成符合 v2 标准的 dataset.json
    d_json = OrderedDict({
        "channel_names": {"0": "CT"},
        "labels": LABEL_MAP,
        "numTraining": len(all_stls),
        "file_ending": ".nii.gz"
    })
    with open(os.path.join(NNUNET_RAW, TASK_NAME, "dataset.json"), 'w') as f:
        json.dump(d_json, f, indent=4)
    print("所有数据转换完成。")

if __name__ == "__main__":
    process_data()