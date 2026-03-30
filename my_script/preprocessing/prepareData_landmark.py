import os
import json
import numpy as np
import SimpleITK as sitk
import trimesh
from collections import OrderedDict
"""
    数据预处理脚本，不能保证每次运行得到的顺序一样
    使用数字来重新命名
"""

# ================= 配置区域 =================
# 建议路径使用正斜杠 / 避免转义问题
BASE_DIR = "/data1/xyh/data/software_analysis2"
CBCT_DIR = os.path.join(BASE_DIR, "CBCT")
DIAN_DIR = os.path.join(BASE_DIR, "DIAN")

NNUNET_RAW = "/data1/xyh/data/nnUNet/nnUNet_raw"
TASK_ID = 509
TASK_NAME = f"Dataset{TASK_ID}_AirwayLandmarks"

OUT_IMAGES_TR = os.path.join(NNUNET_RAW, TASK_NAME, "imagesTr")
OUT_LABELS_TR = os.path.join(NNUNET_RAW, TASK_NAME, "labelsTr")

LANDMARK_MAP = {
    "AICV": 1,
    "ANS": 2,
    "BEP": 3,
    "PNS": 4,
    "TEE": 5,
    "TEP": 6,
    "TUV": 7
}

# 定义物理半径 (毫米)，保证所有人的标签球在物理空间一样大
# SPHERE_RADIUS_MM = 3.0 
SPHERE_RADIUS_MM = 6.0  # 将物理半径提高为6
# SPHERE_RADIUS_MM = 8.0  # 将物理半径提高为6

# ===========================================

def create_folder_structure():
    os.makedirs(OUT_IMAGES_TR, exist_ok=True)
    os.makedirs(OUT_LABELS_TR, exist_ok=True)

def read_dicom_series(folder_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def get_centroid_from_stl(stl_path):
    try:
        mesh = trimesh.load_mesh(stl_path)
        return mesh.centroid 
    except Exception as e:
        print(f"Warning: Failed to load STL {stl_path}: {e}")
        return None

def draw_sphere(mask_array, center_idx, radius_px, label_id, shape):
    """
    radius_px: 像素单位的半径
    """
    # 【修复处】这里之前写成了 radius_pixels，应改为 radius_px
    r = max(1, int(radius_px))
    
    # 简单的包围盒优化，避免全图计算距离，加速处理
    z_min = max(0, int(center_idx[2] - r - 1))
    z_max = min(shape[0], int(center_idx[2] + r + 2))
    y_min = max(0, int(center_idx[1] - r - 1))
    y_max = min(shape[1], int(center_idx[1] + r + 2))
    x_min = max(0, int(center_idx[0] - r - 1))
    x_max = min(shape[2], int(center_idx[0] + r + 2))

    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # 注意：SimpleITK 的 center_idx 是 (x, y, z)，而这里 numpy 网格也是对应 x, y, z
    dist_sq = (x - center_idx[0])**2 + (y - center_idx[1])**2 + (z - center_idx[2])**2
    
    # 提取局部区域进行赋值
    mask_roi = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_roi[dist_sq <= r**2] = label_id

def generate_dataset_json(num_training):
    """自动生成 dataset.json"""
    json_dict = OrderedDict()
    json_dict['name'] = TASK_NAME
    json_dict['description'] = "Airway Landmark Detection"
    json_dict['reference'] = "Private Data"
    json_dict['licence'] = "Private"
    json_dict['release'] = "0.0"
    json_dict['tensorImageSize'] = "3D"
    
    # 【修复点】将 modality 改为 channel_names
    # nnU-Net V2 新标准：使用 channel_names 替代 modality
    json_dict['channel_names'] = {"0": "CBCT"} 
    
    # 这里的 labels 需要包含 'background': 0
    labels = {"background": 0}
    for k, v in LANDMARK_MAP.items():
        labels[k] = v
    json_dict['labels'] = labels
    
    json_dict['numTraining'] = num_training
    json_dict['file_ending'] = ".nii.gz"
    
    json_path = os.path.join(NNUNET_RAW, TASK_NAME, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
    print(f"Generated dataset.json at {json_path}")

def process_data():
    create_folder_structure()
    patient_folders = os.listdir(CBCT_DIR)
    
    # 过滤掉非文件夹项
    patient_folders = [p for p in patient_folders if os.path.isdir(os.path.join(CBCT_DIR, p))]
    
    valid_cases = 0
    
    for i, patient_name in enumerate(patient_folders):
        patient_cbct_path = os.path.join(CBCT_DIR, patient_name)
        patient_dian_path = os.path.join(DIAN_DIR, patient_name)
        
        print(f"Processing case {i+1}/{len(patient_folders)}: {patient_name}")
        
        try:
            image_sitk = read_dicom_series(patient_cbct_path)
        except Exception as e:
            print(f"  Error reading DICOM: {e}")
            continue
            
        # Image 数组顺序 (z, y, x)
        label_np = np.zeros(image_sitk.GetSize()[::-1], dtype=np.uint8)
        
        # 根据 Spacing 动态计算像素半径
        # 获取 Spacing (x_mm, y_mm, z_mm)
        spacing = image_sitk.GetSpacing() 
        # 使用最小 spacing 还是平均 spacing 都可以，这里取平均值来估算
        avg_spacing = np.mean(spacing) 
        radius_pixels = SPHERE_RADIUS_MM / avg_spacing
        
        has_landmark = False

        if os.path.exists(patient_dian_path):
            stl_files = [f for f in os.listdir(patient_dian_path) if f.endswith('.stl')]
            
            for stl_file in stl_files:
                current_label_id = 0
                lower_name = stl_file.lower()
                
                # 修复大小写匹配逻辑
                for key, val in LANDMARK_MAP.items():
                    # 将 key 也转为小写进行比较
                    if key.lower() in lower_name:
                        current_label_id = val
                        break
                
                if current_label_id == 0:
                    continue # 跳过未定义的 STL
                
                stl_path = os.path.join(patient_dian_path, stl_file)
                world_centroid = get_centroid_from_stl(stl_path)
                
                if world_centroid is not None:
                    voxel_idx = image_sitk.TransformPhysicalPointToContinuousIndex(world_centroid)
                    # 这里调用时传入的是 radius_pixels (变量)，传给函数参数 radius_px
                    draw_sphere(label_np, voxel_idx, radius_pixels, current_label_id, label_np.shape)
                    has_landmark = True
        
        # 保存文件
        case_identifier = f"Case_{valid_cases:03d}"
        
        out_image_name = f"{case_identifier}_0000.nii.gz"       # _0000 表示该病例第0哥输入通道，nnUNet的强制要求，为了统一支持多模态输入
        sitk.WriteImage(image_sitk, os.path.join(OUT_IMAGES_TR, out_image_name))
        
        label_sitk = sitk.GetImageFromArray(label_np)
        label_sitk.CopyInformation(image_sitk)
        out_label_name = f"{case_identifier}.nii.gz"
        sitk.WriteImage(label_sitk, os.path.join(OUT_LABELS_TR, out_label_name))
        
        valid_cases += 1

    # 最后生成 dataset.json
    generate_dataset_json(valid_cases)
    print("Done! Data conversion complete.")

if __name__ == "__main__":
    process_data()