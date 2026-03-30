import os
import json
import re
import numpy as np
import SimpleITK as sitk
import trimesh
from collections import OrderedDict
"""
    保留患者姓名不使用数字来命名
    正则精准匹配文件名
    不能保证每次运行得到的顺序一样
"""

# ================= 配置区域 =================
BASE_DIR = "/data1/xyh/data/software_analysis2"
CBCT_DIR = os.path.join(BASE_DIR, "CBCT")
DIAN_DIR = os.path.join(BASE_DIR, "DIAN")

NNUNET_RAW = "/data1/xyh/data/nnUNet/nnUNet_raw"
TASK_ID = 511
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

# 大靶心策略
SPHERE_RADIUS_MM = 6.0  

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

def is_landmark_in_filename(landmark, filename):
    """【修复核心】精准正则匹配，免疫 Hermans 等子串陷阱"""
    pattern = re.compile(rf"(^|[^a-zA-Z]){landmark}([^a-zA-Z]|$)", re.IGNORECASE)
    return bool(pattern.search(filename))

def draw_sphere(mask_array, center_idx, radius_px, label_id, shape):
    r = max(1, int(radius_px))
    z_min = max(0, int(center_idx[2] - r - 1))
    z_max = min(shape[0], int(center_idx[2] + r + 2))
    y_min = max(0, int(center_idx[1] - r - 1))
    y_max = min(shape[1], int(center_idx[1] + r + 2))
    x_min = max(0, int(center_idx[0] - r - 1))
    x_max = min(shape[2], int(center_idx[0] + r + 2))

    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    dist_sq = (x - center_idx[0])**2 + (y - center_idx[1])**2 + (z - center_idx[2])**2
    
    mask_roi = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_roi[dist_sq <= r**2] = label_id

def generate_dataset_json(num_training):
    json_dict = OrderedDict()
    json_dict['name'] = TASK_NAME
    json_dict['description'] = "Airway Landmark Detection"
    json_dict['reference'] = "Private Data"
    json_dict['licence'] = "Private"
    json_dict['release'] = "0.0"
    json_dict['tensorImageSize'] = "3D"
    json_dict['channel_names'] = {"0": "CBCT"} 
    
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
    patient_folders = [p for p in os.listdir(CBCT_DIR) if os.path.isdir(os.path.join(CBCT_DIR, p))]
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
            
        label_np = np.zeros(image_sitk.GetSize()[::-1], dtype=np.uint8)
        
        spacing = image_sitk.GetSpacing() 
        avg_spacing = np.mean(spacing) 
        radius_pixels = SPHERE_RADIUS_MM / avg_spacing
        
        if os.path.exists(patient_dian_path):
            # 【修复 1】解决大小写后缀问题
            all_files = os.listdir(patient_dian_path)
            stl_files = [f for f in all_files if f.lower().endswith('.stl')]
            
            # 记录当前病例已经画过的关键点，防止同一个点有多个重复文件导致反复覆盖
            processed_labels = set()
            
            for stl_file in stl_files:
                current_label_id = 0
                
                # 【修复 2】使用正则引擎匹配，免疫子串陷阱
                for key, val in LANDMARK_MAP.items():
                    if is_landmark_in_filename(key, stl_file):
                        current_label_id = val
                        break
                
                # 如果没匹配到，或者这个点已经画过了，直接跳过
                if current_label_id == 0 or current_label_id in processed_labels:
                    continue 
                    
                processed_labels.add(current_label_id)
                
                stl_path = os.path.join(patient_dian_path, stl_file)
                world_centroid = get_centroid_from_stl(stl_path)
                
                if world_centroid is not None:
                    voxel_idx = image_sitk.TransformPhysicalPointToContinuousIndex(world_centroid)
                    draw_sphere(label_np, voxel_idx, radius_pixels, current_label_id, label_np.shape)
        
        # 【修改重点】不再使用 valid_cases 强行生成数字，直接使用原始病人名字
        case_identifier = patient_name
        
        out_image_name = f"{case_identifier}_0000.nii.gz"      
        sitk.WriteImage(image_sitk, os.path.join(OUT_IMAGES_TR, out_image_name))
        
        label_sitk = sitk.GetImageFromArray(label_np)
        label_sitk.CopyInformation(image_sitk)
        out_label_name = f"{case_identifier}.nii.gz"
        sitk.WriteImage(label_sitk, os.path.join(OUT_LABELS_TR, out_label_name))
        
        valid_cases += 1

    generate_dataset_json(valid_cases)
    print("Done! Data conversion complete.")

if __name__ == "__main__":
    process_data()