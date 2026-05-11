import os
import re
import json
import numpy as np
import SimpleITK as sitk
import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
"""
    在服务器上面处理原始数据太慢了（机械硬盘打开DICOM很为小图像，速度很慢，进度经常卡在IO上面），在本地进行数据预处理，处理完成以后直接上传到服务器中
"""

# ===================== 【用户配置区域】 =====================
RAW_DATA_ROOT = "/data1/xyh/data/SA-LSTM/raw_data"  
OUTPUT_ROOT = "/data1/xyh/data/nnUNet/nnUNet_raw"

# nnUNetv2 数据集配置
DATASET_ID = 601 # 建议使用 501-999 之间的自定义数据集 ID
DATASET_NAME = "CBCT_Landmark"
SPHERE_RADIUS_MM = 6.0          # 可视化的时候，半径为6太大了（半径为3比较合适），会有几个点重合，但是为了定点结果更好这里使用半径为6

GLOBAL_LABEL_MAP = {
    "M5": 1,
    "M5L": 2,
    "M5R": 3,
    "M7": 4,
    "M7R": 5,
    "M7L": 6,
    "前鼻棘": 7,
    "原点": 8,
    "右颧点": 9,
    "后鼻棘": 10,
    "左颧点": 11,
    "鼻根点": 12
}
# ========================================================================

def create_nnunet_folders():
    """创建 nnUNetv2 标准目录结构"""
    dataset_dirname = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    base_dir = os.path.join(OUTPUT_ROOT, dataset_dirname)
    
    image_dir = os.path.join(base_dir, "imagesTr")
    label_dir = os.path.join(base_dir, "labelsTr")
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    return base_dir, image_dir, label_dir

def generate_dataset_json(base_dir, num_training):
    """自动生成 nnUNetv2 所需的 dataset.json"""
    # nnUNetv2 要求 background 必须为 0
    labels_dict = {"background": 0}
    # 将你的 label 字典合并进去
    for name, val in GLOBAL_LABEL_MAP.items():
        labels_dict[name] = val
        
    dataset_info = {
        "channel_names": {
            "0": "CBCT" # 0000 对应的通道名称
        },
        "labels": labels_dict,
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    json_path = os.path.join(base_dir, "dataset.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    print(f"\n📄 dataset.json 已生成: {json_path}")

def get_clean_pinyin(text):
    """中文转拼音，保留数字和字母，无后缀"""
    clean_text = re.sub(r'[\\/*?:"<>|]', "", text.strip())
    result = []
    for char in clean_text:
        if '\u4e00' <= char <= '\u9fff':
            pinyin = lazy_pinyin(char, style=Style.NORMAL)[0]
            result.append(pinyin)
        elif char.isalnum():
            result.append(char)
    return "".join(result)

def read_dicom_series(folder_path):
    """读取DICOM序列，保留原始空间信息"""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
        if not dicom_names:
            return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image
    except Exception as e:
        print(f"    [错误] DICOM读取失败: {str(e)}")
        return None

def parse_xml_landmarks(xml_path):
    """解析Materialise格式的XML标注"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        namespace = {'m': 'urn:materialise'}
        
        landmarks = {}
        for point_elem in root.findall('m:Point', namespace):
            name_elem = point_elem.find('m:Name', namespace)
            coord_elem = point_elem.find('m:Coordinate', namespace)
            
            if name_elem is not None and coord_elem is not None:
                name = name_elem.text.strip()
                coord_str = coord_elem.text.strip()
                x, y, z = map(float, coord_str.split())
                landmarks[name] = (x, y, z)
        return landmarks
    except Exception as e:
        print(f"    [错误] XML解析失败: {str(e)}")
        return None

def draw_sphere_on_mask(mask_array, center_idx, radius_px, label_value):
    """在mask上画球体"""
    r = max(1, int(radius_px))
    shape = mask_array.shape
    
    z_min = max(0, int(center_idx[2] - r - 1))
    z_max = min(shape[0], int(center_idx[2] + r + 2))
    y_min = max(0, int(center_idx[1] - r - 1))
    y_max = min(shape[1], int(center_idx[1] + r + 2))
    x_min = max(0, int(center_idx[0] - r - 1))
    x_max = min(shape[2], int(center_idx[0] + r + 2))

    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    dist_sq = (x - center_idx[0])**2 + (y - center_idx[1])**2 + (z - center_idx[2])**2
    
    mask_roi = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_roi[dist_sq <= r**2] = label_value 

def process_single_patient(patient_folder_name, image_dir, label_dir):
    """处理单个患者，返回(是否成功, 显示名, 错误信息, 存在的点, 拼音名)"""
    patient_folder_path = os.path.join(RAW_DATA_ROOT, patient_folder_name)
    
    # 1. 定义路径
    dicom_folder_path = os.path.join(patient_folder_path, patient_folder_name)
    xml_file_path = os.path.join(patient_folder_path, "标注点.xml")
    
    # 2. 检查文件是否存在
    if not os.path.isdir(dicom_folder_path):
        subfolders = [f for f in os.listdir(patient_folder_path) if os.path.isdir(os.path.join(patient_folder_path, f))]
        if len(subfolders) == 0:
            return False, patient_folder_name, "未找到DICOM文件夹", None, None
        else:
            dicom_folder_path = os.path.join(patient_folder_path, subfolders[0])
            
    if not os.path.exists(xml_file_path):
        return False, patient_folder_name, "未找到标注点.xml", None, None

    # 3. 读取数据
    image_sitk = read_dicom_series(dicom_folder_path)
    if image_sitk is None:
        return False, patient_folder_name, "DICOM读取失败", None, None

    landmarks = parse_xml_landmarks(xml_file_path)
    if landmarks is None:
        return False, patient_folder_name, "XML解析失败", None, None

    # 4. 生成Label
    try:
        label_np = np.zeros(image_sitk.GetSize()[::-1], dtype=np.uint8)
        spacing = image_sitk.GetSpacing()
        avg_spacing = np.mean(spacing)
        radius_px = SPHERE_RADIUS_MM / avg_spacing

        present_landmarks = {}
        for name, world_coord in landmarks.items():
            if name in GLOBAL_LABEL_MAP:
                current_label = GLOBAL_LABEL_MAP[name]
                voxel_idx = image_sitk.TransformPhysicalPointToContinuousIndex(world_coord)
                draw_sphere_on_mask(label_np, voxel_idx, radius_px, current_label)
                present_landmarks[name] = current_label
    except Exception as e:
        return False, patient_folder_name, f"Label生成失败: {str(e)}", None, None

    # 5. 保存文件 (nnUNetv2 命名规范)
    pinyin_name = get_clean_pinyin(patient_folder_name)
    display_name = f"{patient_folder_name} -> {pinyin_name}"
    
    # nnUNetv2 要求: 图像必须带模态后缀 _0000，标签不带
    out_image_path = os.path.join(image_dir, f"{pinyin_name}_0000.nii.gz")
    out_label_path = os.path.join(label_dir, f"{pinyin_name}.nii.gz")

    try:
        # 保存原始图像
        sitk.WriteImage(image_sitk, out_image_path)
        # 保存Label，严格对齐
        label_sitk = sitk.GetImageFromArray(label_np)
        label_sitk.CopyInformation(image_sitk)
        sitk.WriteImage(label_sitk, out_label_path)
    except Exception as e:
        return False, patient_folder_name, f"文件保存失败: {str(e)}", None, None

    return True, display_name, "", present_landmarks, pinyin_name

def main():
    print("="*80)
    print("CBCT批量处理工具 (适配 nnUNetv2 格式)")
    print("="*80)

    # 0. 确认配置
    print("\n📌 全局Label映射表:")
    for name, label_val in sorted(GLOBAL_LABEL_MAP.items(), key=lambda x: x[1]):
        print(f"  Label {label_val:2d} -> {name}")

    # 1. 创建 nnUNetv2 目录
    base_dir, image_dir, label_dir = create_nnunet_folders()
    print(f"\n📂 nnUNetv2 输出目录创建完毕:")
    print(f"  根目录 -> {base_dir}")
    print(f"  图像(imagesTr) -> {image_dir}")
    print(f"  标签(labelsTr) -> {label_dir}")

    # 2. 获取患者列表
    patient_folders = [f for f in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, f))]
    patient_folders = sorted(patient_folders)
    total_patients = len(patient_folders)
    print(f"\n📋 找到 {total_patients} 个患者样本")

    # 3. 批量处理
    success_count = 0
    fail_count = 0
    failed_cases = []
    all_records = []

    print("\n🚀 开始处理...")
    print("-"*80)

    for i, patient_folder in enumerate(tqdm(patient_folders, desc="处理进度")):
        is_success, display_name, error_msg, present_landmarks, pinyin_name = process_single_patient(patient_folder, image_dir, label_dir)
        
        if is_success:
            success_count += 1
            record = {
                "OriginalName": patient_folder,
                "CaseID": pinyin_name
            }
            for name in GLOBAL_LABEL_MAP.keys():
                record[name] = "✓" if (present_landmarks and name in present_landmarks) else "✗"
            all_records.append(record)
        else:
            fail_count += 1
            failed_cases.append((patient_folder, error_msg))
            print(f"\n  ❌ 失败: {patient_folder} - {error_msg}")

    # 4. 生成 dataset.json
    if success_count > 0:
        generate_dataset_json(base_dir, success_count)

    # 5. 总结与日志生成
    print("\n" + "="*80)
    print("🎉 nnUNetv2 数据集构建完成！")
    print("="*80)
    print(f"  总计处理: {total_patients}")
    print(f"  成功转换: {success_count}")
    print(f"  失败数量: {fail_count}")

    if fail_count > 0:
        print("\n❌ 失败病例列表:")
        for name, msg in failed_cases:
            print(f"  - {name}: {msg}")

    if len(all_records) > 0:
        csv_path = os.path.join(base_dir, "转换日志对照表.csv")
        fieldnames = ["OriginalName", "CaseID"] + list(GLOBAL_LABEL_MAP.keys())
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n📊 详细对照表已保存: {csv_path}")

if __name__ == "__main__":
    main()