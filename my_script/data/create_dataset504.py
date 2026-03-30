import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
"""
    从501任务中筛选出同时包含1:AICV, 3:BEP, 5:TEE这三个点的样本，同时屏蔽剩下4个点的标签，设置为504任务
"""
# ================= 配置区域 =================
RAW_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw"
SOURCE_DATASET = "Dataset501_AirwayLandmarks"
TARGET_DATASET = "Dataset504_AirwayLandmarks"  # 改为全新的 504 任务

# 原标签中我们要死磕的三个点
TARGET_POINTS_IN_SRC = {"AICV": 1, "BEP": 3, "TEE": 5}

# 强行重映射为纯粹的 1, 2, 3 分类
NEW_MAPPING = {1: 1, 3: 2, 5: 3}
# ============================================

def main():
    src_dir = os.path.join(RAW_DIR, SOURCE_DATASET)
    dst_dir = os.path.join(RAW_DIR, TARGET_DATASET)

    src_images_dir = os.path.join(src_dir, "imagesTr")
    src_labels_dir = os.path.join(src_dir, "labelsTr")
    
    dst_images_dir = os.path.join(dst_dir, "imagesTr")
    dst_labels_dir = os.path.join(dst_dir, "labelsTr")

    # 创建 504 目录
    if os.path.exists(dst_dir):
        print(f"🧹 发现已存在的 {TARGET_DATASET}，正在清理...")
        shutil.rmtree(dst_dir)
        
    for d in [dst_images_dir, dst_labels_dir]:
        os.makedirs(d, exist_ok=True)

    label_files = [f for f in os.listdir(src_labels_dir) if f.endswith('.nii.gz')]
    total_files = len(label_files)
    valid_cases_count = 0

    print(f"🔍 开始洗盘：提取并重映射纯净的 3 分类数据到 504 任务...\n")

    for idx, label_file in enumerate(label_files, 1):
        print(f"[{idx:03d}/{total_files}] 处理 {label_file:<20} ... ", end="", flush=True)
        
        label_path = os.path.join(src_labels_dir, label_file)
        img = sitk.ReadImage(label_path)
        arr = sitk.GetArrayFromImage(img)
        unique_vals = np.unique(arr)

        # 检查是否同时包含 1, 3, 5
        if all(src_id in unique_vals for src_id in TARGET_POINTS_IN_SRC.values()):
            valid_cases_count += 1
            print(f"🌟 命中！(提取并重映射为 1,2,3...)")
            
            # 创建一个纯黑的背景
            new_arr = np.zeros_like(arr)
            
            # 只把 1, 3, 5 画上去，变成 1, 2, 3
            for src_id, new_id in NEW_MAPPING.items():
                new_arr[arr == src_id] = new_id
                
            # 保存纯净的 3分类标签
            new_img = sitk.GetImageFromArray(new_arr)
            new_img.CopyInformation(img)
            sitk.WriteImage(new_img, os.path.join(dst_labels_dir, label_file))
            
            # 复制图像
            base_name = label_file.replace('.nii.gz', '')
            image_file = f"{base_name}_0000.nii.gz"
            shutil.copy(os.path.join(src_images_dir, image_file), os.path.join(dst_images_dir, image_file))
        else:
            print("跳过")

    print("-" * 60)
    print(f"🎉 彻底洗盘完成！共提取 {valid_cases_count} 个纯净 3分类病例到 Dataset504。")

    # 生成全新的 3分类 dataset.json
    dataset_info = {
        "name": TARGET_DATASET,
        "description": "Pure Specialist for AICV, BEP, TEE",
        "reference": "Private Data",
        "licence": "Private",
        "release": "0.0",
        "tensorImageSize": "3D",
        "channel_names": {"0": "CBCT"},
        "labels": {
            "background": 0,
            "AICV": 1,
            "BEP": 2,
            "TEE": 3
        },
        "numTraining": valid_cases_count,
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(dst_dir, "dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)

if __name__ == "__main__":
    main()