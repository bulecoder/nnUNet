import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
# 从501任务中筛选出同时包含1:AICV, 3:BEP, 5:TEE这三个点的样本，但不屏蔽剩下4个点的标签

# ================= 配置区域 (请确认路径) =================
# nnUNet_raw 的根目录路径
RAW_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw"

# 源数据集和目标数据集名称
SOURCE_DATASET = "Dataset501_AirwayLandmarks"
TARGET_DATASET = "Dataset503_AirwayLandmarks"

# 需要完整包含的 Label 列表 (1 到 7)
REQUIRED_LABELS = [1, 3, 5]  # 1:AICV, 3:BEP, 5:TEE
# =========================================================

def main():
    src_dir = os.path.join(RAW_DIR, SOURCE_DATASET)
    dst_dir = os.path.join(RAW_DIR, TARGET_DATASET)

    src_images_dir = os.path.join(src_dir, "imagesTr")
    src_labels_dir = os.path.join(src_dir, "labelsTr")
    
    dst_images_dir = os.path.join(dst_dir, "imagesTr")
    dst_labels_dir = os.path.join(dst_dir, "labelsTr")

    # 1. 创建目标文件夹结构
    for d in [dst_images_dir, dst_labels_dir]:
        os.makedirs(d, exist_ok=True)
    
    print(f"✅ 已创建目标数据集文件夹: {dst_dir}")

    # 2. 遍历源标签，筛选出“完美的 132 例”
    label_files = [f for f in os.listdir(src_labels_dir) if f.endswith('.nii.gz')]
    total_files = len(label_files)
    
    perfect_cases_count = 0
    print(f"🔍 开始扫描 {total_files} 个病例标签，寻找包含全部 7 个点的完美病例...\n")
    print("-" * 60)

    for idx, label_file in enumerate(label_files, 1):
        print(f"[{idx:03d}/{total_files}] 正在检查 {label_file:<20} ... ", end="", flush=True)
        
        # 读取标签 NIfTI 寻找唯一的像素值
        label_path = os.path.join(src_labels_dir, label_file)
        img = sitk.ReadImage(label_path)
        arr = sitk.GetArrayFromImage(img)
        unique_vals = np.unique(arr)

        # 检查是否 1~7 全部都在 unique_vals 里面
        is_perfect = all(L in unique_vals for L in REQUIRED_LABELS)

        if is_perfect:
            perfect_cases_count += 1
            print("🌟 完美！(复制中...)")
            
            # 复制标签
            shutil.copy(label_path, os.path.join(dst_labels_dir, label_file))
            
            # 复制对应的图像 (注意 nnUNet v2 的图像文件名有 _0000 尾缀)
            # 例如 label 是 Case_160.nii.gz，image 就是 Case_160_0000.nii.gz
            base_name = label_file.replace('.nii.gz', '')
            image_file = f"{base_name}_0000.nii.gz"
            src_image_path = os.path.join(src_images_dir, image_file)
            
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, os.path.join(dst_images_dir, image_file))
            else:
                print(f"   ❌ 严重错误：找不到对应的图像文件 {image_file}！")
        else:
            print("跳过 (数据缺失)")

    print("-" * 60)
    print(f"🎉 扫描与复制完成！共筛选出 {perfect_cases_count} 个完美病例。")

    # 3. 生成新的 dataset.json
    src_json_path = os.path.join(src_dir, "dataset.json")
    dst_json_path = os.path.join(dst_dir, "dataset.json")

    if os.path.exists(src_json_path):
        with open(src_json_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        # 更新关键信息
        dataset_info['name'] = TARGET_DATASET
        dataset_info['numTraining'] = perfect_cases_count
        
        with open(dst_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=4)
        print(f"✅ 成功生成新的 dataset.json (numTraining: {perfect_cases_count})")
    else:
        print("⚠️ 找不到源 dataset.json，请手动复制并修改。")

    print("\n🚀 第 1 步与第 2 步已全部完成！现在你可以开始预处理 Dataset503 了。")

if __name__ == "__main__":
    main()