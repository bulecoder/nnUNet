import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
"""
    计算目标目录下 上气道的容积（使用nii.gz格式进行计算,stl本身有一定误差，还需要使用特殊公式进行计算，nii.gz可以直接使用体素计算）
"""

TASK_ID = 510

# ================= 配置区域 =================
# 1. 设置 5 个 Fold 预测结果的文件夹路径 (请确保路径与你实际的 LCC 后处理路径一致)
FOLD_PATHS = {
    "Fold 0": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation_postprocessed_LCC",
    "Fold 1": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation_postprocessed_LCC",
    "Fold 2": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation_postprocessed_LCC",
    "Fold 3": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation_postprocessed_LCC",
    "Fold 4": f"/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation_postprocessed_LCC",
}

# 2. 最终汇总结果保存路径
OUTPUT_CSV_PATH = "/data1/xyh/projects/nnUNet/my_script/evaluation/metrics/1.volumes.csv"
# ===========================================

def calculate_volume(nifti_path):
    """
    读取 nii.gz 并计算气道容积
    返回容积 (mm³) 和 容积 (cm³)
    """
    try:
        img = sitk.ReadImage(nifti_path)
        spacing = img.GetSpacing()  # (dx, dy, dz)
        
        # 计算单个体素的物理体积 (mm³)
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        
        # 获取 numpy 阵列并计算前景体素数量
        arr = sitk.GetArrayFromImage(img)
        voxel_count = np.sum(arr > 0)
        
        # 计算总体积
        total_volume_mm3 = voxel_count * voxel_volume_mm3
        total_volume_cm3 = total_volume_mm3 / 1000.0  # 转换为 cm³ (mL)
        
        return voxel_count, total_volume_mm3, total_volume_cm3
    except Exception as e:
        print(f"❌ 读取 {nifti_path} 出错: {e}")
        return None, None, None

def main():
    results = []
    total_processed = 0

    print("🚀 开始依次计算 5 个 Fold 的上气道容积...")

    # 遍历每个 Fold 的路径
    for fold_name, pred_dir in FOLD_PATHS.items():
        if not os.path.exists(pred_dir):
            print(f"⚠️ 找不到目录，跳过: {pred_dir}")
            continue

        nii_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
        if not nii_files:
            continue

        print(f"\n📂 正在处理 {fold_name} ({len(nii_files)} 个样本)...")
        
        # 使用 tqdm 显示当前 Fold 的进度
        for f_name in tqdm(nii_files, desc=fold_name):
            file_path = os.path.join(pred_dir, f_name)
            case_id = f_name.replace('.nii.gz', '')
            
            vox_count, vol_mm3, vol_cm3 = calculate_volume(file_path)
            
            if vox_count is not None:
                results.append({
                    "Fold": fold_name,           # 新增列：标记数据所属的 Fold
                    "Case_ID": case_id,
                    "Voxel_Count": vox_count,
                    "Volume_mm3": round(vol_mm3, 2),
                    "Volume_cm3_mL": round(vol_cm3, 4)
                })
                total_processed += 1

    if not results:
        print("❌ 未能提取到任何有效数据，请检查路径。")
        return

    # 转换为 DataFrame 并保存为单个大 CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    
    # 打印统计信息
    mean_vol = df['Volume_cm3_mL'].mean()
    std_vol = df['Volume_cm3_mL'].std()
    
    print("\n" + "="*50)
    print("✅ 5-Fold 容积计算全部完成！")
    print(f"📦 共计处理样本: {total_processed} 例")
    print(f"📊 全局数据概览: 平均容积 = {mean_vol:.2f} ± {std_vol:.2f} cm³ (mL)")
    print(f"💾 汇总结果已成功保存至:\n   {OUTPUT_CSV_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()