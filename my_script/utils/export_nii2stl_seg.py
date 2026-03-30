import os
import numpy as np
import SimpleITK as sitk
import trimesh
from skimage import measure
import warnings
"""
    将nii.gz格式的label和validation结果转换为stl格式且保存下来，便于后续相关指标的计算（例如最小横截面积）
"""

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 1. 金标准 (GT) 配置
GT_LABELS_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset506_AirwaySegmentation/labelsTr"
GT_EXPORT_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/comparison_stls/GT"

# 2. 预测结果 (Validation) 配置
RESULTS_ROOT = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres"
VAL_EXPORT_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/comparison_stls/Validation"

# 3. 修正参数 (虽然不修正坐标，但可用于筛选有效样本)
PARAMS_FILE = "rotation_params.txt"
# ===========================================

def load_valid_cases(file_path):
    """加载不缺失关键点的病例列表"""
    valid_cases = []
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2 and parts[1] != "MISSING":
                valid_cases.append(parts[0])
    return valid_cases

def voxel_to_smoothed_mesh_save(nii_path, save_path, smoothing_iterations=10):
    """
    将 NIfTI 掩膜转换为平滑的 STL (保持原始物理坐标)
    """
    try:
        img = sitk.ReadImage(nii_path)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = np.array(img.GetDirection()).reshape(3,3)
        data = sitk.GetArrayFromImage(img) > 0 
        
        if np.sum(data) == 0: return False

        # Marching Cubes 提取表面
        verts, faces, normals, values = measure.marching_cubes(data, level=0.5)
        
        # 物理坐标转换
        verts_physical = np.zeros_like(verts)
        for i in range(len(verts)):
            v_voxel = verts[i][::-1] 
            p = origin + np.dot(direction, v_voxel * spacing)
            verts_physical[i] = p

        mesh = trimesh.Trimesh(vertices=verts_physical, faces=faces)
        
        # 应用拉普拉斯平滑，消除体素块感 
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

        mesh.export(save_path)
        return True
    except Exception as e:
        print(f"\n [错误] 转换 {os.path.basename(nii_path)} 失败: {e}")
        return False

def main():
    # 加载有效病例列表，确保 GT 和 Val 对齐
    valid_cases = load_valid_cases(PARAMS_FILE)
    
    # --- 1. 处理金标准 (GT) ---
    print(f"🚀 开始转换金标准 (GT) 标签...")
    os.makedirs(GT_EXPORT_DIR, exist_ok=True)
    gt_files = [f for f in os.listdir(GT_LABELS_DIR) if f.endswith('.nii.gz')]
    gt_success = 0
    for idx, f_name in enumerate(sorted(gt_files)):
        case_name = f_name.replace('.nii.gz', '')
        if valid_cases and case_name not in valid_cases: continue
        
        save_path = os.path.join(GT_EXPORT_DIR, case_name + ".stl")
        if voxel_to_smoothed_mesh_save(os.path.join(GT_LABELS_DIR, f_name), save_path):
            gt_success += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  [GT] 进度: {idx + 1}/{len(gt_files)} | 成功: {gt_success}")

    # --- 2. 处理各折预测结果 (Validation) ---
    print(f"\n🚀 开始转换各折验证集 (Validation) 结果...")
    for fold in range(5):
        pred_dir = os.path.join(RESULTS_ROOT, f"fold_{fold}", "validation_postprocessed_LCC")
        if not os.path.exists(pred_dir):
            pred_dir = os.path.join(RESULTS_ROOT, f"fold_{fold}", "validation")
        
        if not os.path.exists(pred_dir): continue

        fold_export_dir = os.path.join(VAL_EXPORT_DIR, f"fold_{fold}")
        os.makedirs(fold_export_dir, exist_ok=True)

        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
        val_success = 0
        for idx, f_name in enumerate(pred_files):
            case_name = f_name.replace('.nii.gz', '')
            if valid_cases and case_name not in valid_cases: continue

            save_path = os.path.join(fold_export_dir, case_name + ".stl")
            if voxel_to_smoothed_mesh_save(os.path.join(pred_dir, f_name), save_path):
                val_success += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  [Fold {fold}] 进度: {idx + 1}/{len(pred_files)} | 成功: {val_success}")

    print(f"\n✅ 任务全部完成！")
    print(f"GT STL 路径: {GT_EXPORT_DIR}")
    print(f"Val STL 路径: {VAL_EXPORT_DIR}")

if __name__ == "__main__":
    main()