import os
import numpy as np
import trimesh
import pandas as pd
import warnings
"""
    计算validation的最小横截面积
"""

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 1. 之前脚本导出的平滑 Validation STL 根目录
VAL_STL_ROOT = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/comparison_stls/Validation"
# 2. 修正后的旋转参数文件
PARAMS_FILE = "rotation_params.txt"
# 3. 输出 CSV 名称
OUTPUT_CSV = "5fold_mcsa_pred_standardized_results.csv"
# ===========================================

def load_rotation_params(file_path):
    params = {}
    if not os.path.exists(file_path): return params
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) != 2: continue
            name, val = parts
            if val != "MISSING":
                data = list(map(float, val.split(',')))
                params[name] = (data[0], (data[1], data[2], data[3]))
    return params

def calculate_mesh_mcsa(stl_path, angle, center):
    """
    几何法计算 mCSA
    """
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)

        # 应用旋转变换 (与 GT 脚本完全一致)
        transform = trimesh.transformations.rotation_matrix(-angle, [1, 0, 0], center)
        mesh.apply_transform(transform)
        
        # 获取 Z 轴核心区间
        z_min, z_max = mesh.bounds[:, 2]
        z_start = z_min + (z_max - z_min) * 0.1
        z_end = z_min + (z_max - z_min) * 0.9
        
        z_levels = np.arange(z_start, z_end, 0.5)
        areas = []
        for z in z_levels:
            section_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if section_3d is not None:
                try:
                    section_2d, _ = section_3d.to_2D()
                    # 过滤掉逻辑上不可能的极小噪声面积
                    if section_2d.area > 5.0:
                        areas.append(section_2d.area)
                except: continue
        
        if not areas: return 0.0
        
        # 滑动平均平滑
        if len(areas) > 5:
            smoothed = np.convolve(areas, np.ones(5)/5, mode='valid')
        else:
            smoothed = areas
            
        return np.min(smoothed)
    except Exception:
        return 0.0

def main():
    rot_params = load_rotation_params(PARAMS_FILE)
    all_val_results = []

    print(f"🚀 开始批量计算 5-Fold 预测结果的标准化 mCSA...")

    for fold in range(5):
        fold_dir = os.path.join(VAL_STL_ROOT, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            print(f"⚠️ 跳过 Fold {fold}: 路径不存在")
            continue

        stl_files = sorted([f for f in os.listdir(fold_dir) if f.endswith('.stl')])
        count = 0
        for f_name in stl_files:
            case_name = f_name.replace('.stl', '')
            if case_name not in rot_params: continue
            
            angle, center = rot_params[case_name]
            stl_path = os.path.join(fold_dir, f_name)
            
            area_val = calculate_mesh_mcsa(stl_path, angle, center)
            
            if area_val > 0:
                all_val_results.append({
                    "Fold": f"Fold_{fold}",
                    "Case Name": case_name,
                    "Pred_mCSA_mm2": round(area_val, 2)
                })
            
            count += 1
            if count % 10 == 0:
                print(f"  [Fold {fold}] 已处理: {count}/{len(stl_files)} | 当前面积: {area_val:.2f}")

    # 保存结果
    df = pd.DataFrame(all_val_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 预测结果汇总完成！保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()