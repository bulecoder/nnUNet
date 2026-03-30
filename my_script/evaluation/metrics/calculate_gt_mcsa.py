import os
import numpy as np
import trimesh
import pandas as pd
import warnings
"""
    计算gt的最小横截面积
"""

# 1. 彻底屏蔽所有过时警告和 Shapely 内部警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 指向你刚才通过 Label (nii.gz) 转换得到的平滑 STL 路径
STL_DIR = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/comparison_stls/GT"
# 必须使用最新修正 theta 逻辑后生成的参数文件
PARAMS_FILE = "rotation_params.txt" 
OUTPUT_CSV = "5fold_mcsa_gt_standardized_results.csv"
# ===========================================

def load_rotation_params(file_path):
    params = {}
    if not os.path.exists(file_path):
        return params
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) != 2: continue
            name, val = parts
            if val == "MISSING":
                params[name] = "MISSING"
            else:
                data = list(map(float, val.split(',')))
                # theta, (cx, cy, cz)
                params[name] = (data[0], (data[1], data[2], data[3]))
    return params

def calculate_mesh_mcsa(stl_path, angle, center):
    """
    通过 STL 几何截面计算标准化 mCSA (修正旋转逻辑版)
    """
    try:
        # 加载网格
        mesh = trimesh.load(stl_path, force='mesh')
        
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)

        # 对网格应用旋转校正
        # 核心：使用 -angle 来抵消原始 CBCT 中的倾斜，使 ANS-PNS 连线水平
        # 绕 X 轴旋转 (矢状位校正)
        transform = trimesh.transformations.rotation_matrix(-angle, [1, 0, 0], center)
        mesh.apply_transform(transform)
        
        # 获取 Z 轴核心范围 (15% - 85%)
        # 旋转后，Z 轴应代表患者的 Superior-Inferior 方向
        z_min, z_max = mesh.bounds[:, 2]
        z_start = z_min + (z_max - z_min) * 0.1
        z_end = z_min + (z_max - z_min) * 0.9
        
        # 步长 0.5mm 扫描
        z_levels = np.arange(z_start, z_end, 0.5)
        areas = []

        for z in z_levels:
            # 这里的 plane_normal=[0, 0, 1] 确保我们切的是轴状面 (Axial Section)
            section_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            
            if section_3d is not None:
                try:
                    # 将 3D 交线投射到 2D 平面计算面积
                    section_2d, _ = section_3d.to_2D()
                    if section_2d.area > 5.0: # 过滤掉小于 5mm^2 的噪声面积
                        areas.append(section_2d.area)
                except:
                    continue
        
        if not areas:
            return 0.0
            
        # 5层物理滑动平均 (约 2.5mm 长度)
        if len(areas) > 5:
            smoothed_areas = np.convolve(areas, np.ones(5)/5, mode='valid')
        else:
            smoothed_areas = areas
            
        # 返回平滑后的最小值
        return np.min(smoothed_areas)
        
    except Exception as e:
        return 0.0

def main():
    if not os.path.exists(STL_DIR):
        print(f"Error: Directory not found: {STL_DIR}")
        return

    rot_params = load_rotation_params(PARAMS_FILE)
    if not rot_params:
        print(f"Error: Params file is empty or missing: {PARAMS_FILE}")
        return

    results = []
    print(f"🚀 开始计算基于 Label-STL 的标准化 mCSA...")
    
    # 获取目录下的 STL 文件
    stl_files_map = {os.path.splitext(f)[0]: f for f in os.listdir(STL_DIR) if f.lower().endswith('.stl')}
    
    cases = list(rot_params.items())
    total = len(cases)
    count = 0

    for case_name, params in cases:
        count += 1
        if params == "MISSING" or case_name not in stl_files_map:
            continue
            
        stl_path = os.path.join(STL_DIR, stl_files_map[case_name])
        angle, center = params
        
        area = calculate_mesh_mcsa(stl_path, angle, center)
        
        if area > 0:
            results.append({"Case Name": case_name, "GT_mCSA_mm2": round(area, 2)})
        
        if count % 10 == 0:
            print(f"进度: [{count}/{total}] | 当前: {case_name[:15]} | 面积: {area:.2f}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ STL mCSA (GT) 计算完成！")
        print(f"平均面积: {df['GT_mCSA_mm2'].mean():.2f} mm^2")
    else:
        print("\n❌ 失败: 未能获取有效结果，请检查文件映射。")

if __name__ == "__main__":
    main()