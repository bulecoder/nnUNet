import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
"""
    计算目标目录下 ANS-PNS 的距离（单位mm）
"""

# ================= 配置区域 =================
# 1. 标签所在的文件夹列表 (List 格式)
# 💡 用法说明：
# - 如果是单路径 (比如真实标签 GT)，列表里只留一个路径即可。
# - 如果是 5-Fold，把 5 个预测文件夹的路径都贴进来。
LABEL_DIRS = [
    # 单路径示例：
    # "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset511_AirwayLandmarks/labelsTr"
    
    # 5-Fold 预测结果示例：
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation_postprocessed_LCC",
    "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation_postprocessed_LCC"
]

# 2. 结果保存的 CSV 路径 (自动保存在脚本同级目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "3.ans_pns_length.csv")

# 3. 目标点的 ID
ANS_ID = 2
PNS_ID = 4
# ============================================

def get_landmark_centroid(shape_filter, label_id):
    """从形状滤波器中安全地获取物理质心坐标 (x, y, z)"""
    if shape_filter.HasLabel(label_id):
        return np.array(shape_filter.GetCentroid(label_id))
    return None

def main():
    results = []
    
    # 实例化 C++ 底层的形状分析滤波器 (放在外层循环外以获得极致性能)
    shape_filter = sitk.LabelShapeStatisticsImageFilter()

    print(f"🚀 开始测量 ANS(ID:{ANS_ID}) 到 PNS(ID:{PNS_ID}) 的物理长度...")
    print("-" * 60)

    total_processed_files = 0

    # 遍历配置列表中的每一个目录
    for current_dir in LABEL_DIRS:
        if not os.path.exists(current_dir):
            print(f"⚠️ 找不到目录，已跳过 -> {current_dir}")
            continue

        label_files = [f for f in os.listdir(current_dir) if f.endswith('.nii.gz')]
        # 仅用于终端打印提示，不写入 CSV
        folder_name = os.path.basename(os.path.dirname(current_dir)) + "/" + os.path.basename(current_dir)
        
        if not label_files:
            print(f"⚠️ 目录下未发现 .nii.gz 文件 -> {current_dir}")
            continue

        print(f"📁 [{folder_name}] (共 {len(label_files)} 个文件)")
        
        for filename in tqdm(label_files, leave=False):
            # 提取纯净的患者姓名
            patient_name = filename.replace('.nii.gz', '')
            label_path = os.path.join(current_dir, filename)
            
            try:
                # 只读取 Label
                label_img = sitk.ReadImage(label_path)
                
                # 执行形状分析 (自动根据 Spacing 转换为真实物理坐标)
                shape_filter.Execute(label_img)
                
                # 获取两点的物理质心
                p_ans = get_landmark_centroid(shape_filter, ANS_ID)
                p_pns = get_landmark_centroid(shape_filter, PNS_ID)

                length_mm = None
                status = "Success"

                if p_ans is not None and p_pns is not None:
                    # 计算三维物理空间中的欧氏距离 (单位: mm)
                    length_mm = np.linalg.norm(p_ans - p_pns)
                else:
                    status = "Missing Landmark(s)"
                    if p_ans is None and p_pns is not None:
                        status = "Missing ANS"
                    elif p_pns is None and p_ans is not None:
                        status = "Missing PNS"
                
                # 【修改点】移除了 Source_Folder
                results.append({
                    "Patient_Name": patient_name,
                    "ANS_PNS_Length_mm": length_mm,
                    "Status": status
                })
                total_processed_files += 1
                
            except Exception as e:
                print(f"\n❌ 读取文件失败 {filename}: {e}")

    if not results:
        print("❌ 未从任何目录中提取到有效数据，程序退出。")
        return

    # ================= 整理并保存结果 =================
    df = pd.DataFrame(results)
    
    # 按照患者姓名进行自然排序
    df = df.sort_values(by='Patient_Name')
    
    # 格式化输出 (保留 3 位小数)
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.3f', encoding='utf-8-sig')
    
    # 统计信息打印
    valid_count = df['ANS_PNS_Length_mm'].notna().sum()
    print("-" * 60)
    print(f"🎉 测量任务全部完成！共扫描了 {total_processed_files} 个文件。")
    print(f"📊 成功测量双点距离的样本数: {valid_count} / {len(df)}")
    if valid_count > 0:
        mean_len = df['ANS_PNS_Length_mm'].mean()
        std_len = df['ANS_PNS_Length_mm'].std()
        print(f"📏 总体 ANS-PNS 平均长度: {mean_len:.3f} ± {std_len:.3f} mm")
    print(f"📂 详细测量报告已保存至: \n   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()