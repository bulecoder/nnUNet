import os
import csv
import re

def parse_eval_file(file_path, fold_num):
    """
    解析单个fold_eval.txt文件，提取Case Name、Dice、HD95
    file_path: 单个eval文件的路径
    fold_num: 手动指定的fold编号（0~4）
    """
    results = []
    # 匹配数据行的正则：Case Name（含空格/特殊字符） | Dice | HD95
    pattern = re.compile(r'^(.+?)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s*$')
    
    if not os.path.exists(file_path):
        print(f'❌ 错误：{file_path} 文件不存在！')
        return results
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # 跳过分隔线、表头、平均值行
            if any([line.startswith('===') , line.startswith('----'), 
                   line.startswith('Case Name'), line.startswith('AVERAGE')]):
                continue
            # 匹配数据行
            match = pattern.match(line)
            if match:
                case_name = match.group(1).strip()
                dice = match.group(2).strip()
                hd95 = match.group(3).strip()
                # 改动1：删除Fold字段
                results.append({
                    'Case Name': case_name,
                    'Dice': dice,
                    'HD95 (mm)': hd95
                })
    print(f'✅ 已解析 Fold{fold_num}：{file_path}，共 {len(results)} 条数据')
    return results

def merge_all_folds_to_csv(fold_file_paths, output_csv='airway_segmentation_results.csv'):
    """
    合并不同目录下的fold结果到CSV
    fold_file_paths: 字典，key=fold编号，value=文件路径
    output_csv: 输出的CSV文件名
    """
    all_results = []
    # 遍历每个fold的文件路径
    for fold_num, file_path in fold_file_paths.items():
        fold_results = parse_eval_file(file_path, fold_num)
        all_results.extend(fold_results)
    
    # 写入CSV文件
    if all_results:
        # 改动2：删除headers中的Fold列
        headers = ['Case Name', 'Dice', 'HD95 (mm)']
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_results)
        print(f'\n🎉 所有结果已汇总到：{os.path.abspath(output_csv)}')
    else:
        print('❌ 未解析到任何有效数据！')

# -------------------------- 核心配置（无需修改） --------------------------
FOLD_FILE_PATHS = {
    0: '/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/fold0_eval.txt',
    1: '/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/fold1_eval.txt',  
    2: '/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/fold2_eval.txt',  
    3: '/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval.txt',  
    4: '/data1/xyh/data/nnUNet/nnUNet_results/Dataset502_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval.txt'  
}
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    merge_all_folds_to_csv(FOLD_FILE_PATHS)