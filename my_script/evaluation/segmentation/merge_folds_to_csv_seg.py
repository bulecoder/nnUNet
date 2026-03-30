import os
import csv
import re
"""
    合并5个fold的分割评价结果到csv
"""

TASK_ID = 510   # 任务编号

def parse_eval_file(file_path, fold_num):
    """
    解析单个fold_eval.txt文件，提取Case Name、Dice、IoU、HD95、ASSD
    file_path: 单个eval文件的路径
    fold_num: 手动指定的fold编号（0~4）
    """
    results = []
    # 更新正则表达式：匹配 5 列数据（Case Name | Dice | IoU | HD95 | ASSD）
    # 允许数值为数字、点或 NaN
    pattern = re.compile(r'^(.+?)\s+\|\s+([0-9.Na]+)\s+\|\s+([0-9.Na]+)\s+\|\s+([0-9.Na]+)\s+\|\s+([0-9.Na]+)\s*$')
    
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
                iou = match.group(3).strip()
                hd95 = match.group(4).strip()
                assd = match.group(5).strip()
                
                results.append({
                    'Case Name': case_name,
                    'Dice': dice,
                    'IoU': iou,
                    'HD95 (mm)': hd95,
                    'ASSD (mm)': assd
                })
    print(f'✅ 已解析 Fold{fold_num}：{file_path}，共 {len(results)} 条数据')
    return results

def merge_all_folds_to_csv(fold_file_paths, output_csv=f'airway_segmentation_{TASK_ID}_results_LCC.csv'):
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
        # 更新表头以包含 IoU 和 ASSD
        headers = ['Case Name', 'Dice', 'IoU', 'HD95 (mm)', 'ASSD (mm)']
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_results)
        print(f'\n🎉 所有结果已汇总到：{os.path.abspath(output_csv)}')
    else:
        print('❌ 未解析到任何有效数据！')

# -------------------------- 核心配置 --------------------------
# FOLD_FILE_PATHS = {
#     0: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/fold0_eval.txt',
#     1: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/fold1_eval.txt',  
#     2: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/fold2_eval.txt',  
#     3: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval.txt',  
#     4: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval.txt'  
# }
FOLD_FILE_PATHS = {
    0: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/fold0_eval_LCC.txt',
    1: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/fold1_eval_LCC.txt',  
    2: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/fold2_eval_LCC.txt',  
    3: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/fold3_eval_LCC.txt',  
    4: f'/data1/xyh/data/nnUNet/nnUNet_results/Dataset{TASK_ID}_AirwaySegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/fold4_eval_LCC.txt'  
}
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    merge_all_folds_to_csv(FOLD_FILE_PATHS)