import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
"""
    绘制 分割结果 的 ROC 和 PR图像，每个fold选一个样本进行predict 得到npz进行计算
"""

# ================= 配置区域 =================
# 1. 真实标签目录 (GT)
GT_DIR = "/data1/xyh/data/nnUNet/nnUNet_raw/Dataset506_AirwaySegmentation/labelsTr"

# 2. 5个 Fold 的概率图所在文件夹 (每个文件夹里应有你选出的那个样本的 .npz)
PROB_FOLD_PATHS = {
    "Fold 0": "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/temp_data_validation/fold0",
    "Fold 1": "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/temp_data_validation/fold1",
    "Fold 2": "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/temp_data_validation/fold2",
    "Fold 3": "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/temp_data_validation/fold3",
    "Fold 4": "/data1/xyh/data/nnUNet/nnUNet_results/Dataset506_AirwaySegmentation/temp_data_validation/fold4",
}

# 3. 采样比例 (0.01 表示只取 1% 的体素参与绘图，防止内存溢出)
SAMPLE_RATE = 0.05 
# ===========================================

def load_data_from_npz(fold_dir):
    """从文件夹中读取第一个 .npz 文件及其对应的 GT"""
    npz_files = [f for f in os.listdir(fold_dir) if f.endswith('.npz')]
    if not npz_files: return None, None
    
    # 读取第一个样本
    case_id = npz_files[0]
    prob_path = os.path.join(fold_dir, case_id)
    gt_path = os.path.join(GT_DIR, case_id.replace('.npz', '.nii.gz'))
    
    # 加载概率图 (通道1是气道概率)
    prob_data = np.load(prob_path)['probabilities'][1] 
    # 加载 GT
    gt_img = sitk.ReadImage(gt_path)
    gt_data = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
    
    # 展平并采样
    y_true = gt_data.flatten()
    y_scores = prob_data.flatten()
    
    # 随机下采样以节省内存
    indices = np.random.choice(len(y_true), int(len(y_true) * SAMPLE_RATE), replace=False)
    return y_true[indices], y_scores[indices]

def plot_all_curves():
    plt.figure(figsize=(12, 5))
    
    # 初始化子图
    ax_roc = plt.subplot(1, 2, 1)
    ax_pr = plt.subplot(1, 2, 2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    all_y_true = []
    all_y_scores = []

    print("🚀 开始读取数据并计算曲线...")
    for i, (fold_name, path) in enumerate(PROB_FOLD_PATHS.items()):
        y_true, y_score = load_data_from_npz(path)
        if y_true is None: continue
        
        all_y_true.append(y_true)
        all_y_scores.append(y_score)
        
        # 计算该 Fold 的曲线
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        # 绘图
        ax_roc.plot(fpr, tpr, color=colors[i], alpha=0.3, label=f'{fold_name} (AUC={auc(fpr, tpr):.4f})')
        ax_pr.plot(recall, precision, color=colors[i], alpha=0.3, label=f'{fold_name} (AUC={auc(recall, precision):.4f})')

    # 计算 5-Fold 汇总后的平均曲线 (Pooling 方式)
    print("📊 正在计算总平均曲线...")
    final_y_true = np.concatenate(all_y_true)
    final_y_scores = np.concatenate(all_y_scores)
    
    # 绘制总 ROC
    fpr_all, tpr_all, _ = roc_curve(final_y_true, final_y_scores)
    ax_roc.plot(fpr_all, tpr_all, color='black', linewidth=2, label=f'Mean ROC (AUC={auc(fpr_all, tpr_all):.4f})')
    
    # 绘制总 PR
    pre_all, rec_all, _ = precision_recall_curve(final_y_true, final_y_scores)
    ax_pr.plot(rec_all, pre_all, color='black', linewidth=2, label=f'Mean PR (AUC={auc(rec_all, pre_all):.4f})')

    # 装饰 ROC 图
    ax_roc.set_title('Receiver Operating Characteristic (ROC)')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right', fontsize='small')
    ax_roc.grid(alpha=0.3)

    # 装饰 PR 图
    ax_pr.set_title('Precision-Recall (PR) Curve')
    ax_pr.set_xlabel('Recall (Sensitivity)')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='lower left', fontsize='small')
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('5fold_curves_summary.png', dpi=300)
    print("✅ 绘图完成！图片已保存为: 5fold_curves_summary.png")
    plt.show()

if __name__ == "__main__":
    plot_all_curves()