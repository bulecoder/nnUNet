import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
# 1. 你的汇总 CSV 文件路径 (默认在同级目录)
# ⚠️ 注意: 请确保你的 CSV 文件中第一列 Patient_Name 是映射后的患者姓名
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "airway_landmark_results_LCC.csv")

# 2. 输出的高清图像文件名 (支持 .png, .pdf, .svg)
OUTPUT_IMG = os.path.join(SCRIPT_DIR, "landmark_error_boxplot_updated.png")

# 3. 标签列表
LABELS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误: 找不到文件 {CSV_PATH}，请确认路径。")
        return

    print(f"📥 正在读取数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 1. 数据转换 (Wide to Long format)
    df_melted = pd.melt(df, id_vars=['Patient_Name'], value_vars=LABELS, 
                        var_name='Landmark', value_name='Error')

    # 2. 数据清洗 (errors='coerce' 完美处理 'FP', 'Miss' 等干扰项)
    df_melted['Error'] = pd.to_numeric(df_melted['Error'], errors='coerce')
    df_clean = df_melted.dropna(subset=['Error'])

    print(f"✨ 数据清洗完毕，有效评估点数量: {len(df_clean)}")

    # 3. 开始绘图 (设置期刊级美化风格)
    plt.figure(figsize=(10, 6)) # 设置画布比例
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 【改动点1】更明显的颜色区分度
    # 我为你更换了色彩更鲜艳、对比度更高的自定义 palette（手动指定鲜艳颜色），代替了原来的 "Set2"
    my_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"] # 自定义更鲜艳的颜色
    # 也可以使用 Seaborn 自带的高对比度色板，例如：palette="viridis" 或 palette="rocket"

    # 绘制箱线图
    ax = sns.boxplot(
        x='Landmark', y='Error', data=df_clean,
        palette=my_palette,      # 配色方案
        showfliers=False,    # 隐藏默认的离群点，因为我们要用散点图来画
        width=0.6, 
        linewidth=1.5,
        boxprops=dict(alpha=0.8) # 箱体轻微透明
    )

    # 绘制叠加散点图 (展示数据点的真实分布，比纯箱线图更高级)
    # 【改动点2】微调 zorder，确保 stripplot 在boxplot底层
    sns.stripplot(
        x='Landmark', y='Error', data=df_clean,
        color=".2",          # 散点颜色为深灰
        alpha=0.4,           # 半透明，防止点堆叠看不清
        size=4,              # 散点大小
        jitter=0.2,          # 随机抖动宽度
        zorder=1             # 图层顺序设为最底层
    )

    # 4. 图表细节设置
    plt.title('Landmark Detection Error (5-Fold Cross Validation)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Anatomical Landmarks', fontsize=14, labelpad=10)
    plt.ylabel('Localization Error (mm)', fontsize=14, labelpad=10)

    # 【改动点3】解决数据压缩问题：截断 Y 轴
    # 明确解开了你代码中的 plt.ylim(-1, 20) 这行注释，并将上限设为 20mm
    # 这是解决压缩问题的核心操作！限制 Y 轴范围后，主体数据区（0-20mm）将被放大显示，清晰可见
    # 极端的离群点会放在图表上方而不拉大 Y 轴比例。你可以根据需要调整范围（例如 plt.ylim(-1, 30)）。
    plt.ylim(-1, 20) 

    # 添加临床可接受的参考阈值线
    # 【改动点4】参考线 zorder 设高，确保最顶层，不被遮挡
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='2mm (Strict)', zorder=10)
    plt.axhline(y=4.0, color='orange', linestyle=':', alpha=0.8, linewidth=1.5, label='4mm (Acceptable)', zorder=10)
    
    # 调整图例
    plt.legend(loc='upper right', frameon=True, shadow=True, bbox_to_anchor=(1, 1))

    # 5. 调整边距并保存
    plt.tight_layout()
    # 300 dpi 满足绝大多数期刊要求。如需 LaTeX 论文排版，建议后缀改为 .pdf 或 .svg。
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight') 
    
    print(f"🎉 高清 Boxplot 已更新并保存！请查看: \n   {OUTPUT_IMG}")

if __name__ == "__main__":
    main()