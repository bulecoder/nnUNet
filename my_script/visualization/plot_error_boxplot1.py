import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================= 配置区域 =================
# 1. 你的汇总 CSV 文件路径 (默认在同级目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "airway_landmark_results_LCC.csv")

# 2. 输出的高清图像文件名 (支持 .png, .pdf, .svg)
OUTPUT_IMG = os.path.join(SCRIPT_DIR, "landmark_error_boxplot.png")

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
    # 将宽表转换为适合 seaborn 绘图的长表
    df_melted = pd.melt(df, id_vars=['Patient_Name'], value_vars=LABELS, 
                        var_name='Landmark', value_name='Error')

    # 2. 数据清洗
    # 将无法转换为浮点数的字符（如 'FP', 'Miss', '---'）强制转换为 NaN，并剔除
    df_melted['Error'] = pd.to_numeric(df_melted['Error'], errors='coerce')
    df_clean = df_melted.dropna(subset=['Error'])

    print(f"✨ 数据清洗完毕，有效评估点数量: {len(df_clean)}")

    # 3. 开始绘图 (设置期刊级美化风格)
    plt.figure(figsize=(10, 6)) # 设置画布比例
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 绘制箱线图
    ax = sns.boxplot(
        x='Landmark', y='Error', data=df_clean,
        palette="Set2",      # 配色方案
        showfliers=False,    # 隐藏默认的离群点，因为我们要用散点图来画
        width=0.6, 
        linewidth=1.5,
        boxprops=dict(alpha=0.8) # 箱体轻微透明
    )

    # 绘制叠加散点图 (展示数据点的真实分布，比纯箱线图更高级)
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

    # 【可选】限制 Y 轴的显示范围
    # 如果有些异常点高达 80mm，会导致整个箱体被压缩得看不清。
    # 解开下面这行代码可以截断 Y 轴（例如只显示 0~20mm 的范围）：
    # plt.ylim(-1, 20)

    # 添加临床可接受的参考阈值线
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='2mm (Strict)')
    plt.axhline(y=4.0, color='orange', linestyle=':', alpha=0.8, linewidth=1.5, label='4mm (Acceptable)')
    
    # 调整图例
    plt.legend(loc='upper right', frameon=True, shadow=True)

    # 5. 调整边距并保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight') # 300 dpi 满足绝大多数期刊要求
    
    print(f"🎉 高清 Boxplot 已成功生成！请查看: \n   {OUTPUT_IMG}")

if __name__ == "__main__":
    main()