import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "airway_landmark_results_LCC.csv")
OUTPUT_IMG = os.path.join(SCRIPT_DIR, "landmark_error_broken_axis.png")
LABELS = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
# ============================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误: 找不到文件 {CSV_PATH}")
        return

    print(f"📥 正在读取数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df_melted = pd.melt(df, id_vars=['Patient_Name'], value_vars=LABELS, 
                        var_name='Landmark', value_name='Error')
    df_melted['Error'] = pd.to_numeric(df_melted['Error'], errors='coerce')
    df_clean = df_melted.dropna(subset=['Error'])

    print(f"✨ 数据清洗完毕...")

    # 1. 创建两个子图，上下拼接
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    my_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # --- 2. 绘制 ax_bottom (下方的子图，展示主要数据区) ---
    sns.boxplot(x='Landmark', y='Error', data=df_clean, palette=my_palette, showfliers=False, width=0.6, linewidth=1.5, boxprops=dict(alpha=0.8), ax=ax_bottom)
    sns.stripplot(x='Landmark', y='Error', data=df_clean, color=".2", alpha=0.4, size=4, jitter=0.2, zorder=1, ax=ax_bottom)
    
    # 设定下子图 Y 轴范围
    ax_bottom.set_ylim(-1, 20)
    # 添加参考线
    ax_bottom.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2mm (Strict)', zorder=10)
    ax_bottom.axhline(y=4.0, color='orange', linestyle=':', alpha=0.8, label='4mm (Acceptable)', zorder=10)
    
    # --- 3. 绘制 ax_top (上方的子图，单独展示极少数离群点) ---
    # 【高级技巧】ax_top 建议只画 stripplot 离群点散点图
    sns.stripplot(x='Landmark', y='Error', data=df_clean, color=".2", alpha=0.4, size=4, jitter=0.2, zorder=1, ax=ax_top)
    
    # 设定上子图 Y 轴范围（根据你数据的极端值调整）
    # ⚠️ 明确指出把上限设为了 90mm，容纳 AICV 那个个别离群点
    ax_top.set_ylim(20, 90) # 容纳例如 Case_059、Case_090 AICV 的离群点

    # --- 4. 设置 Broken Axis 的视觉效果 ---
    # 隐藏上子图的底部边框
    ax_top.spines['bottom'].set_visible(False)
    # 隐藏下子图的顶部边框
    ax_bottom.spines['top'].set_visible(False)
    # 将 X 轴刻度放在最底部的 ax_bottom
    ax_bottom.xaxis.tick_bottom()
    
    # 在 Y 轴上添加断裂标记 ("dashes")
    d = .015 # dashes 大小
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # 左上角
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上角
    kwargs.update(transform=ax_bottom.transAxes)     # 改变 zorder
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左下角
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右下角
    
    # --- 5. 图表全局细节设置 ---
    plt.suptitle('Landmark Detection Error (5-Fold Cross Validation)', fontsize=16, fontweight='bold', y=0.95)
    ax_bottom.set_xlabel('Anatomical Landmarks', fontsize=14, labelpad=10)
    # fig.text 在全局中心添加 Y 轴标签
    fig.text(0.04, 0.5, 'Localization Error (mm)', va='center', rotation='vertical', fontsize=14)
    # 移除 ax_top 的标题和标签
    ax_top.set_ylabel('')
    ax_top.set_title('')
    ax_bottom.legend(loc='upper right', frameon=True, shadow=True, bbox_to_anchor=(1, 1))

    # --- 6. 调整边距并保存 ---
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    
    print(f"🎉 Broken Axis 高清图已保存！请查看: \n   {OUTPUT_IMG}")

if __name__ == "__main__":
    main()