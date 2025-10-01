import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
folder = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/mancolll-test/class-9"
out_folder = os.path.join(folder, "plots")
os.makedirs(out_folder, exist_ok=True)  # 创建保存图片的目录

# 固定的 collision_type 顺序
collision_types = [0, 1, 2, 4, 5, 6, 9]

for fname in os.listdir(folder):
    if not fname.endswith(".xlsx"):
        continue

    fpath = os.path.join(folder, fname)
    df = pd.read_excel(fpath)

    total = len(df)
    subset = df[df["MANCOLL"] == 9]
    subset_count = len(subset)
    proportion = subset_count / total if total > 0 else 0

    # === 输出 MANCOLL=9 占比（文字形式） ===
    print(f"\nFile: {fname}")
    print(f"MANCOLL=9 count: {subset_count}/{total} ({proportion*100:.2f}%)")

    # === 画 collision_type 分布柱状图 ===
    collision_dist = subset["collision_type"].value_counts(normalize=True)

    # 重新索引，确保顺序固定，缺失类别填0
    collision_dist = collision_dist.reindex(collision_types, fill_value=0)

    fig, ax = plt.subplots(figsize=(5,4))  # 更紧凑
    collision_dist.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black", width=0.7)

    # 在每个柱子上写数值
    for p in ax.patches:
        value = p.get_height()
        if value > 0:  # 避免标 0
            ax.text(
                p.get_x() + p.get_width()/2,      # x 坐标：柱子中点
                value + 0.01,                     # y 坐标：柱子顶端往上 0.01
                f"{value*100:.1f}%",              # 格式化为百分比
                ha="center", va="bottom",
                fontsize=15,   # 字体更大更粗
                color="black"
            )

    # 如果 collision_type 里有 9，就单独标注
    if 9 in collision_dist.index and collision_dist.loc[9] > 0:
        val9 = collision_dist.loc[9]
    # ax.text(
    #     0.95, 0.75, 
    #     f"Share of Unknown (original) = {57/30:.1f}%", 
    #     ha="right", va="top", 
    #     color="red", fontsize=13, fontweight="bold",
    #     transform=ax.transAxes   # 用坐标轴的相对坐标系 (0~1)
    # )

    title_name = fname[:-5] if fname.lower().endswith(".xlsx") else fname
    ymax = collision_dist.max() + 0.05   # 在最大值基础上多加 0.05
    ax.set_ylim(0, ymax)
    ax.set_title(title_name, fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.set_xlabel("Manner of collision", fontsize=14)
    ax.set_xticks(range(len(collision_types)))
    ax.set_xticklabels(collision_types, fontsize=14, rotation=0)

    # plt.tight_layout(pad=1)  # 紧凑布局
    # 保存图片（文件名与 Excel 文件对应）
    out_path = os.path.join(out_folder, fname.replace(".xlsx", ".png"))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # 关闭图，避免内存占用

print(f"\n所有图已保存到: {out_folder}")


import os
import pandas as pd
import matplotlib.pyplot as plt

# 指定文件
pie_file = r"/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/mancolll-test/self-con/MANCOLL-classification_LlaMA3 8B-1251-2.xlsx"

# 关心的类别与含义字典
mancoll_order = [0, 1, 2, 4, 5, 6, 9]
mancoll_meaning = {
    0: "Not Collision with Vehicle in Transport",
    1: "Rear-End",
    2: "Head-On",
    4: "Angle",
    5: "Sideswipe, Same Direction",
    6: "Sideswipe, Opposite Direction",
    9: "Unknown",
}

# 读取
df_pie = pd.read_excel(pie_file)

# 统计比例（按 MANCOLL 列），缺失类别补 0
prop = (df_pie["MANCOLL"]
        .value_counts(normalize=True)
        .reindex(mancoll_order, fill_value=0.0))

# 输出到控制台
print("\nMANCOLL proportions in file:")
for k in mancoll_order:
    print(f"{k} ({mancoll_meaning[k]}): {prop[k]*100:.2f}%")

# 画图（长方形画布）
# —— 更和谐的蓝系调色板（主蓝 + 蓝绿 + 灰蓝）——
colors = [
    "#1f77b4",  # deep blue
    "#4c92c3",  # blue
    "#74add1",  # light blue
    "#a6cbe3",  # pale blue
    "#5aa9a6",  # teal
    "#82c0cc",  # light teal
    "#9aa7bd",  # gray-blue
]
import numpy as np

# 颜色不变，先画饼
fig, ax = plt.subplots(figsize=(7.5, 5))
wedges, _ = ax.pie(
    prop.values, startangle=20, colors=colors,
    wedgeprops=dict(linewidth=1, edgecolor="white")
)

# ---- 标注设置 ----
THRESH = 0.01       # 只标注 >=2% 的扇区；更小的放图例里
R_TEXT = 1.        # 文本半径（离饼更远）
R_TAIL = 0.8        # 折线起点半径（扇形边缘）
MIN_GAP = 0.06       # 相邻文本在 y 方向的最小间距（轴坐标）

# 先计算所有候选文本位置
labels_xy = []
for i, w in enumerate(wedges):
    p = float(prop.values[i])
    if p < THRESH:
        continue  # 小于阈值不画文本
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
    x_txt, y_txt = R_TEXT*np.cos(ang), R_TEXT*np.sin(ang)
    x_tail, y_tail = R_TAIL*np.cos(ang), R_TAIL*np.sin(ang)
    ha = "left" if x_txt >= 0 else "right"
    labels_xy.append([i, p, x_txt, y_txt, x_tail, y_tail, ha])

# 按 y 排序并做简单的“避让”
labels_xy.sort(key=lambda t: t[3])  # 3 是 y_txt
for j in range(1, len(labels_xy)):
    # 与前一个比较，保持最小间距
    if labels_xy[j][3] - labels_xy[j-1][3] < MIN_GAP:
        labels_xy[j][3] = labels_xy[j-1][3] + MIN_GAP

# 画带折线的外部标注
EXTRA_ROTATE = np.deg2rad(70)  # 向右旋转 70°

for i, w in enumerate(wedges):
    p = float(prop.values[i])
    if p < 0:
        continue
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)

    # 对小比例（或指定 ID，比如 5,6,9）的切片，加角度偏移
    # if i in [1,4, 5, 6]:   # 这里 i 是索引，对应 MANCOLL=5,6,9 的那几个
    #     ang += EXTRA_ROTATE

    # 计算外部标注位置
    x_txt, y_txt = R_TEXT*np.cos(ang), R_TEXT*np.sin(ang)
    x_tail, y_tail = R_TAIL*np.cos(ang), R_TAIL*np.sin(ang)
    ha = "left" if x_txt >= 0 else "right"

    ax.annotate(
        f"{p*100:.1f}%", xy=(x_tail, y_tail), xytext=(x_txt, y_txt),
        ha=ha, va="center", fontsize=14, fontweight="bold", color="#2b2b2b",
        arrowprops=dict(arrowstyle="-", lw=1.2, color="#555",
                        connectionstyle="arc3,rad=0.25")
    )


# 标题上移，避免重叠
ax.set_title(f"MANCOLL distribution", fontsize=14)
ax.axis("equal")

# 标题、图例同之前
title_name = os.path.splitext(os.path.basename(pie_file))[0]
# ax.set_title(f"MANCOLL distribution\n{title_name}", fontsize=14, pad=12)
legend_labels = [
    f"{k}: {mancoll_meaning[k]}  —  {prop[k]*100:.1f}%"
    for k in mancoll_order
]
ax.legend(wedges, legend_labels, title="MANCOLL (ID: Meaning — Share)",
          loc="center left", bbox_to_anchor=(0.9, 0.5), frameon=True,
          fontsize=14, title_fontsize=15)
ax.axis("equal")

out_dir = os.path.join(os.path.dirname(pie_file), "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, title_name.replace(" ", "_") + "_pie.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
