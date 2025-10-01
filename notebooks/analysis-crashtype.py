import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

# ============== 读取与清洗 ==============
file_path = "reports/crash-type-test/crash_type_resultsqkv-temp5.csv"
df = pd.read_csv(file_path).copy()

conf_map = {
    "A":"Right roadside departure","B":"Left roadside departure","C":"Forward impact",
    "D":"Rear-end","E":"Forward impact","F":"Angle, sideswipe",
    "G":"Head-on","H":"Forward impact","I":"Angle, sideswipe",
    "J":"Turn across path","K":"Turn into path","L":"Straight paths","M":"Backing, etc."
}
letters_by_group = {
    1:["A","B","C"],
    2:["D","E","F"],
    3:["G","H","I"],
    4:["J","K"],
    5:["L"],
    6:["M"],
}
group_name = {
    1:"Single driver",
    2:"Same trafficway, same direction",
    3:"Same trafficway, opposite direction",
    4:"Changing trafficway, vehicle turning",
    5:"Intersecting paths (vehicle damage)",
    6:"Miscellaneous",
}
digit_group = {c:g for g, L in letters_by_group.items() for c in L}

# 大类专属颜色（你可以改成自己喜欢的 HEX）
# cat_colors = {
#     1: "tab:blue",
#     2: "tab:orange",
#     3: "tab:green",
#     4: "tab:purple",
#     5: "tab:brown",
#     6: "tab:gray",
# }
from matplotlib import cm

# 从 tab20 里取前 6 种颜色

cat_colors = {
    1: "#a6cee3",  # 浅蓝
    2: "#fb9a99",  # 浅红
    3: "#fdbf6f",  # 浅橙
    4: "#cab2d6",  # 浅紫
    5: "#ffff99",  # 浅黄
    6: "#d9d9d9",  # 浅灰
}
# 清洗
for col in ["gt_type","pred_type"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
df["gt_conf"] = df["gt_conf"].astype(str).str.strip()
df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)

# ============== 按 configuration（字母）统计 ==============
acc_by_conf = (
    df.groupby("gt_conf")["correct"]
      .agg(total="count", correct="sum")
      .assign(accuracy=lambda x: x["correct"]/x["total"])
      .reset_index()
)
order = list("ABCDEFGHIJKLM")
acc_by_conf["label"] = acc_by_conf["gt_conf"].map(conf_map)
acc_by_conf["ord"] = acc_by_conf["gt_conf"].apply(lambda x: order.index(x) if x in order else 999)
acc_by_conf = acc_by_conf.sort_values(["ord","gt_conf"]).reset_index(drop=True)

overall_total = float(acc_by_conf["total"].sum())
acc_by_conf["share_total"]   = acc_by_conf["total"]   / overall_total
acc_by_conf["share_correct"] = acc_by_conf["correct"] / overall_total

# ============== 按 category（数字大类）统计 ==============
df["group"] = df["gt_conf"].map(digit_group)
group_stats = (
    df.groupby("group")["correct"]
      .agg(total="count", correct="sum")
      .reindex(range(1,7), fill_value=0)
      .reset_index()
)
group_stats["accuracy"]      = group_stats["correct"] / group_stats["total"].replace(0, np.nan)
group_stats["share_total"]   = group_stats["total"]   / overall_total
group_stats["share_correct"] = group_stats["correct"] / overall_total

group_share       = group_stats["share_total"].to_numpy(float)
group_share_corr  = group_stats["share_correct"].to_numpy(float)
group_acc         = group_stats["accuracy"].fillna(0).to_numpy(float)

# ============== 画两个柱状图（上：配置；下：类别） ==============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios":[2,2]})

# ---- 上图：每个 configuration 的底色按所属 category 着色，红色不变 ----
x = np.arange(len(acc_by_conf))
barw_top = 0.8
gap_top  = 1.0 - barw_top

blue = acc_by_conf["share_total"].to_numpy()
red  = acc_by_conf["share_correct"].to_numpy()

conf_letters_ordered = acc_by_conf["gt_conf"].tolist()
conf_groups = [digit_group[c] for c in conf_letters_ordered]
base_colors_top = [cat_colors[g] for g in conf_groups]

# 底色（类别色）
ax1.bar(x, blue, width=barw_top, color=base_colors_top, edgecolor="k")
correct_color = "#ccebc5"

ax1.bar(x, red, width=barw_top, color=correct_color)

# 标注：把准确率放在红柱上方
ax1.set_ylim(0, max(1e-6, blue.max()) * 1.25)
for i, (r_h, acc) in enumerate(zip(red, acc_by_conf["accuracy"])):
    ax1.text(i, r_h + max(blue.max(), 1e-6)*0.035, f"{acc:.1%}",
             ha="center", va="bottom", fontsize=12, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(acc_by_conf["label"], rotation=20, ha="right", fontsize=12, fontweight="bold")
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_ylabel("Share of all samples", fontsize=14, fontweight="bold")
ax1.set_title("LLM Prediction Accuracy of Crash Type Across Crash Configurations",
              fontsize=14, fontweight="bold")

# 自定义图例：6 个类别色 + 红色
legend_handles_top = [Patch(facecolor=cat_colors[g], edgecolor="k", label=group_name[g]) for g in range(1,7)]
legend_handles_top.append(Patch(facecolor=correct_color, edgecolor="k", label="Correct share"))
# ax1.legend(handles=legend_handles_top, fontsize=10, ncol=2, loc="upper left", frameon=True)

# ---- 下图：柱子与上图对齐（中心=上面成员跨度中心；宽度=成员宽+间隙）----
letter_to_idx = {conf: idx for idx, conf in enumerate(acc_by_conf["gt_conf"].tolist())}
centers = []
widths  = []
for g in range(1, 7):
    members = letters_by_group[g]
    idxs = sorted(letter_to_idx[m] for m in members if m in letter_to_idx)
    if not idxs:
        centers.append(np.nan); widths.append(0.0); continue
    i_start, i_end = idxs[0], idxs[-1]
    n = len(idxs)
    center = (x[i_start] + x[i_end]) / 2.0
    width  = n * barw_top + (n - 1) * gap_top
    centers.append(center); widths.append(width)
centers = np.array(centers)
widths  = np.array(widths)

# 类别柱（底色=类别色）+ 正确（红）
ax2.bar(centers, group_share,      width=widths, color=[cat_colors[g] for g in range(1,7)], edgecolor="k")
ax2.bar(centers, group_share_corr, width=widths, color=correct_color)

ax2.set_ylim(0, max(1e-6, group_share.max()) * 1.25)
for cx, r_h, acc in zip(centers, group_share_corr, group_acc):
    ax2.text(cx, r_h + max(group_share.max(), 1e-6)*0.035, f"{acc:.1%}",
             ha="center", va="bottom", fontsize=12, fontweight="bold")

ax2.set_xticks(centers)
ax2.set_xticklabels([f"{group_name[g]}" for g in range(1,7)],
                    rotation=20, ha="right", fontsize=12, fontweight="bold")
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.set_ylabel("Share of all samples", fontsize=14, fontweight="bold")
ax2.set_title("LLM Prediction Accuracy of Crash Type Across Crash Categories",
              fontsize=14, fontweight="bold")

legend_handles = [Patch(facecolor=cat_colors[g], edgecolor="k", label=group_name[g]) for g in range(1,7)]
legend_handles.append(Patch(facecolor=correct_color, edgecolor="k", label="Correct share"))

# 为右侧图例腾出空间（根据字体大小可微调 0.80~0.85）
fig.subplots_adjust(right=0.82)
fig.subplots_adjust(hspace=0.5)
# 在右侧中部放一个纵向图例（从上到下）
import textwrap

# 预先对 label 进行换行
wrapped_handles = []
for h in legend_handles:
    label = h.get_label()
    wrapped_label = "\n".join(textwrap.wrap(label, width=20))  # 25 字符后换行
    wrapped_handles.append(Patch(facecolor=h.get_facecolor(), edgecolor="k", label=wrapped_label)
)

# 调整图例格式
fig.legend(
    handles=wrapped_handles,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),  # 右侧居中
    frameon=False,
    fontsize=13,
    ncol=1,
    title="",
    title_fontsize=14,
    labelspacing=2,   # 增加行距（默认 0.5）
    handleheight=2.0    # 增加图标和文字之间的垂直距离
)


# 最后保存
# plt.tight_layout()
out_dir = "reports/crash-type-test/metrics-temp5"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "two_panel_percent_with_category_colors_and_accuracy(Temperature = 0.5).png"),
            dpi=300, bbox_inches="tight")
plt.show()
# === 2) 每个 gt_conf 的混淆热图（显示百分比） ===
def plot_heatmap_percent(matrix, row_labels, col_labels, title, save_path):
    total = matrix.sum()
    if total == 0:
        total = 1  # 避免除零

    fig, ax = plt.subplots(figsize=(6,5))
    perc_matrix = matrix / total * 100
    vmax = perc_matrix.max()  # 自动取最大值
    im = ax.imshow(perc_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=vmax*1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of total")

    # 坐标轴刻度
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("pred_type")
    ax.set_ylabel("gt_type")
    ax.set_title(title)

    # 格子上标注百分比
    for i in range(perc_matrix.shape[0]):
        for j in range(perc_matrix.shape[1]):
            ax.text(j, i, f"{perc_matrix[i, j]:.1f}%", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

# 预定义每个 gt_conf 对应的 gt_type 列表
conf_gt_type_map = {
    "A": [1, 2, 3,4,5],           # 这里的数字随你定义
    "B": [6, 7, 8, 9, 10],
    "C": [11, 12, 13, 14, 15, 16],
    "D": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    "E": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    "F": [44, 45, 46, 47, 48, 49],
    "G": [50, 51, 52, 53],
    "H": [54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
    "I": [64, 65, 66, 67],
    "J": [68, 69, 70, 71, 72, 73, 74, 75],
    "K": [76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
    "L": [86, 87, 88, 89, 90, 91],
    "M": [92, 93, 98, 99, 100]
    
    # ... 其他的也填好
}

for conf, sub in df.groupby("gt_conf"):
    # 如果该 gt_conf 没有定义，跳过
    if conf not in conf_gt_type_map:
        print(f"[Warning] gt_conf {conf} not in predefined map, skipped.")
        continue

    row_labels = conf_gt_type_map[conf]   # 按你指定的顺序
    col_labels = conf_gt_type_map[conf]   # 这里我假设预测类集合也相同

    r_index = {v: i for i, v in enumerate(row_labels)}
    c_index = {v: i for i, v in enumerate(col_labels)}

    # 初始化矩阵
    mat = np.zeros((len(row_labels), len(col_labels)), dtype=float)

    for _, r in sub.iterrows():
        if pd.isna(r["gt_type"]) or pd.isna(r["pred_type"]):
            continue
        gt_val = int(r["gt_type"])
        pred_val = int(r["pred_type"])
        # 如果该类在映射表中才计数
        if gt_val in r_index and pred_val in c_index:
            mat[r_index[gt_val], c_index[pred_val]] += 1.0

    save_path = os.path.join(out_dir, f"heatmap_conf_{conf}_percent.png")
    plot_heatmap_percent(mat, row_labels, col_labels, f"Confusion (gt_conf={conf})", save_path)

