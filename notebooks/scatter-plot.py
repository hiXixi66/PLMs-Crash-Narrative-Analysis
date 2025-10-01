import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 读取 Excel 文件
# file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/main/evaluation_qwen-7b-2884.xlsx"
# file_path = "reports/crash-type-test/results/main/evaluation_llama-8b-2884-all.xlsx"
# file_path_GT = "reports/crash-type-test/results/main/evaluation_llama-8b-2884-all.xlsx"
# file_path= "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/main/evaluation_qwen-7b-2163.xlsx"
file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/evaluation_gpt-all.xlsx"
# file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/evaluation_llama-8b-all.xlsx"

# file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/main/evaluation_llama-8b-2884-all.xlsx"
# file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/evaluation_llama-70b-all.xlsx"

file_path_GT = "reports/crash-type-test/results/main/evaluation_qwen-7b-2163.xlsx"

df = pd.read_excel(file_path)
df_GT = pd.read_excel(file_path_GT)

gt_x, gt_y = [], []
pred_x, pred_y = [], []

# 遍历每一行 case
for idx, row in df.iterrows():
    if row["Total Ground Truth Vehicles"] != 2:
        continue

    entries = str(row["GT → Pred"]).split("\n")
    if len(entries) < 1:
        continue

    # 车辆1
    nums1 = re.findall(r"[-]?\d+\.?\d*", entries[0])
    if len(nums1) >= 6:
        gt1 = int(float(nums1[-6]))
        pred1 = int(float(nums1[-1]))
    else:
        continue

    # 车辆2
    nums2 = re.findall(r"[-]?\d+\.?\d*", entries[1])
    if len(nums2) >= 6:
        gt2 = int(float(nums2[-6]))
        pred2 = int(float(nums2[-1]))
    else:
        continue


    pred_x.append(pred1)
    pred_y.append(pred2)

for idx, row in df_GT.iterrows():
    if row["Total Ground Truth Vehicles"] != 2:
        continue

    entries = str(row["GT → Pred"]).split("\n")
    if len(entries) < 2:
        continue

    # 车辆1
    nums1 = re.findall(r"[-]?\d+\.?\d*", entries[0])
    if len(nums1) >= 6:
        gt1 = int(float(nums1[-6]))
        pred1 = int(float(nums1[-1]))
    else:
        continue

    # 车辆2
    nums2 = re.findall(r"[-]?\d+\.?\d*", entries[1])
    if len(nums2) >= 6:
        gt2 = int(float(nums2[-6]))
        pred2 = int(float(nums2[-1]))
    else:
        continue


    gt_x.append(gt1)
    gt_y.append(gt2)

# 计算 Pearson 相关系数
# gt_corr = np.corrcoef(gt_x, gt_y)[0,1]
# pred_corr = np.corrcoef(pred_x, pred_y)[0,1]

# gt_corr, _ = spearmanr(gt_x, gt_y)
# pred_corr, _ = spearmanr(pred_x, pred_y)

gt_corr, _ = kendalltau(gt_x, gt_y)
pred_corr, _ = kendalltau(pred_x, pred_y)
abs_delta_corr = abs(pred_corr - gt_corr)

# 绘制散点图
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(gt_x, gt_y, c='blue', alpha=0.4, edgecolor='k',marker='v', label=f"Ground Truth ")
ax.scatter(pred_x, pred_y, c='red', alpha=0.4, edgecolor='k', marker='^',label=f"Predicted ")

ax.set_xlabel("CrashType of Vehicle 1", fontsize=14)
ax.set_ylabel("CrashType of Vehicle 2", fontsize=14)
# ax.set_title("CrashType combinations - Qwen-7B")
ax.legend()
ax.grid(True)

# 标注相关系数
ax.text(2, 80, f"GT Corr = {gt_corr:.3f}", color='blue', fontsize=14)
ax.text(2, 70, f"Pred Corr = {pred_corr:.3f}", color='red', fontsize=14)
ax.text(2, 60, f"Delta Corr = {abs_delta_corr:.3f}", color='red', fontsize=14)

# 添加局部放大图 (75–85)
# 在原图的 (x0,y0) 位置放置 inset 图，单位是父轴坐标比例 (0~1)
# axins = inset_axes(ax, width="30%", height="30%", 
#                    bbox_to_anchor=(0.6, 0.2, 0.4, 0.4),  # (x0, y0, width, height)
#                    bbox_transform=ax.transAxes,          # 使用父轴的坐标系
#                    loc='lower left')
axins = fig.add_axes([0.60, 0.30, 0.25, 0.25])
# axins = inset_axes(ax, width="30%", height="30%", loc="center right")  
axins.scatter(gt_x, gt_y, c='blue', alpha=0.3, marker='v', edgecolor='k')
axins.scatter(pred_x, pred_y, c='red', alpha=0.3, marker='^',edgecolor='k')
axins.set_xlim(65, 85)
axins.set_ylim(65, 85)
axins.set_xticks([65, 75, 85])
axins.set_yticks([65, 75, 85])
axins.grid(True)
import matplotlib.patches as patches

# 在原图画矩形框
rect = patches.Rectangle((65, 65), 20, 20, linewidth=1.5, edgecolor="g", facecolor="none", linestyle="--")
ax.add_patch(rect)

# 保存与展示
plt.savefig("notebooks/crashtype_combinations_zoom-GPT-4o.png", dpi=300, bbox_inches="tight")
plt.show()