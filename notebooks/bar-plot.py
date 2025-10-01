import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.spatial.distance import jensenshannon

# 文件路径
# file_path_pred = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/main/evaluation_qwen-7b-2163.xlsx"
# file_path_pred = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/evaluation_gpt-all.xlsx"
file_path_pred = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/reports/crash-type-test/results/evaluation_llama-70b-all.xlsx"

# file_path_pred = "reports/crash-type-test/results/main/evaluation_llama-8b-2163-all.xlsx"
file_path_gt = "reports/crash-type-test/results/main/evaluation_qwen-7b-2163.xlsx"

df_pred = pd.read_excel(file_path_pred)
df_gt = pd.read_excel(file_path_gt)
p = 16
# 提取单车事故的 GT 和 Pred CrashType
gt_crashtypes = []
pred_crashtypes = []

for idx, row in df_pred.iterrows():
    if row["Total Ground Truth Vehicles"] != 1:
        continue

    entries = str(row["GT → Pred"]).split("\n")
    if len(entries) < 1:
        continue

    nums = re.findall(r"[-]?\d+\.?\d*", entries[0])
    if len(nums) >= 6:
        gt = int(float(nums[-6]))
        pred = int(float(nums[-1]))
        # 只保留 0–20 范围
        if 0 <= gt <= p and 0 <= pred <= p:
            gt_crashtypes.append(gt)
            pred_crashtypes.append(pred)

# 从 GT 文件里提取单车事故的真实 GT 分布（用于 baseline）
gt_crashtypes_all = []
for idx, row in df_gt.iterrows():
    if row["Total Ground Truth Vehicles"] != 1:
        continue

    entries = str(row["GT → Pred"]).split("\n")
    if len(entries) < 1:
        continue

    nums = re.findall(r"[-]?\d+\.?\d*", entries[0])
    if len(nums) >= 6:
        gt_val = int(float(nums[-6]))
        if 0 <= gt_val <= p:
            gt_crashtypes_all.append(gt_val)

# -------------------------------
# 统计分布 (只统计 0–p)
# -------------------------------
max_class = p+1  # 只考虑 0–p

gt_counts = np.bincount(gt_crashtypes_all, minlength=max_class)[:max_class]
pred_counts = np.bincount(pred_crashtypes, minlength=max_class)[:max_class]


# 转换为概率分布
gt_dist = gt_counts / gt_counts.sum() if gt_counts.sum() > 0 else gt_counts
pred_dist = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else pred_counts

# 计算 Jensen-Shannon Divergence
js_div = jensenshannon(gt_dist, pred_dist)**2
print(f"Jensen-Shannon Divergence between GT and Pred distributions: {js_div:.4f}")

# -------------------------------
# 可视化
# -------------------------------
classes = np.arange(max_class)

plt.figure(figsize=(4.5, 3.8))
plt.bar(classes - 0.2, gt_dist, width=0.4, label="Ground Truth (All GT)", color="blue", alpha=0.6)
plt.bar(classes + 0.2, pred_dist, width=0.4, label="LLM Predicted", color="red", alpha=0.6)

plt.xlabel("CrashType", fontsize=14)
plt.ylabel("Relative Frequency", fontsize=14)
plt.title("LLaMA3-70B", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# 设置 x 轴刻度
plt.xticks([0, 4, 8, 12, 16])

# 在图里标注 JS 散度结果
plt.text(3,max(max(gt_dist), max(pred_dist)) * 0.9,
         f"JS Divergence = {js_div:.4f}",
         fontsize=15, color="black")
plt.legend(loc='center right')

plt.savefig("notebooks/crashtype_distribution_single_vehicle_llama70B.png", dpi=300, bbox_inches="tight")
plt.show()
