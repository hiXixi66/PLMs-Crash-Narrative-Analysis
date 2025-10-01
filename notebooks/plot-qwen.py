import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

# data = {
#     "Model": ["Qwen7b"] * 4 + ["Qwen7b-721"] * 4 + ["Qwen7b-1442"] * 4 + ["Qwen7b-2146"] * 4 + ["GPT-4o"] * 4,
#     "Number of vehicles": ["1", "2", "3", ">3"] * 5,
#     "Proportion of cases": [0.374, 0.519, 0.061, 0.016] * 5,
#     "Accuracy": [
#         0.253, 0.217, 0.148, 0.062,
#         0.767, 0.528, 0.622, 0.687,
#         0.783, 0.643, 0.728, 0.734,
#         0.786, 0.679, 0.739, 0.731,
#         0.453, 0.643, 0.588, 0.703
#     ]
# }
# data = {
#     "Model": ["Qwen7b"] * 4 + ["Qwen7b-721"] * 4 + ["Qwen7b-1442"] * 4 + ["Qwen7b-2146"] * 4 + ["GPT-4o"] * 4,
#     "Number of vehicles": ["1", "2", "3", ">3"] * 5,
#     "Proportion of cases": [0.374, 0.519, 0.061, 0.016] * 5,
#     "Accuracy": [
#         0.253, 0.217, 0.148, 0.062,
#         0.751, 0.549, 0.622, 0.691,
#         0.761, 0.613, 0.672, 0.729,
#         0.769, 0.621, 0.671, 0.730,
#         0.453, 0.643, 0.588, 0.703
#     ]
# }
# q
data = {
    "Model": ["Original"] * 4 + ["721-step"] * 4 + ["1442-step"] * 4 + ["2163-step"] * 4 + ["GPT-4o"] * 4,
    "Number of vehicles": ["1", "2", "3", ">3"] * 5,
    "Proportion of cases": [0.374, 0.519, 0.061, 0.016] * 5,
    # "Accuracy": [
    #     0.253, 0.217, 0.148, 0.062,
    #     0.772, 0.576, 0.689, 0.703,
    #     0.783, 0.689, 0.728, 0.794,
    #     0.791, 0.703, 0.806, 0.809,
    #     0.453, 0.643, 0.588, 0.703
    # ]
    # "Accuracy": [
    #     0.253, 0.217, 0.148, 0.062,
    #     0.751, 0.549, 0.622, 0.691,
    #     0.761, 0.613, 0.672, 0.729,
    #     0.769, 0.621, 0.671, 0.730,
    #     0.453, 0.643, 0.588, 0.703
    # ]
    "Accuracy": [
        0.253, 0.217, 0.148, 0.062,
        0.767, 0.528, 0.622, 0.687,
        0.783, 0.643, 0.728, 0.734,
        0.786, 0.679, 0.739, 0.731,
        0.453, 0.643, 0.588, 0.703
    ]
}

df = pd.DataFrame(data)
# 创建热力图用的数据透视表
heatmap_df = df.pivot(index="Number of vehicles", columns="Model", values="Accuracy").reindex([">3", "3", "2", "1"])

# 设置输出路径
output_path_heatmap = "reports/crash-type-test/figure/accuracy_heatmap_by_vehicles_and_model.png"
os.makedirs(os.path.dirname(output_path_heatmap), exist_ok=True)

annot_df = df.pivot(index="Number of vehicles", columns="Model", values="Accuracy").reindex([">3", "3", "2", "1"])
cases_df = df.pivot(index="Number of vehicles", columns="Model", values="Proportion of cases").reindex([">3", "3", "2", "1"])

# 合并为带注释格式，例如 "76.7%\n(n=374)"
annot_combined = annot_df.copy()
for i in annot_df.index:
    for j in annot_df.columns:
        acc = annot_df.loc[i, j]
        count = cases_df.loc[i, j]
        annot_combined.loc[i, j] = f"{acc:.1%}\n(n={count:.1%})"

# 输出路径
new_model_order = ["Original", "721-step", "1442-step", "2163-step","GPT-4o"]

# 根据新的顺序重新排序数据
annot_df_ordered = annot_df[new_model_order]
cases_df_ordered = cases_df[new_model_order]

# 构造新的注释文本
annot_combined_ordered = annot_df_ordered.copy()
for i in annot_df_ordered.index:
    for j in annot_df_ordered.columns:
        acc = annot_df_ordered.loc[i, j]
        count = cases_df_ordered.loc[i, j]
        annot_combined_ordered.loc[i, j] = f"{acc:.1%}"

# 输出路径
output_path_reordered = "reports/crash-type-test/figure/accuracy_heatmap_reorderedqv.png"

# 绘制图像
plt.figure(figsize=(6, 4.5))
sns.heatmap(
    annot_df_ordered.astype(float),
    annot=annot_combined_ordered,
    fmt='',
    cmap="YlGnBu",
    vmin=0.0, vmax=0.85,
    linewidths=0.5,
    cbar_kws={"label": "Accuracy (%)"},annot_kws={"size": 14}
)

# plt.title("", fontsize=15)
plt.xlabel("Model", fontsize=15)
plt.ylabel("Number of Vehicles", fontsize=15)
plt.tight_layout()

# 保存图像
plt.savefig(output_path_reordered, dpi=300)
plt.close()

output_path_reordered
