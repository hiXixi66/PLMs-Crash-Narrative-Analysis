import pandas as pd
import numpy as np

# 输入和输出路径
input_file = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/data/processed_data/case_info_2021.xlsx"
output_file = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/data/processed_data/case_info_2021_15perc_noise.xlsx"

# 读取 Excel 文件中 "CRASH" sheet
df = pd.read_excel(input_file, sheet_name="CRASH")

# 确认需要的列存在
required_cols = ["CASEID", "SUMMARY", "MANCOLL"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# 拷贝一份 MANCOLL 作为新列
df["MANCOLLNEW"] = df["MANCOLL"]

# 可选的 MANCOLL 值
possible_values = [0, 1, 2, 4, 5, 6, 9]

# 随机抽取 5% 的索引
n_samples = int(len(df) * 0.15)
sample_indices = np.random.choice(df.index, size=n_samples, replace=False)
# bias

sample_indices = np.random.choice(df.index, size=n_samples, replace=False)

# randomly change MANCOLLNEW for the sampled indices
for idx in sample_indices:
    current_value = df.at[idx, "MANCOLL"]
    other_values = [v for v in possible_values if v != current_value]
    new_value = np.random.choice(other_values)
    df.at[idx, "MANCOLLNEW"] = new_value

# bias
# for idx in sample_indices:
#     current_value = df.at[idx, "MANCOLLNEW"]
#     if current_value in [0, 1, 2, 4, 5, 6]:
#         df.at[idx, "MANCOLLNEW"] = 9

    

# 只保留需要的列
df_out = df[["CASEID", "SUMMARY", "MANCOLL", "MANCOLLNEW"]]

# 保存到新的 Excel
df_out.to_excel(output_file, index=False)

print(f"处理完成，新文件已保存到: {output_file}")
