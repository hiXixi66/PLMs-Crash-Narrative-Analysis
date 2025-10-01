import pandas as pd
import numpy as np

# Input and output file paths
input_file = "data/processed_data/case_info_2021.xlsx"
output_file = "data/processed_data/case_info_2021_15perc_noise.xlsx"

# Read the "CRASH" sheet from the Excel file
df = pd.read_excel(input_file, sheet_name="CRASH")

# Verify that the required columns exist
required_cols = ["CASEID", "SUMMARY", "MANCOLL"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Copy the MANCOLL column to a new column named MANCOLLNEW
df["MANCOLLNEW"] = df["MANCOLL"]

# Possible MANCOLL values
possible_values = [0, 1, 2, 4, 5, 6, 9]

# Randomly sample 15% of the rows
n_samples = int(len(df) * 0.15)
sample_indices = np.random.choice(df.index, size=n_samples, replace=False)

# Randomly change MANCOLLNEW for the sampled rows
for idx in sample_indices:
    current_value = df.at[idx, "MANCOLL"]
    other_values = [v for v in possible_values if v != current_value]
    new_value = np.random.choice(other_values)
    df.at[idx, "MANCOLLNEW"] = new_value

# Keep only the necessary columns
df_out = df[["CASEID", "SUMMARY", "MANCOLL", "MANCOLLNEW"]]

# Save the result to a new Excel file
df_out.to_excel(output_file, index=False)

print(f"Processing complete. New file saved to: {output_file}")
