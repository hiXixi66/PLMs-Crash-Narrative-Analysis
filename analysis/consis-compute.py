import os
import glob
import hashlib
import pandas as pd
import numpy as np

# =========================
# Configuration
# =========================
ROOT_DIR = "reports/mancolll-test/cons"

# Convert file name to a model tag (e.g., "MANCOLL-classification_llama3b-417.xlsx" → "MANCOLL-classification_llama3b-417")
def file_to_model_tag(path: str) -> str:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return name[2:]  # Remove the first two characters

# =========================
# Read all .xlsx files
# =========================
xlsx_files = sorted(glob.glob(os.path.join(ROOT_DIR, "*.xlsx")))
if not xlsx_files:
    raise FileNotFoundError(f"No .xlsx found under: {ROOT_DIR}")

frames = []

for fp in xlsx_files:
    tag = file_to_model_tag(fp)
    # Use openpyxl engine for better compatibility
    df = pd.read_excel(fp, engine="openpyxl")

    # Column name tolerance (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}

    def safe_to_str(x):
        if isinstance(x, str):
            return x.strip()
        elif isinstance(x, float) and x.is_integer():  # Handle float integers
            return str(int(x))
        elif isinstance(x, (int, np.integer)):
            return str(x)
        else:
            return str(x).strip()

    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        raise KeyError(f"Column {cands} not found in {fp}. Found columns: {list(df.columns)}")

    col_summary = pick("SUMMARY", "summary")
    col_mancoll = pick("MANCOLL", "mancoll", "label", "labels")
    col_pred    = pick("collision_type", "pred", "prediction", "output")

    tmp = df[[col_summary, col_mancoll, col_pred]].copy()
    tmp.columns = ["SUMMARY", "MANCOLL", tag]
    # Convert to string and strip spaces
    tmp["SUMMARY"] = tmp["SUMMARY"].astype(str).str.strip()
    tmp["MANCOLL"] = tmp["MANCOLL"].apply(safe_to_str)
    tmp[tag]       = tmp[tag].apply(safe_to_str)
    frames.append(tmp)

# =========================
# Merge all files into a wide table
# =========================
merged = frames[0]
for t in frames[1:]:
    merged = merged.merge(t, on=["SUMMARY", "MANCOLL"], how="outer")

# Create a sample ID (based on hash of SUMMARY + MANCOLL)
def hash_id(row) -> str:
    s = (row["SUMMARY"] or "") + "||" + (row["MANCOLL"] or "")
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

merged.insert(0, "sample_id", merged.apply(hash_id, axis=1))

# Identify model columns and GT column
model_cols = [c for c in merged.columns if c not in ["sample_id", "SUMMARY", "MANCOLL"]]
merged["GT"] = merged["MANCOLL"].astype(str).str.strip()
models_plus = model_cols + ["GT"]

# =========================
# Pairwise exact match (consistency)
# =========================
def pairwise_consistency(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                out.loc[a, b] = 1.0
            else:
                both = df[[a, b]].dropna()
                out.loc[a, b] = np.nan if len(both) == 0 else (both[a] == both[b]).mean()
    return out

pairwise = pairwise_consistency(merged, model_cols)          # Model ↔ Model
pairwise_with_gt = pairwise_consistency(merged, models_plus) # Model + GT

# Accuracy of each model against GT
acc_vs_gt = pairwise_with_gt.loc[model_cols, "GT"].rename("accuracy_vs_GT").to_frame()

# =========================
# Sample-level consistency (majority agreement)
# =========================
def majority_agreement(row: pd.Series, cols: list[str]) -> float:
    preds = [row[c] for c in cols if pd.notna(row[c])]
    if len(preds) <= 1:
        return np.nan
    vc = pd.Series(preds).value_counts()
    return vc.iloc[0] / len(preds)

# Models only
merged["sample_consistency"] = merged.apply(lambda r: majority_agreement(r, model_cols), axis=1)
overall_consistency = merged["sample_consistency"].mean()

# Models + GT
merged["sample_consistency_with_GT"] = merged.apply(lambda r: majority_agreement(r, models_plus), axis=1)
overall_consistency_with_GT = merged["sample_consistency_with_GT"].mean()

# Excluding class 9 (based on GT)
mask_excl9 = merged["GT"] != "9"
overall_excl9             = merged.loc[mask_excl9, "sample_consistency"].mean()
overall_excl9_with_GT     = merged.loc[mask_excl9, "sample_consistency_with_GT"].mean()

# =========================
# (Optional) Fleiss' kappa for multi-rater agreement
# =========================
fleiss_kappa = None
try:
    from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa
    labels = pd.unique(pd.concat([merged[c] for c in model_cols], ignore_index=True).dropna())
    label_to_idx = {lab: i for i, lab in enumerate(sorted(labels, key=str))}
    rows = []
    for _, r in merged.iterrows():
        preds = [r[c] for c in model_cols if pd.notna(r[c])]
        if len(preds) <= 1:
            continue
        cnt = np.zeros(len(label_to_idx), dtype=int)
        for p in preds:
            if p in label_to_idx:
                cnt[label_to_idx[p]] += 1
        rows.append(cnt)
    if rows:
        table = np.vstack(rows)
        fleiss_kappa = _fleiss_kappa(table)
except Exception:
    pass  # Skip if statsmodels is not installed or any error occurs

# =========================
# Export results
# =========================
pairwise_out         = os.path.join(ROOT_DIR, "consistency_matrix.csv")
pairwise_with_gt_out = os.path.join(ROOT_DIR, "consistency_matrix_with_GT.csv")
acc_vs_gt_out        = os.path.join(ROOT_DIR, "accuracy_vs_GT.csv")
samples_out          = os.path.join(ROOT_DIR, "sample_consistency.csv")
merged_out           = os.path.join(ROOT_DIR, "merged_predictions.parquet")

pairwise.to_csv(pairwise_out, index=True)
pairwise_with_gt.to_csv(pairwise_with_gt_out, index=True)
acc_vs_gt.to_csv(acc_vs_gt_out)
merged[["sample_id", "MANCOLL", "GT", *model_cols, "sample_consistency",
        "sample_consistency_with_GT"]].to_csv(samples_out, index=False)
merged.to_parquet(merged_out, index=False)

print(f"Loaded files: {len(xlsx_files)}")
print(f"Models ({len(model_cols)}): {model_cols}")
print(f"Overall consistency (models only): {overall_consistency:.4f}")
print(f"Overall consistency (with GT): {overall_consistency_with_GT:.4f}")
print(f"Overall consistency Excl. 9 (models only): {overall_excl9:.4f}")
print(f"Overall consistency Excl. 9 (with GT): {overall_excl9_with_GT:.4f}")
if fleiss_kappa is not None:
    print(f"Fleiss' kappa (models only): {fleiss_kappa:.4f}")
else:
    print("Fleiss' kappa not computed (install statsmodels to enable).")

# =========================
# Plotting (larger fonts for readability)
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# Increase font sizes globally
sns.set_context("talk", font_scale=1.5)
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18

# Figure 1: Cross-model consistency heatmap (models only)
plt.figure(figsize=(11, 9))
sns.heatmap(pairwise.astype(float), annot=True, fmt=".2f", cmap="YlGnBu",
            cbar_kws={'label': 'Exact Match Rate'}, annot_kws={"size": 18})
plt.title("Cross-Model Consistency Heatmap (Models Only)")
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "cross_consistency_heatmap_models_only.png"), dpi=300)
plt.close()

# Figure 1b: Cross-model + GT consistency heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pairwise_with_gt.astype(float), annot=True, fmt=".2f", cmap="YlGnBu",
            cbar_kws={'label': 'Exact Match Rate'}, annot_kws={"size": 18})
plt.title("Cross-Model Consistency Heatmap")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "cross_consistency_heatmap_with_GT.png"), dpi=300)
plt.close()

# Figure 2: Sample-level consistency distribution (models only vs models+GT)
plt.figure(figsize=(10, 8))
sns.histplot(merged["sample_consistency"].dropna(), bins=20, kde=True, label="Models only", alpha=0.8)
sns.histplot(merged["sample_consistency_with_GT"].dropna(), bins=20, kde=True, label="Models + GT", alpha=0.5)
plt.xlabel("Sample-level Majority Agreement")
plt.ylabel("Number of Samples")
plt.title("Distribution of Sample Consistency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "sample_consistency_distribution_with_GT.png"), dpi=300)
plt.close()

print("Saved figures:",
      os.path.join(ROOT_DIR, "cross_consistency_heatmap_models_only.png"), ",",
      os.path.join(ROOT_DIR, "cross_consistency_heatmap_with_GT.png"), ",",
      os.path.join(ROOT_DIR, "sample_consistency_distribution_with_GT.png"))
