import os
import re
import glob
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# ========== 配置 ==========
ROOT_DIR = "mancoll_bert2"
FILE_GLOB = os.path.join(ROOT_DIR, "*.xlsx")

# ========== 工具函数 ==========
def read_one(fp: str) -> pd.DataFrame:
    df = pd.read_excel(fp, engine="openpyxl",nrows=200)
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in df.columns: return c
            if c.lower() in cols_lower: return cols_lower[c.lower()]
        raise KeyError(f"{fp} not found column in {cands}, has {list(df.columns)}")
    s_sum = pick("SUMMARY", "summary")
    s_gt  = pick("MANCOLL", "mancoll", "label")
    s_pr  = pick("collision_type", "pred", "prediction", "output")
    out = df[[s_sum, s_gt, s_pr]].copy()
    out.columns = ["SUMMARY", "MANCOLL", "PRED"]
    out["SUMMARY"] = out["SUMMARY"].astype(str).str.strip()
    out["MANCOLL"] = out["MANCOLL"].astype(str).str.strip()
    out["PRED"]    = out["PRED"].astype(str).str.strip()
    return out

def base_key_from_filename(fn: str):
    name = os.path.splitext(os.path.basename(fn))[0]
    name = re.sub(r"^bert_test_results_", "", name)
    
    # 匹配末尾的 -数字
    m = re.search(r"-(\d+)$", name)
    if m:
        run_id = int(m.group(1))
        base = name[:m.start()]
    else:
        run_id = -1
        base = name
    return base, run_id

def majority_vote(labels):
    cnt = Counter(labels)
    lab, c = cnt.most_common(1)[0]
    return lab, c / len(labels)

def accuracy(gt, pred):
    gt = np.array(gt); pred = np.array(pred)
    mask = ~pd.isna(gt) & ~pd.isna(pred)
    if mask.sum() == 0: return np.nan
    return (gt[mask] == pred[mask]).mean()

# ========== 读取并按模型聚合 ==========
files = sorted(glob.glob(FILE_GLOB))
if not files:
    raise FileNotFoundError(f"No .xlsx under {ROOT_DIR}")

by_model = defaultdict(list)
for fp in files:
    base, rid = base_key_from_filename(fp)
    df = read_one(fp)
    by_model[base].append((rid, df))

summary_rows = []

# 画图依赖
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=1.1)

for model_key, runs in sorted(by_model.items()):
    runs = sorted(runs, key=lambda x: x[0])
    run_tags = []
    merged = None

    for rid, df in runs:
        tag = f"run{ rid}" if rid >= 0 else f"run"
        run_tags.append(tag)
        if merged is None:
            merged = df.rename(columns={"PRED": tag})
        else:
            merged = merged.merge(df.rename(columns={"PRED": tag}),
                                  on=["SUMMARY", "MANCOLL"], how="outer")

    def row_vote(row):
        preds = [row[t] for t in run_tags if pd.notna(row[t])]
        if len(preds) == 0:
            return pd.Series({"maj_pred": np.nan, "consistency": np.nan})
        lab, ratio = majority_vote(preds)
        return pd.Series({"maj_pred": lab, "consistency": ratio})

    vote_df = merged.apply(row_vote, axis=1)
    merged = pd.concat([merged, vote_df], axis=1)

    overall_sc_incl9 = merged["consistency"].mean()
    mask_ex9 = merged["MANCOLL"] != "9"
    overall_sc_excl9 = merged.loc[mask_ex9, "consistency"].mean()

    acc_incl9 = accuracy(merged["MANCOLL"], merged["maj_pred"])
    acc_excl9 = accuracy(merged.loc[mask_ex9, "MANCOLL"], merged.loc[mask_ex9, "maj_pred"])

    # 两两一致率矩阵
    pair = pd.DataFrame(index=run_tags, columns=run_tags, dtype=float)
    for a in run_tags:
        for b in run_tags:
            if a == b:
                pair.loc[a, b] = 1.0
            else:
                both = merged[[a, b]].dropna()
                pair.loc[a, b] = np.nan if len(both) == 0 else (both[a] == both[b]).mean()

    # ===== 导出 CSV =====
    out_dir = ROOT_DIR
    merged_out = os.path.join(out_dir, f"{model_key}_self_consistency.csv")
    pair_out   = os.path.join(out_dir, f"{model_key}_pairwise_agreement.csv")
    merged[["SUMMARY", "MANCOLL", *run_tags, "maj_pred", "consistency"]].to_csv(merged_out, index=False)
    pair.to_csv(pair_out)

    # ===== 画热力图（heatmap）=====
    plt.figure(figsize=(8, 5.5))
    ax = sns.heatmap(
        pair.astype(float),
        mask=pair.isna(),
        cmap="YlGnBu",
        vmin=0.95, vmax=1,  # 一致率[0,1]
        annot=True, fmt=".2f",
        cbar_kws={"label": "Exact Match Rate"},
        square=True
    )
    ax.set_title(f"Self-Consistency", pad=10)
    ax.set_xlabel("")  # 轴标题留空更干净
    ax.set_ylabel("")
    plt.xticks(rotation=15, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{model_key}_pairwise_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    summary_rows.append({
        "model": model_key,
        "num_runs": len(run_tags),
        "self_consistency_incl9": overall_sc_incl9,
        "self_consistency_excl9": overall_sc_excl9,
        "majority_acc_incl9": acc_incl9,
        "majority_acc_excl9": acc_excl9
    })

# 汇总表
summary_df = pd.DataFrame(summary_rows).sort_values(by="self_consistency_incl9", ascending=False)
summary_df.to_csv(os.path.join(ROOT_DIR, "summary_self_consistency.csv"), index=False)
print(summary_df)
print(f"Saved to: {ROOT_DIR} (pairwise CSV + heatmaps)")
