import os
import re
import time
import random
import json
import numpy as np
import pandas as pd

from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sklearn_crfsuite

# ======== 配置 ========
SEED = 42
EXCEL_PATH = "data/processed_data/case_info_2021.xlsx"
TEST_PATH = "data/processed_data/case_info_2020.xlsx"

TEXT_COL = "SUMMARY"
LABEL_COL = "CRASHTYPE"
VAL_SIZE = 0.1
TEST_SIZE = 0.1

OUTPUT_DIR = "./crashtype_crf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== 随机种子 ========
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seed()

# ======== 按 case 中车辆数量分组，只统计 accuracy ========
def eval_accuracy_by_vehicle_group(df_crash, df_dataset, model):
    if df_dataset is None or len(df_dataset) == 0:
        return

    caseid_to_vehicles = df_crash.set_index("CASEID")["VEHICLES"].to_dict()

    df_eval = df_dataset.copy()
    df_eval["VEHICLES"] = df_eval["CASEID"].map(caseid_to_vehicles)
    df_eval = df_eval.dropna(subset=["VEHICLES"]).copy()
    df_eval["VEHICLES"] = df_eval["VEHICLES"].astype(int)

    X_eval = [text_to_features(t) for t in df_eval[TEXT_COL].astype(str).tolist()]
    y_true = df_eval[LABEL_COL].values
    y_pred_seq = model.predict(X_eval)
    y_pred = np.array([seq_to_sentence_label(seq) for seq in y_pred_seq])

    df_eval["correct"] = (y_true == y_pred).astype(int)

    groups = {
        "VEHICLES=1": df_eval[df_eval["VEHICLES"] == 1],
        "VEHICLES=2": df_eval[df_eval["VEHICLES"] == 2],
        "VEHICLES=3": df_eval[df_eval["VEHICLES"] == 3],
        "VEHICLES>3": df_eval[df_eval["VEHICLES"] > 3],
    }

    print("\n== Accuracy by vehicle count group ==")
    results = {}
    for name, g in groups.items():
        if len(g) == 0:
            print(f"{name}: no samples")
            results[name] = None
        else:
            acc = g["correct"].mean()
            print(f"{name}: {acc:.4f}")
            results[name] = float(acc)
    return results

# ======== 文本替换函数 ========
def replace_vehicle_reference(label_id: int, text: str) -> str:
    pattern = fr'\b(V#{label_id}|V{label_id}|Vehicle #{label_id}|Vehicle {label_id})\b'
    return re.sub(pattern, 'the vehicle to be classified', text, flags=re.IGNORECASE)

# ======== 构造逐车辆样本 ========
def build_examples_from_crash_and_gv(df_crash, df_gv, text_col="SUMMARY"):
    records = []
    for _, row in df_gv.iterrows():
        caseid = row["CASEID"]
        vehno = row["VEHNO"]
        crashtype = row["CRASHTYPE"]

        crash_row = df_crash[df_crash["CASEID"] == caseid]
        if crash_row.empty:
            continue

        summary = crash_row.iloc[0][text_col]
        text = replace_vehicle_reference(vehno, str(summary))

        records.append({
            "CASEID": caseid,
            "VEHNO": vehno,
            "SUMMARY": text,
            "CRASHTYPE": crashtype
        })
    return pd.DataFrame(records)

# ======== 分词和 CRF 特征 ========
def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"\b\w+\b", text)
    if len(tokens) == 0:
        tokens = ["<empty>"]
    return tokens

def word2features(sent, i):
    word = sent[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word_len": len(word),
    }

    if i > 0:
        word1 = sent[i - 1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
        })
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
        })
    else:
        features["EOS"] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def text_to_features(text):
    tokens = tokenize(text)
    return sent2features(tokens)

def text_to_labels(text, label):
    tokens = tokenize(text)
    return [str(label)] * len(tokens)

def seq_to_sentence_label(pred_seq):
    counter = Counter(pred_seq)
    pred_label = counter.most_common(1)[0][0]
    return int(pred_label)

# ======== 单条 case 推理函数 ========
def predict_case(df_crash, caseid, model, id2label):
    crash_row = df_crash[df_crash["CASEID"] == caseid]
    if crash_row.empty:
        return []

    summary = crash_row.iloc[0]["SUMMARY"]
    vehnos = int(crash_row.iloc[0]["VEHICLES"])

    preds = []
    for vehno in range(1, vehnos + 1):
        text = replace_vehicle_reference(vehno, str(summary))
        x = text_to_features(text)
        pred_seq = model.predict_single(x)
        pred = seq_to_sentence_label(pred_seq)
        preds.append({
            "CASEID": caseid,
            "VEHNO": vehno,
            "PRED_CRASHTYPE": id2label[int(pred)]
        })
    return preds

# ======== case-level 评估并导出 ========
def eval_and_print_crf(name, df_crash, df_gv, df_dataset, model, label2id, output_dir):
    if df_dataset is None or len(df_dataset) == 0:
        return None

    time_start = time.time()

    X_data = [text_to_features(t) for t in df_dataset[TEXT_COL].astype(str).tolist()]
    y_true = df_dataset[LABEL_COL].values
    y_pred_seq = model.predict(X_data)
    y_pred = np.array([seq_to_sentence_label(seq) for seq in y_pred_seq])

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    print(f"\n== {name} ==")
    print(f"Accuracy (all vehicles): {acc:.4f}")
    print(f"F1-macro (all vehicles): {f1m:.4f}")

    records = []
    for caseid in df_gv["CASEID"].unique():
        crash_row = df_crash[df_crash["CASEID"] == caseid]
        if crash_row.empty:
            continue

        total = int(crash_row.iloc[0]["VEHICLES"])
        summary = crash_row.iloc[0]["SUMMARY"]

        vehs = df_gv[df_gv["CASEID"] == caseid]
        correct = 0

        for _, vrow in vehs.iterrows():
            vehno = vrow["VEHNO"]
            true_label = label2id.get(vrow["CRASHTYPE"], None)
            if true_label is None:
                continue

            matched = df_dataset[
                (df_dataset["CASEID"] == caseid) & (df_dataset["VEHNO"] == vehno)
            ]
            if matched.empty:
                continue

            idx = matched.index[0]
            pred_label = y_pred[df_dataset.index.get_loc(idx)]

            if pred_label == true_label:
                correct += 1

        acc_case = correct / total if total > 0 else 0.0
        records.append({
            "CASEID": caseid,
            "vehicles": total,
            "SUMMARY": summary,
            "case_accuracy": acc_case
        })

    result_df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, "case_level_results.xlsx")
    result_df.to_excel(output_path, index=False)

    time_end = time.time()
    print(f"\nCase-level results saved to {output_path}")
    print(f"Time elapsed: {time_end - time_start:.2f} sec")

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m)
    }

# ======== 读取数据 ========
df_crash = pd.read_excel(EXCEL_PATH, sheet_name="CRASH")
df_gv = pd.read_excel(EXCEL_PATH, sheet_name="GV")
df = build_examples_from_crash_and_gv(df_crash, df_gv, text_col="SUMMARY")

df_crash_test = pd.read_excel(TEST_PATH, sheet_name="CRASH")
df_gv_test = pd.read_excel(TEST_PATH, sheet_name="GV")
df_test_extra = build_examples_from_crash_and_gv(df_crash_test, df_gv_test, text_col="SUMMARY")

# 清理空值
df = df[["SUMMARY", "CRASHTYPE", "CASEID", "VEHNO"]].dropna().reset_index(drop=True)
df_test_extra = df_test_extra[["SUMMARY", "CRASHTYPE", "CASEID", "VEHNO"]].dropna().reset_index(drop=True)[:2000]

# ======== 划分训练/验证 ========
if TEST_SIZE > 0:
    df_trainval, df_test_unused = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED
    )
else:
    df_trainval, df_test_unused = df, None

df_train, df_val = train_test_split(
    df_trainval, test_size=VAL_SIZE, random_state=SEED
)

# ======== 标签映射：只用 train 的类别，避免不连续 ========
train_unique_labels = sorted(df_train[LABEL_COL].dropna().unique().tolist())
label2id = {label: idx for idx, label in enumerate(train_unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(train_unique_labels)

print("Number of labels:", num_labels)
print("Label to ID mapping:", label2id)

df_train = df_train[df_train[LABEL_COL].isin(label2id)].copy()
df_val = df_val[df_val[LABEL_COL].isin(label2id)].copy()
df_test_extra = df_test_extra[df_test_extra[LABEL_COL].isin(label2id)].copy()

df_train[LABEL_COL] = df_train[LABEL_COL].map(label2id).astype(int)
df_val[LABEL_COL] = df_val[LABEL_COL].map(label2id).astype(int)
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].map(label2id).astype(int)

print("Train size:", len(df_train))
print("Val size:", len(df_val))
print("External test size:", len(df_test_extra))

# ======== 构造 CRF 数据 ========
print("Building CRF features...")
X_train = [text_to_features(t) for t in df_train[TEXT_COL].astype(str).tolist()]
y_train = [text_to_labels(t, l) for t, l in zip(df_train[TEXT_COL].astype(str).tolist(), df_train[LABEL_COL].values)]

X_val = [text_to_features(t) for t in df_val[TEXT_COL].astype(str).tolist()]
y_val_sentence = df_val[LABEL_COL].values

print("Num train samples:", len(X_train))
print("Num val samples:", len(X_val))

# ======== 超参数搜索 ========
print("\nStarting hyperparameter search...")
search_start = time.time()


param_grid = {
    "algorithm": ["lbfgs"],
    "c1": [0.1],
    "c2": [ 0.5],
    "max_iterations": [100],
    "all_possible_transitions": [True],

}

all_settings = list(product(
    param_grid["algorithm"],
    param_grid["c1"],
    param_grid["c2"],
    param_grid["max_iterations"],
    param_grid["all_possible_transitions"],
))

best_score = -1.0
best_acc = -1.0
best_params = None
search_results = []

for i, (algorithm, c1, c2, max_iterations, all_possible_transitions) in enumerate(all_settings, 1):
    params = {
        "algorithm": algorithm,
        "c1": c1,
        "c2": c2,
        "max_iterations": max_iterations,
        "all_possible_transitions": all_possible_transitions,
    }

    print(f"\n[{i}/{len(all_settings)}] Trying params: {params}")
    one_start = time.time()

    model = sklearn_crfsuite.CRF(
        algorithm=algorithm,
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions,
    )

    model.fit(X_train, y_train)

    val_pred_seq = model.predict(X_val)
    val_pred = np.array([seq_to_sentence_label(seq) for seq in val_pred_seq])

    val_acc = accuracy_score(y_val_sentence, val_pred)
    val_f1 = f1_score(y_val_sentence, val_pred, average="macro")

    elapsed = time.time() - one_start
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val Macro-F1: {val_f1:.4f}")
    print(f"Elapsed: {elapsed:.2f} sec")

    result_row = params.copy()
    result_row["val_accuracy"] = float(val_acc)
    result_row["val_f1_macro"] = float(val_f1)
    result_row["time_sec"] = float(elapsed)
    search_results.append(result_row)

    if (val_f1 > best_score) or (val_f1 == best_score and val_acc > best_acc):
        best_score = val_f1
        best_acc = val_acc
        best_params = params

print("\nHyperparameter search finished.")
print("Best params:", best_params)
print(f"Best val accuracy: {best_acc:.4f}")
print(f"Best val macro-F1: {best_score:.4f}")
print(f"Search time: {time.time() - search_start:.2f} sec")

search_df = pd.DataFrame(search_results).sort_values(
    by=["val_f1_macro", "val_accuracy"], ascending=False
)
search_csv_path = os.path.join(OUTPUT_DIR, "hyperparam_search_results.csv")
search_df.to_csv(search_csv_path, index=False)
print(f"Saved search results to {search_csv_path}")

# ======== 用最佳参数在 train+val 上重训 ========
print("\nRetraining best model on train+val...")
df_train_final = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

X_train_final = [text_to_features(t) for t in df_train_final[TEXT_COL].astype(str).tolist()]
y_train_final = [text_to_labels(t, l) for t, l in zip(df_train_final[TEXT_COL].astype(str).tolist(), df_train_final[LABEL_COL].values)]

final_model = sklearn_crfsuite.CRF(
    algorithm=best_params["algorithm"],
    c1=best_params["c1"],
    c2=best_params["c2"],
    max_iterations=best_params["max_iterations"],
    all_possible_transitions=best_params["all_possible_transitions"],
)

final_train_start = time.time()
final_model.fit(X_train_final, y_train_final)
final_train_time = time.time() - final_train_start
print(f"Final training time: {final_train_time:.2f} sec")

# ======== 在 external test 上评估 ========
print("\nEvaluating on external test...")
X_test = [text_to_features(t) for t in df_test_extra[TEXT_COL].astype(str).tolist()]
y_test = df_test_extra[LABEL_COL].values

test_pred_seq = final_model.predict(X_test)
test_pred = np.array([seq_to_sentence_label(seq) for seq in test_pred_seq])

test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average="macro")

print("\n== External Test ==")
print(f"Accuracy (all vehicles): {test_acc:.4f}")
print(f"F1-macro (all vehicles): {test_f1:.4f}")

# ======== case-level 评估并导出 ========
test_metrics = eval_and_print_crf(
    "Test",
    df_crash_test,
    df_gv_test,
    df_test_extra,
    final_model,
    label2id,
    OUTPUT_DIR
)

# ======== 按车辆数量分组，只输出正确率 ========
group_metrics = eval_accuracy_by_vehicle_group(
    df_crash_test,
    df_test_extra,
    final_model
)

# ======== 保存摘要 ========
summary = {
    "num_labels": num_labels,
    "train_size": int(len(df_train)),
    "val_size": int(len(df_val)),
    "external_test_size": int(len(df_test_extra)),
    "best_params": best_params,
    "best_val_accuracy": float(best_acc),
    "best_val_f1_macro": float(best_score),
    "final_train_time_sec": float(final_train_time),
    "external_test_accuracy": float(test_acc),
    "external_test_f1_macro": float(test_f1),
    "group_accuracy": group_metrics,
}

summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Saved summary to {summary_path}")