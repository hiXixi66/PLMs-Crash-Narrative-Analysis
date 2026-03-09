import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# ===============================
# 1. 读取数据
# ===============================
file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/data/processed_data/case_info_2021.xlsx"
test_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/data/processed_data/case_info_2020.xlsx"

df = pd.read_excel(file_path, sheet_name="CRASH")

texts = df["SUMMARY"].astype(str).tolist()
labels = df["MANCOLL"].astype(int).tolist()

unique_labels = sorted(set(labels))
label2id = {v: i for i, v in enumerate(unique_labels)}
id2label = {i: v for v, i in label2id.items()}

labels = [label2id[l] for l in labels]
num_classes = len(unique_labels)

# train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# ===============================
# 2. TF-IDF 特征提取
# ===============================
print("Building TF-IDF features...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=20000,
    ngram_range=(1, 2),   # unigram + bigram
    min_df=2
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# ===============================
# 3. 定义并训练 XGBoost 模型
# ===============================
print("Starting XGBoost training...")
start_time = time.time()
print("start_time:", start_time)

model = XGBClassifier(
    objective="multi:softmax",   # 直接输出类别
    num_class=num_classes,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
)

model.fit(X_train, train_labels)

print("XGBoost training finished.")
print("end_time:", time.time())
print("training_time:", time.time() - start_time)

# ===============================
# 4. 在验证集上评估
# ===============================
val_preds = model.predict(X_val)

acc = accuracy_score(val_labels, val_preds)
macro_f1 = f1_score(val_labels, val_preds, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# ===============================
# 5. 在测试集上评估
# ===============================
print("Evaluating on test set...")
df_test = pd.read_excel(test_path, sheet_name="CRASH")
test_texts = df_test["SUMMARY"].astype(str).tolist()[:3500]
test_labels = df_test["MANCOLL"].astype(int).tolist()[:3500]

# 用训练集的 label2id 映射标签
test_labels = [label2id.get(l, -1) for l in test_labels]

# 过滤掉不在训练类别里的样本
test_pairs = [(t, l) for t, l in zip(test_texts, test_labels) if l != -1]

if len(test_pairs) == 0:
    raise ValueError("测试集中没有可用样本：所有标签都不在训练集标签空间中。")

test_texts, test_labels = zip(*test_pairs)
test_true = list(test_labels)

print("test start_time:", time.time())

X_test = vectorizer.transform(test_texts)
test_preds = model.predict(X_test)

test_acc = accuracy_score(test_true, test_preds)
test_macro_f1 = f1_score(test_true, test_preds, average="macro")

print(f"[Test] Accuracy: {test_acc:.4f}")
print(f"[Test] Macro F1: {test_macro_f1:.4f}")
print("test end_time:", time.time())

# ===============================
# 6. 排除最后一个类别后的评估
# ===============================
last_class = max(test_true)
mask = [y != last_class for y in test_true]

filtered_true = [y for y, m in zip(test_true, mask) if m]
filtered_preds = [y for y, m in zip(test_preds, mask) if m]

test_acc_excl = accuracy_score(filtered_true, filtered_preds)
test_macro_f1_excl = f1_score(filtered_true, filtered_preds, average="macro")

print(f"[Test excl. last class={last_class}] Accuracy: {test_acc_excl:.4f}")
print(f"[Test excl. last class={last_class}] Macro F1: {test_macro_f1_excl:.4f}")