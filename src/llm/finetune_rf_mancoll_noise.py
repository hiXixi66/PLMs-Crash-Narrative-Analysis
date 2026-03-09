import pandas as pd
import time
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ===============================
# 1. 读取数据并循环不同噪声比例
# ===============================
# for i in [ 40, 30, 20, 10, 5]:
for i in [200,400,600,800,1200,1600,2000]:
    reset_seed(42)
    print("time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    file_path = f"data/processed_data/case_info_2021_10perc_noise.xlsx"
    test_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/data/processed_data/case_info_2020.xlsx"

    print(f"Training with {i}% noise")
    print(f"Loading data from {file_path}")

    df = pd.read_excel(file_path)
    texts = df["SUMMARY"].astype(str).tolist()[:i]
    labels = df["MANCOLL"].astype(int).tolist()[:i]
    # texts = df["SUMMARY"].astype(str).tolist()
    # labels = df["MANCOLLNEW"].astype(int).tolist()

    unique_labels = sorted(set(labels))
    label2id = {v: idx for idx, v in enumerate(unique_labels)}
    id2label = {idx: v for v, idx in label2id.items()}
    labels = [label2id[l] for l in labels]
    num_classes = len(unique_labels)

    print(f"Number of classes: {num_classes}, label2id: {label2id}")

    # train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # ===============================
    # 2. TF-IDF 特征
    # ===============================
    print("Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    # ===============================
    # 3. 定义随机森林模型
    # ===============================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )

    # ===============================
    # 4. 训练
    # ===============================
    print("Starting Random Forest training...")
    train_start = time.time()

    model.fit(X_train, train_labels)

    train_end = time.time()
    print("Random Forest training finished.")
    print(f"Training time: {train_end - train_start:.2f} sec")
    print("time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ===============================
    # 5. 验证集评估
    # ===============================
    val_preds = model.predict(X_val)

    acc = accuracy_score(val_labels, val_preds)
    macro_f1 = f1_score(val_labels, val_preds, average="macro")

    print(f"[Val] Accuracy: {acc:.4f}")
    print(f"[Val] Macro F1: {macro_f1:.4f}")

    # ===============================
    # 6. 在测试集上评估
    # ===============================
    df_test = pd.read_excel(test_path)
    test_texts = df_test["SUMMARY"].astype(str).tolist()
    test_labels = df_test["MANCOLL"].astype(int).tolist()

    # 用训练集的 label2id 映射标签
    test_labels = [label2id.get(l, -1) for l in test_labels]

    # 过滤掉不在训练类别里的样本
    test_pairs = [(t, l) for t, l in zip(test_texts, test_labels) if l != -1]
    test_texts, test_labels = zip(*test_pairs)

    X_test = vectorizer.transform(test_texts)

    test_start = time.time()
    test_preds = model.predict(X_test)
    test_end = time.time()

    test_true = list(test_labels)

    test_acc = accuracy_score(test_true, test_preds)
    test_macro_f1 = f1_score(test_true, test_preds, average="macro")

    print(f"[Test] Accuracy: {test_acc:.4f}")
    print(f"[Test] Macro F1: {test_macro_f1:.4f}")
    print(f"[Test] Inference time: {test_end - test_start:.2f} sec")

    # ===============================
    # 7. 排除最后一个类别后的评估
    # ===============================
    last_class = 6
    mask = [y != last_class for y in test_true]

    filtered_true = [y for y, m in zip(test_true, mask) if m]
    filtered_preds = [y for y, m in zip(test_preds, mask) if m]

    test_acc_excl = accuracy_score(filtered_true, filtered_preds)
    test_macro_f1_excl = f1_score(filtered_true, filtered_preds, average="macro")

    print(f"[Test excl. last class={last_class}] Accuracy: {test_acc_excl:.4f}")
    print(f"[Test excl. last class={last_class}] Macro F1: {test_macro_f1_excl:.4f}")
    print("time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("-" * 60)