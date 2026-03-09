import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sklearn_crfsuite
import time
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
# 2. 分词
# ===============================
def tokenize(text):
    return text.lower().split()

# ===============================
# 3. 构造 CRF 特征
#    CRF 需要每个 token 的特征字典
# ===============================
def word2features(sent, i):
    word = sent[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.isdigit()": word.isdigit(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word_len": len(word),
    }

    if i > 0:
        word1 = sent[i - 1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.isdigit()": word1.isdigit(),
            "-1:word.isupper()": word1.isupper(),
        })
    else:
        features["BOS"] = True  # beginning of sentence

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.isdigit()": word1.isdigit(),
            "+1:word.isupper()": word1.isupper(),
        })
    else:
        features["EOS"] = True  # end of sentence

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# 把单标签分类任务“伪装”成序列标注：
# 每个 token 都赋同一个标签
def text_to_crf_sample(text, label):
    tokens = tokenize(text)
    if len(tokens) == 0:
        tokens = ["<EMPTY>"]
    x = sent2features(tokens)
    y = [str(label)] * len(tokens)
    return x, y

# ===============================
# 4. Dataset 构造
# ===============================
X_train, y_train = [], []
for text, label in zip(train_texts, train_labels):
    x, y = text_to_crf_sample(text, label)
    X_train.append(x)
    y_train.append(y)

X_val, y_val = [], []
for text, label in zip(val_texts, val_labels):
    x, y = text_to_crf_sample(text, label)
    X_val.append(x)
    y_val.append(y)

# ===============================
# 5. 定义并训练 CRF 模型
# ===============================
print("Starting CRF training...")
start_time = time.time()
print("start_time:", start_time)
crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)
print("CRF training finished.")
print("end_time:", time.time())
# ===============================
# 6. 评估
#    token 级预测 -> 聚合成句子级预测
# ===============================
def seq_to_sentence_label(pred_seq):
    # 因为训练时每个 token 都是同一个标签
    # 所以这里用多数投票还原为句子级标签
    counter = Counter(pred_seq)
    pred_label = counter.most_common(1)[0][0]
    return int(pred_label)

val_pred_seqs = crf.predict(X_val)

all_preds = [seq_to_sentence_label(seq) for seq in val_pred_seqs]
all_labels = val_labels

acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# ===============================
# 7. 在测试集上评估
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
print("test start_time:", time.time()  )

X_test = []
for text in test_texts:
    tokens = tokenize(text)
    if len(tokens) == 0:
        tokens = ["<EMPTY>"]
    X_test.append(sent2features(tokens))

test_pred_seqs = crf.predict(X_test)

test_preds = [seq_to_sentence_label(seq) for seq in test_pred_seqs]
test_true = list(test_labels)

test_acc = accuracy_score(test_true, test_preds)
test_macro_f1 = f1_score(test_true, test_preds, average="macro")

print(f"[Test] Accuracy: {test_acc:.4f}")
print(f"[Test] Macro F1: {test_macro_f1:.4f}")
print("test end_time:", time.time()  )
# ===============================
# 8. 排除最后一个类别后的评估
# ===============================
last_class = max(test_true)
mask = [y != last_class for y in test_true]

filtered_true = [y for y, m in zip(test_true, mask) if m]
filtered_preds = [y for y, m in zip(test_preds, mask) if m]

test_acc_excl = accuracy_score(filtered_true, filtered_preds)
test_macro_f1_excl = f1_score(filtered_true, filtered_preds, average="macro")

print(f"[Test excl. last class={last_class}] Accuracy: {test_acc_excl:.4f}")
print(f"[Test excl. last class={last_class}] Macro F1: {test_macro_f1_excl:.4f}")