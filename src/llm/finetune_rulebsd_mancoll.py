import pandas as pd
import re
import time
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
    text = str(text).lower()
    # 只保留字母数字，简单清洗
    tokens = re.findall(r"\b\w+\b", text)
    return tokens

# 可选：简单停用词
stopwords = {
    "the", "a", "an", "of", "to", "and", "in", "on", "at", "for", "with",
    "from", "by", "is", "was", "were", "are", "be", "been", "this", "that",
    "it", "as", "or", "into", "after", "before", "during", "vehicle", "driver"
}

def clean_tokens(tokens):
    return [t for t in tokens if t not in stopwords and len(t) > 1]

# ===============================
# 3. 从训练集自动抽取每个类别的规则关键词
#    核心思想：
#    - 统计每个类别中词出现频率
#    - 用 “类别内频率 / 全局频率” 作为词的区分度分数
#    - 每个类别保留 top-K 关键词作为规则
# ===============================
def build_keyword_rules(train_texts, train_labels, top_k=30, min_freq=3):
    global_counter = Counter()
    class_counters = defaultdict(Counter)

    for text, label in zip(train_texts, train_labels):
        tokens = clean_tokens(tokenize(text))
        global_counter.update(tokens)
        class_counters[label].update(tokens)

    class_keywords = {}

    for label in sorted(class_counters.keys()):
        scores = []
        for word, cnt in class_counters[label].items():
            if cnt < min_freq:
                continue
            # 区分度分数：类内词频 / 全局词频
            score = cnt / global_counter[word]
            scores.append((word, score, cnt, global_counter[word]))

        # 先按 score 排，再按类内出现次数排
        scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        class_keywords[label] = scores[:top_k]

    return class_keywords

print("Building rule base...")
rule_base = build_keyword_rules(train_texts, train_labels, top_k=30, min_freq=3)

# 打印每类前10个关键词，方便你人工检查
print("\nTop keywords for each class:")
for label in sorted(rule_base.keys()):
    preview = [w for w, s, c, g in rule_base[label][:10]]
    print(f"class {label}: {preview}")

# ===============================
# 4. 定义规则预测函数
#    对一条文本：
#    - 看它命中了每个类别多少关键词
#    - 用关键词权重累计得分
#    - 选得分最高的类别
#    - 如果一个词出现在多个类中，谁分高归谁
# ===============================
# 转成便于查找的结构
keyword_weight_by_class = {}
for label, items in rule_base.items():
    keyword_weight_by_class[label] = {w: s for w, s, c, g in items}

# 默认类别：训练集中最多的类别
default_class = Counter(train_labels).most_common(1)[0][0]
print("Default class:", default_class)

def predict_rule_based(text):
    tokens = clean_tokens(tokenize(text))
    token_counts = Counter(tokens)

    class_scores = {}

    for label in keyword_weight_by_class:
        score = 0.0
        for word, count in token_counts.items():
            if word in keyword_weight_by_class[label]:
                # 命中词的权重 * 出现次数
                score += keyword_weight_by_class[label][word] * count
        class_scores[label] = score

    # 如果所有类别都没命中任何规则，回退到默认类
    best_label = max(class_scores, key=class_scores.get)
    if class_scores[best_label] == 0:
        return default_class

    return best_label

# ===============================
# 5. 在验证集上评估
# ===============================
print("\nEvaluating on validation set...")
val_start_time = time.time()

val_preds = [predict_rule_based(text) for text in val_texts]

acc = accuracy_score(val_labels, val_preds)
macro_f1 = f1_score(val_labels, val_preds, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("val_time:", time.time() - val_start_time)

# ===============================
# 6. 在测试集上评估
# ===============================
print("\nEvaluating on test set...")
print("start_time eval:", time.time())
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

test_start_time = time.time()

test_preds = [predict_rule_based(text) for text in test_texts]

test_acc = accuracy_score(test_true, test_preds)
test_macro_f1 = f1_score(test_true, test_preds, average="macro")

print(f"[Test] Accuracy: {test_acc:.4f}")
print(f"[Test] Macro F1: {test_macro_f1:.4f}")
print("test_time:", time.time() - test_start_time)

# ===============================
# 7. 排除最后一个类别后的评估
# ===============================
last_class = max(test_true)
mask = [y != last_class for y in test_true]

filtered_true = [y for y, m in zip(test_true, mask) if m]
filtered_preds = [y for y, m in zip(test_preds, mask) if m]

test_acc_excl = accuracy_score(filtered_true, filtered_preds)
test_macro_f1_excl = f1_score(filtered_true, filtered_preds, average="macro")

print(f"[Test excl. last class={last_class}] Accuracy: {test_acc_excl:.4f}")
print(f"[Test excl. last class={last_class}] Macro F1: {test_macro_f1_excl:.4f}")