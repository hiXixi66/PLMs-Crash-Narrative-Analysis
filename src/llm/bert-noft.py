import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

EXCEL_PATH = "data/processed_data/case_info_2020.xlsx"
df = pd.read_excel(EXCEL_PATH)  # For example, contains ["SUMMARY", "LABEL"]
texts = df["SUMMARY"].astype(str).tolist()
labels = df["MANCOLL"].tolist()  # Labels are 0, 1, 2, 4, 5, 6, 9

# 2. Load pre-trained BERT (no fine-tuning)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Evaluation mode, no parameter updates

# 3. Define feature extraction function
def get_bert_embeddings(texts, batch_size=16, max_len=128):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            outputs = model(**inputs)
            # [CLS] vector
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.extend(cls_embeddings.numpy())
    return all_embeddings

# 4. Extract BERT vectors for SUMMARY
X = get_bert_embeddings(texts)

# 5. Split into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# 6. Use Logistic Regression for classification
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# 7. Testing & evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# Original evaluation on all labels
y_pred = clf.predict(X_test)
print("=== All labels (including 9) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred, digits=3))

# Remove samples with label 9
mask = np.array(y_test) != 9
y_test_filtered = np.array(y_test)[mask]
y_pred_filtered = np.array(y_pred)[mask]

print("\n=== After removing label 9 ===")
print("Accuracy:", accuracy_score(y_test_filtered, y_pred_filtered))
print("Macro-F1:", f1_score(y_test_filtered, y_pred_filtered, average="macro"))
print(classification_report(y_test_filtered, y_pred_filtered, digits=3))
