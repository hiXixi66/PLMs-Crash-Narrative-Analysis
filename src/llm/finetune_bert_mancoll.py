import os
import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import time
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


SEED = 42
MODEL_NAME = "bert-base-uncased" 
MAX_LENGTH = 256
OUTPUT_DIR = "./mancoll_bert2"
EXCEL_PATH = "data/processed_data/case_info_2021.xlsx"  
test_path= "data/processed_data/case_info_2020.xlsx"       
TEXT_COL = "SUMMARY"
LABEL_COL = "MANCOLL"
VAL_SIZE = 0.1              
TEST_SIZE = 0.1                 


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


df = pd.read_excel(EXCEL_PATH)
df_test_extra = pd.read_excel(test_path)

df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)
df_test_extra = df_test_extra[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)
# ======== Label Encoding ========
# Original Category Set
unique_labels = sorted(int(x) for x in df[LABEL_COL].unique())
label2id = {orig: i for i, orig in enumerate(unique_labels)}

id2label = {i: orig for orig, i in label2id.items()}
num_labels = len(unique_labels)

df[LABEL_COL] = df[LABEL_COL].map(label2id)
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].map(label2id)
if df[LABEL_COL].isnull().any():
    raise ValueError("Some labels could not be mapped! Check your LABEL_COL values.")
if df_test_extra[LABEL_COL].isnull().any():
    raise ValueError("Some labels in test set could not be mapped! Check your LABEL_COL values.")
df[LABEL_COL] = df[LABEL_COL].astype(int)
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].astype(int)

if TEST_SIZE > 0:
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[LABEL_COL]
    )
else:
    df_trainval, df_test = df, None

df_train, df_val = train_test_split(
    df_trainval, test_size=VAL_SIZE, random_state=SEED, stratify=df_trainval[LABEL_COL]
)

# ======== Tokenizer and Dataset ========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length: int):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

train_ds = TextClsDataset(df_train, TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)
val_ds   = TextClsDataset(df_val,   TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)
# test_ds  = TextClsDataset(df_test,  TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH) if df_test is not None else None
test_ds  = TextClsDataset(df_test_extra,  TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH) if df_test is not None else None
# ======== Dealing with class imbalance (class weights) ========
class_counts = df_train[LABEL_COL].value_counts().sort_index().values
class_weights = (1.0 / (class_counts + 1e-9)) 
class_weights = class_weights / class_weights.sum() * num_labels
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Customize Trainer to inject class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss




# ======== Model ========
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,            # name or path of the pre-trained model
    num_labels=num_labels, # number of output classes
    id2label=id2label,     # mapping from class index -> original label value
    label2id=label2id,     # mapping from original label value -> class index
)

# ======== Evaluation metrics ========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)  # take the highest scoring class
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")  # equal weight for each class
    return {"accuracy": acc, "f1_macro": f1_macro}
# ======== Training arguments ========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=5,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ======== training ========
# trainer.train()
# trainer.save_model(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

from transformers import AutoModelForSequenceClassification

def eval_and_print(name, dataset):
    if dataset is None:
        return
# Reload the model from the saved checkpoint
    checkpoint_dir ="mancoll_bert2/checkpoint-845"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


    tmp_trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    time_start = time.time()
    preds = tmp_trainer.predict(dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    records = []
    print(f"\n== {name} ==")
    print(f"Accuracy (all): {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro (all): {f1_score(y_true, y_pred, average='macro'):.4f}")

    id2label2 = {
        0: "Rear-End",
        1: "Head-On",
        2: "Side-Swipe",
        3: "Angle",
        4: "Other",
        5: "Parked",
        6: "Unknown"
    }

    print(classification_report(
        y_true,
        y_pred,
        target_names=[id2label2[i] for i in range(num_labels)]
    ))
    for i in range(3500):
        records.append({
            'SUMMARY': "*",
            'MANCOLL': y_true[i],
            'collision_type': y_pred[i]
        })
    result_df = pd.DataFrame(records)
    output_path = "mancoll_bert2/bert_test_results-845-4.xlsx"
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    result_df.to_excel(output_path, index=False)
    # 去掉 Unknown
    mask = y_true != 6
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    time_end = time.time()
    print(f"\n-- Excluding 'Unknown' class --")
    print(f"Accuracy (no Unknown): {accuracy_score(y_true_filtered, y_pred_filtered):.4f}")
    print(f"F1-macro (no Unknown): {f1_score(y_true_filtered, y_pred_filtered, average='macro'):.4f}")
    print("***")
    print(time_end - time_start)

# eval_and_print("Validation", val_ds)
eval_and_print("Test", test_ds)


def predict_texts(texts: List[str]) -> List[int]:
    enc = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(trainer.model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = trainer.model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    return [id2label[p] for p in preds]


# if __name__ == "__main__":
#     examples = [
#         "V1 struck the rear of V2 pushing V2 into V3. V3 then left the scene.",
#         "Two vehicles collided head-on on a two-lane road.",
#     ]
#     print("\nPredictions:", predict_texts(examples))
