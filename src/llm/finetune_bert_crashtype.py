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

def replace_vehicle_reference(label_id: int, text: str) -> str:
    """
    Replace references like 'V5', 'V#5', 'Vehicle 5', or 'Vehicle #5'
    with 'the vehicle to be classified'.
    """
    pattern = fr'\b(V#{label_id}|V{label_id}|Vehicle #{label_id}|Vehicle {label_id})\b'
    return re.sub(pattern, 'the vehicle to be classified', text, flags=re.IGNORECASE)

import re

def build_examples_from_crash_and_gv(df_crash, df_gv, text_col="SUMMARY"):
    """
    Combine CRASH and GV sheets to construct per-vehicle training samples.
    """
    records = []
    for _, row in df_gv.iterrows():
        caseid = row["CASEID"]
        vehno  = row["VEHNO"]
        crashtype = row["CRASHTYPE"]

        # Find the corresponding CRASH summary
        crash_row = df_crash[df_crash["CASEID"] == caseid]
        if crash_row.empty:
            continue
        summary = crash_row.iloc[0][text_col]

        # Replace the vehicle reference with 'the vehicle to be classified'
        text = replace_vehicle_reference(vehno, str(summary))

        records.append({
            "CASEID": caseid,
            "VEHNO": vehno,
            "SUMMARY": text,
            "CRASHTYPE": crashtype
        })
    return pd.DataFrame(records)

def get_gt_CrashInfoperVeh(caseid, vehno, df_gv):
    """
    Retrieve CRASHCAT and CRASHCONF for a given CASEID and VEHNO from df_gv.

    Parameters:
        caseid (int or str): The CASEID of the crash case.
        vehno (int or str): The vehicle number.
        df_gv (pd.DataFrame): DataFrame loaded from the 'GV' sheet.

    Returns:
        tuple: (CRASHCAT, CRASHCONF) if found, else (None, None)
    """
    row = df_gv[(df_gv['CASEID'] == caseid) & (df_gv['VEHNO'] == vehno)]
    
    if not row.empty:
        crashcat = row.iloc[0]['CRASHCAT']
        crashconf = row.iloc[0]['CRASHCONF']
        crashtype = row.iloc[0]['CRASHTYPE']
        return crashcat, crashconf, crashtype
    else:
        return None, None, None

# ======== Configuration ========
SEED = 42
# model_id = "/mimer/NOBACKUP/groups/naiss2025-22-321/llama3-8b"
# file_path = "data/processed_data/case_info_2021.xlsx"
# config_crashtype_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet3")
# === Load model and tokenizer ===
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token
MODEL_NAME = "bert-base-uncased" 
MAX_LENGTH = 256
OUTPUT_DIR = "./crashtype_bert3"
EXCEL_PATH = "data/processed_data/case_info_2021.xlsx"  
test_PATH = "data/processed_data/case_info_2020.xlsx"  
test_path = "data/processed_data/case_info_2020.xlsx"    
TEXT_COL = "SUMMARY"
LABEL_COL = "CRASHTYPE"
VAL_SIZE = 0.1              
TEST_SIZE = 0.1                 

# ======== Random seed ========
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ======== Load data ========
# Original sheets
df_crash = pd.read_excel(EXCEL_PATH, sheet_name="CRASH")
df_gv    = pd.read_excel(EXCEL_PATH, sheet_name="GV")
df = build_examples_from_crash_and_gv(df_crash, df_gv, text_col="SUMMARY")

df_crash_test = pd.read_excel(test_PATH, sheet_name="CRASH")
df_gv_test    = pd.read_excel(test_PATH, sheet_name="GV")
df_test = build_examples_from_crash_and_gv(df_crash_test, df_gv_test, text_col="SUMMARY")

# Clean missing values
df = df[["SUMMARY", "CRASHTYPE","CASEID","VEHNO"]].dropna().reset_index(drop=True)
df_test_extra = df_test[["SUMMARY", "CRASHTYPE","CASEID","VEHNO"]].dropna().reset_index(drop=True)

# ======== Label encoding ========
label2id = {i: i for i in range(100)}   # IDs 0–99
id2label = {i: i for i in range(100)}
num_labels = 100
print("Label to ID mapping:", label2id)

df[LABEL_COL] = df[LABEL_COL].map(label2id)
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].map(label2id)
if df[LABEL_COL].isnull().any():
    raise ValueError("Some labels could not be mapped! Check your LABEL_COL values.")
if df_test_extra[LABEL_COL].isnull().any():
    print(df_test_extra[df_test_extra[LABEL_COL].isnull()])
    raise ValueError("Some labels in test set could not be mapped! Check your LABEL_COL values.")
df[LABEL_COL] = df[LABEL_COL].astype(int)
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].astype(int)

if TEST_SIZE > 0:
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED
        # Removed stratify parameter
    )
else:
    df_trainval, df_test = df, None

df_train, df_val = train_test_split(
    df_trainval, test_size=VAL_SIZE, random_state=SEED
)

# ======== Tokenizer & Dataset ========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)  # ✅ Keep original df for CASEID/VEHNO alignment later
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
        
        if "CASEID" in self.df.columns:
            item["CASEID"] = self.df.loc[idx, "CASEID"]
        if "VEHNO" in self.df.columns:
            item["VEHNO"] = self.df.loc[idx, "VEHNO"]
            
        return item

train_ds = TextClsDataset(df_train, TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)
val_ds   = TextClsDataset(df_val,   TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)
test_ds  = TextClsDataset(df_test_extra, TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH) if df_test is not None else None

# ======== Handle class imbalance (class weights) ========
class_counts = df_train[LABEL_COL].value_counts().sort_index().values
class_weights = (1.0 / (class_counts + 1e-9))  # Inverse frequency
class_weights = class_weights / class_weights.sum() * num_labels
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Custom Trainer to inject class weights
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
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# ======== Metrics ========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

# ======== Training arguments ========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",   # for older Transformers versions
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=10,
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

# ======== Training ========
# trainer.train()
# trainer.save_model(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)
# quit()

from transformers import AutoModelForSequenceClassification
def predict_case(df_crash, caseid, model, tokenizer, max_length=256):
    """
    Predict crash type for each vehicle in a case.
    """
    crash_row = df_crash[df_crash["CASEID"] == caseid]
    if crash_row.empty:
        return []
    summary = crash_row.iloc[0]["SUMMARY"]

    preds = []
    vehnos = crash_row.iloc[0]["VEHNO"]  # If this is a number = total number of vehicles
    for vehno in range(1, vehnos+1):
        text = replace_vehicle_reference(vehno, str(summary))
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
            pred = torch.argmax(logits, dim=-1).item()
        preds.append({"CASEID": caseid, "VEHNO": vehno, "PRED_CRASHTYPE": id2label[pred]})
    return preds

def eval_and_print(name, df_crash, df_gv, dataset, checkpoint_dir="crashtype_bert/checkpoint-2044"):
    """
    Evaluate on a dataset and print both vehicle-level and case-level metrics.
    """
    if dataset is None:
        return

    # Reload model from saved checkpoint
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

    print(f"\n== {name} ==")
    print(f"Accuracy (all vehicles): {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro (all vehicles): {f1_score(y_true, y_pred, average='macro'):.4f}")

    # === Case-level accuracy ===
    records = []
    for caseid in df_gv["CASEID"].unique():
        # Check the total number of vehicles in this case from the CRASH sheet
        crash_row = df_crash[df_crash["CASEID"] == caseid]
        if crash_row.empty:
            continue
        total = int(crash_row.iloc[0]["VEHICLES"])   # ✅ Use VEHICLES column in CRASH sheet
        summary = crash_row.iloc[0]["SUMMARY"]

        correct = 0
        vehs = df_gv[df_gv["CASEID"] == caseid]

        for _, vrow in vehs.iterrows():
            vehno = vrow["VEHNO"]
            true_label = label2id.get(vrow["CRASHTYPE"], None)
            if true_label is None:
                continue

            # Find the matching prediction: align dataset index with CASEID & VEHNO
            pred_idx = dataset.df.index[(dataset.df["CASEID"] == caseid) & (dataset.df["VEHNO"] == vehno)]
            if len(pred_idx) == 0:
                continue
            pred_label = y_pred[pred_idx[0]]

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
    output_path = "crashtype_bert/case_level_results-2044.xlsx"
    result_df.to_excel(output_path, index=False)

    time_end = time.time()
    print(f"\nCase-level results saved to {output_path}")
    print(f"Time elapsed: {time_end - time_start:.2f} sec")


# eval_and_print("Validation", val_ds)
eval_and_print("Test", df_crash_test, df_gv_test, test_ds, checkpoint_dir="crashtype_bert3/checkpoint-2044")

# ======== Example inference function ========

# Example usage:
# if __name__ == "__main__":
#     examples = [
#         "V1 struck the rear of V2 pushing V2 into V3. V3 then left the scene.",
#         "Two vehicles collided head-on on a two-lane road.",
#     ]
#     print("\nPredictions:", predict_case(examples))
