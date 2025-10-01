import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import json
import re
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# Prompt building
# ============================================================

def build_crashtype_prompt(vehicle_index, vehicle_summary, crashtype_class_dict):
    """
    Build the prompt for crash type classification.
    """
    crash_type_options = json.dumps(crashtype_class_dict)
    prompt = f"""You are a crash analysis assistant.

        Your task is to assign a crash type ID to a specific vehicle involved in a traffic crash,
        based on the structured context and detailed textual description below.

        Use the following crash type definitions for classification:
        {crash_type_options}

        Crash classification must be based on the following inputs:

        - Vehicle index: {vehicle_index}
        - Vehicle Description (may include context about other involved vehicles):
        \"\"\"{vehicle_summary}\"\"\"

        Instructions:
        - Carefully identify and focus on vehicle {vehicle_index} in the text.
        - Consider not only this vehicle's motion and behavior but also how it interacted with other vehicles
          (e.g., which vehicle was backing, struck another, etc.).
        - Use both the structured crash context and relevant textual evidence to determine the most appropriate crash type ID.

        Respond with only one number or letter corresponding to the correct crash type from the options above.
        Do not include any explanation or extra text.
        """
    return prompt


def replace_vehicle_reference(label_id: int, text: str) -> str:
    """
    Replace mentions like 'V5', 'V#5', 'Vehicle 5', or 'Vehicle #5'
    with 'the vehicle to be classified'.
    """
    pattern = fr'\b(V#{label_id}|V{label_id}|Vehicle #{label_id}|Vehicle {label_id})\b'
    return re.sub(pattern, 'the vehicle to be classified', text, flags=re.IGNORECASE)


def get_gt_CrashInfoperVeh(caseid, vehno, df_gv):
    """
    Retrieve CRASHCAT / CRASHCONF / CRASTYPE for a given CASEID and VEHNO.

    Returns:
        (crashcat, crashconf, crashtype) or (None, None, None) if not found
    """
    row = df_gv[(df_gv['CASEID'] == caseid) & (df_gv['VEHNO'] == vehno)]
    if not row.empty:
        crashcat = row.iloc[0]['CRASHCAT']
        crashconf = row.iloc[0]['CRASHCONF']
        crashtype = row.iloc[0]['CRASHTYPE']
        return crashcat, crashconf, crashtype
    return None, None, None


# ============================================================
# Paths & model setup
# ============================================================

model_id = "/mimer/NOBACKUP/groups/naiss2025-22-321/llama3-8b"
file_path = "data/processed_data/case_info_2021.xlsx"
category_config_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet1")
config_crashtype_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet3")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # ensure we have a pad token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# ============================================================
# Data preparation
# ============================================================

df = pd.read_excel(file_path, sheet_name=0)[["SUMMARY", "VEHICLES", "CASEID"]].dropna()
dataset = Dataset.from_pandas(df)
df_gv = pd.read_excel(file_path, sheet_name='GV')


def format_example(row, df_gv, config_crashtype_df, tokenizer, max_length=512):
    """
    Convert one crash case row into training examples for each vehicle.
    """
    raw_summary = row['SUMMARY']
    case_id = row['CASEID']
    number_of_vehicles = int(row['VEHICLES'])
    examples = []

    for i in range(1, number_of_vehicles + 1):
        summary = replace_vehicle_reference(i, raw_summary)
        GT_crashcat, GT_crashconf, GT_crashtype = get_gt_CrashInfoperVeh(caseid=case_id, vehno=i, df_gv=df_gv)

        if pd.isna(GT_crashtype):
            print(f"[Warning] Missing ground truth crashtype for vehicle {i} in case {case_id}")
            continue

        crashconf = GT_crashconf
        if crashconf not in config_crashtype_df.columns:
            print(f"[Warning] crashconf '{crashconf}' not found in config_crashtype_df columns.")
            continue

        # Get class mapping for this crash configuration
        crashtype_class = config_crashtype_df[crashconf].iloc[4]
        crashtype_offset = int(config_crashtype_df[crashconf].iloc[1])

        prompt = build_crashtype_prompt(
            vehicle_index=i,
            vehicle_summary=summary,
            crashtype_class_dict=crashtype_class
        )

        # Convert ground truth crashtype to label string (0-9 or A-Z)
        label_id = int(GT_crashtype) - crashtype_offset
        if label_id < 10:
            label_str = str(label_id)
        else:
            label_str = chr(ord('A') + (label_id - 10))

        # Tokenize prompt & answer
        prompt_tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        answer_ids = tokenizer(label_str, add_special_tokens=False)["input_ids"]

        input_ids = prompt_tokenized["input_ids"]
        attention_mask = prompt_tokenized["attention_mask"]

        labels = [-100] * len(input_ids)
        answer_start = len(input_ids) - len(answer_ids)
        if answer_start < 0:
            print(f"[Error] Answer too long for vehicle {i} in case {case_id}")
            continue
        labels[answer_start:] = answer_ids

        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })

    return examples


def format_batch(batch):
    """
    Map a batch of raw rows into a list of tokenized examples.
    """
    results = []
    for i in range(len(batch["SUMMARY"])):
        row = {k: batch[k][i] for k in batch}
        examples = format_example(row, df_gv, config_crashtype_df, tokenizer)
        results.extend(examples)

    if not results:
        return {}
    return {k: [ex[k] for ex in results] for k in results[0]}


tokenized_dataset = dataset.map(
    format_batch,
    batched=True,
    remove_columns=dataset.column_names
)

# ============================================================
# LoRA configuration
# ============================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# ============================================================
# Training setup
# ============================================================

training_args = TrainingArguments(
    output_dir="models/qwen-finetune-crashtypeq",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    dataloader_pin_memory=False,
)

# Custom data collator to pad dynamically
data_collator = lambda data: {
    "input_ids": pad_sequence([torch.tensor(f["input_ids"]) for f in data],
                               batch_first=True,
                               padding_value=tokenizer.pad_token_id),
    "attention_mask": pad_sequence([torch.tensor(f["attention_mask"]) for f in data],
                                    batch_first=True,
                                    padding_value=0),
    "labels": pad_sequence([torch.tensor(f["labels"]) for f in data],
                            batch_first=True,
                            padding_value=-100)
}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
