import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import json


def build_crashcat_prompt(vehicle_index, vehicle_summary, crashcat_class_dict):
    """
    Build the prompt for crash category classification.
    """
    rules_text = json.dumps(crashcat_class_dict)
    prompt = f"""You are an expert in vehicle accident classification.

        Each vehicle involved in a crash is described in natural language.
        Your task is to classify each vehicle into one of the following 6 crash categories:

        {rules_text}

        Now, please classify the vehicle {vehicle_index} based on its description.

        Vehicle Description:
        \"\"\"{vehicle_summary}\"\"\"

        Respond only with the category number (1 to 6).
        """
    return prompt


def list_crashcat_classes():
    """
    Returns the crash category description for a given crashcat code.

    Returns:
        dict: Mapping of category number (1-6) to its description.
    """
    crashcat_class = {
        1: "Single Driver: Only one vehicle involved, such as run-off-road crashes, rollovers, or collisions with stationary objects (e.g., tree, pole, or barrier). No interaction with other moving vehicles.",
        2: "Same Trafficway, Same Direction: Two or more vehicles traveling in the same lane or direction on the same roadway (e.g., rear-end collisions, sideswipes while overtaking or merging).",
        3: "Same Trafficway, Opposite Direction: Vehicles traveling on the same roadway but in opposite directions (e.g., head-on collisions, sideswipes when passing an oncoming vehicle).",
        4: "Changing Trafficway, Vehicle Turning: A vehicle changes its path (e.g., by turning, merging, or crossing a centerline), resulting in a crash with another vehicle (e.g., left-turn across path, lane change conflicts).",
        5: "Intersecting Paths (Vehicle Damage): Crashes occurring at intersections or junctions where vehicle paths cross at an angle (e.g., T-bone crashes, intersection collisions), typically involving clear impact damage.",
        6: "Miscellaneous: Any crash type not covered by the other categories, including unusual scenarios (e.g., vehicle-to-pedestrian, off-road crashes involving moving vehicles, animal strikes, etc.)."
    }
    return crashcat_class


def get_gt_CrashInfoperVeh(caseid, vehno, df_gv):
    """
    Retrieve CRASHCAT and CRASHCONF for a given CASEID and VEHNO from df_gv.

    Parameters:
        caseid (int or str): The CASEID of the crash case.
        vehno (int or str): The vehicle number.
        df_gv (pd.DataFrame): DataFrame loaded from the 'GV' sheet.

    Returns:
        tuple: (CRASHCAT, CRASHCONF, CRASTYPE) if found, else (None, None, None)
    """
    row = df_gv[(df_gv['CASEID'] == caseid) & (df_gv['VEHNO'] == vehno)]
    if not row.empty:
        crashcat = row.iloc[0]['CRASHCAT']
        crashconf = row.iloc[0]['CRASHCONF']
        crashtype = row.iloc[0]['CRASHTYPE']
        return crashcat, crashconf, crashtype
    else:
        return None, None, None


# === Path settings ===
model_id = "/mimer/NOBACKUP/groups/naiss2025-22-321/qwen2.5-7b-instruct-1m"
file_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/projects/LLM-crash-data/data/processed_data/case_info_2021.xlsx"
category_config_df = pd.read_excel("/mimer/NOBACKUP/groups/naiss2025-22-321/projects/LLM-crash-data/tests/crashtype.xlsx", sheet_name="Sheet1")
config_crashtype_df = pd.read_excel("/mimer/NOBACKUP/groups/naiss2025-22-321/projects/LLM-crash-data/tests/crashtype.xlsx", sheet_name="Sheet3")

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Load data and convert to HuggingFace Dataset ===
df = pd.read_excel(file_path, sheet_name=0)[["SUMMARY", "VEHICLES", "CASEID"]].dropna()
dataset = Dataset.from_pandas(df)
df_gv = pd.read_excel(file_path, sheet_name='GV')


# === Format data for training ===
def format_example(row, df_gv, config_crashtype_df, tokenizer, max_length=512):
    """
    Convert a single crash case row into training examples for each vehicle.
    """
    summary = row['SUMMARY']
    case_id = row['CASEID']
    number_of_vehicles = int(row['VEHICLES'])

    examples = []
    for i in range(1, number_of_vehicles + 1):
        GT_crashcat, GT_crashconf, GT_crashtype = get_gt_CrashInfoperVeh(caseid=case_id, vehno=i, df_gv=df_gv)

        if pd.isna(GT_crashcat):
            print(f"[Warning] Missing ground truth crashcat for vehicle {i} in case {case_id}")
            continue

        # Build prompt for this vehicle
        crashtype_pre_prompt = list_crashcat_classes()
        prompt = build_crashcat_prompt(
            vehicle_index=i,
            vehicle_summary=summary,
            crashcat_class_dict=crashtype_pre_prompt
        )

        label_id = int(GT_crashcat)
        label_str = str(label_id)

        # Tokenize prompt and label
        prompt_tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        answer_ids = tokenizer(label_str, add_special_tokens=False)["input_ids"]

        input_ids = prompt_tokenized["input_ids"]
        attention_mask = prompt_tokenized["attention_mask"]

        # Mask out non-answer tokens
        labels = [-100] * len(input_ids)
        answer_start = len(input_ids) - len(answer_ids)

        if answer_start < 0:
            print(f"[Error] Answer too long for vehicle {i} in case {case_id}")
            continue

        labels[answer_start:] = answer_ids

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        examples.append(example)

    return examples


def format_batch(batch):
    """
    Process a batch of rows into tokenized examples.
    """
    results = []
    for i in range(len(batch["SUMMARY"])):
        row = {k: batch[k][i] for k in batch}
        examples = format_example(row, df_gv, config_crashtype_df, tokenizer)
        results.extend(examples)

    if not results:
        return {}

    output = {k: [example[k] for example in results] for k in results[0]}
    return output


tokenized_dataset = dataset.map(
    format_batch,
    batched=True,
    remove_columns=dataset.column_names
)

# === LoRA configuration ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# === Trainer configuration ===
training_args = TrainingArguments(
    output_dir="models/qwen2.5-finetune-crashcat",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    dataloader_pin_memory=False,
)

# === Create Trainer and train ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=lambda data: {
        "input_ids": torch.tensor([f["input_ids"] for f in data]).to(model.device),
        "attention_mask": torch.tensor([f["attention_mask"] for f in data]).to(model.device),
        "labels": torch.tensor([f["labels"] for f in data]).to(model.device),
    }
)

trainer.train()
