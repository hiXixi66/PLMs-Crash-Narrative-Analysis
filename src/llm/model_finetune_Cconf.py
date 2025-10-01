import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import json
import re
def build_crashconf_prompt(vehicle_index, vehicle_summary, crashconf_class_dict, text):
    # if isinstance(crashconf_class_dict, str):
    #     crashconf_class_dict = ast.literal_eval(crashconf_class_dict)
    # crashconf_class_dict = json.loads(crashconf_class_dict)
    # config_text = "\n".join([f"{k}: {v}" for k, v in crashconf_class_dict.items()])
    config_text = json.dumps(crashconf_class_dict)

    prompt = f"""You are a vehicle crash analysis expert.

    Each vehicle involved in a crash has already been classified into a crash category.
    Now your task is to classify the crash **configuration** of vehicle V{vehicle_index} into one of the following classes based on the crash description:
    {config_text}
    
    Note that: {text}
    
    Now given the following information to classify the crash **configuration** of vehicle V{vehicle_index}:
    
    Crash Description:
    \"\"\"{vehicle_summary}\"\"\"

    Please respond with the configuration label only (a single uppercase letter froom above classes).
    """
    return prompt
def replace_vehicle_reference(label_id: int, text: str) -> str:
    """
    Replace references like 'V5', 'V#5', 'Vehicle 5', or 'Vehicle #5'
    with 'the vehicle to be classified'.
    """
    pattern = fr'\b(V#{label_id}|V{label_id}|Vehicle #{label_id}|Vehicle {label_id})\b'
    return re.sub(pattern, 'the vehicle to be classified', text, flags=re.IGNORECASE)

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

# === 路径设置 ===
model_id = "/mimer/NOBACKUP/groups/naiss2025-22-321/qwen2.5-7b-instruct-1m"
file_path = "data/processed_data/case_info_2021.xlsx"
category_config_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet1")
config_crashtype_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet3")
# === 加载模型和Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === 加载数据并转换为Dataset ===
df = pd.read_excel(file_path, sheet_name=0)[["SUMMARY", "VEHICLES","CASEID"]].dropna()
dataset = Dataset.from_pandas(df)
df_gv = pd.read_excel(file_path, sheet_name='GV')
# === 格式化数据 ===
def format_example(row, df_gv, category_config_df, tokenizer, max_length=512):
    raw_summary = row['SUMMARY']
    case_id = row['CASEID']
    number_of_vehicles = int(row['VEHICLES'])

    examples = []
    
    for i in range(1, number_of_vehicles + 1):
        summary = replace_vehicle_reference(i, raw_summary)
        GT_crashcat, GT_crashconf, GT_crashtype = get_gt_CrashInfoperVeh(caseid=case_id, vehno=i, df_gv=df_gv)
        
        if pd.isna(GT_crashconf):
            print(f"[Warning] Missing ground truth crashconf for vehicle {i} in case {case_id}")
            continue
        
        crashcat = GT_crashcat
        # if crashconf not in config_crashtype_df.columns:
        #     print(f"[Warning] crashconf '{crashconf}' not found in config_crashtype_df columns.")
        #     continue
        
        
        label_str = GT_crashconf
        # 获取分类信息
        crashconf_class = category_config_df[category_config_df['crash category'] == GT_crashcat].iloc[0]
        # print("Available keys:", crashconf_class.index.tolist())

        # print(f"[Debug] Vehicle {i} in case {case_id}: GT_crashcat: {GT_crashcat}, GT_crashconf: {GT_crashconf}, GT_crashtype: {GT_crashtype}, crashconf_class: {crashconf_class}")
        # 构造 prompt
        prompt = build_crashconf_prompt(
            vehicle_index=i,
            vehicle_summary=summary,
            crashconf_class_dict=crashconf_class['crash configuration long'],
            text=crashconf_class['text']
        )
        # print(f"[Debug] Vehicle {i} in case {case_id}: {prompt}")
        # print(f"[Debug] GT_crashconf: {GT_crashconf}, label_str: {label_str}")

        # 编码 prompt 和 label
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

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        examples.append(example)

    return examples

def format_batch(batch):
    results = []
    for i in range(len(batch["SUMMARY"])):
        row = {k: batch[k][i] for k in batch}
        examples = format_example(row, df_gv, category_config_df, tokenizer)
        results.extend(examples)
    
    # 转换为 dict of lists（HuggingFace datasets 要求）
    if not results:
        return {}
    
    output = {k: [example[k] for example in results] for k in results[0]}
    return output


tokenized_dataset = dataset.map(
    format_batch,
    batched=True,
    remove_columns=dataset.column_names
)


# === LoRA 配置 ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# === Trainer 配置 ===
training_args = TrainingArguments(
    output_dir="models/qwen2.5-finetune-crashconflong",
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

# === 创建 Trainer 并训练 ===
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