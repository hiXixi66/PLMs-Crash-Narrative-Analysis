import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import json
import re

import torch.nn.functional as F

from transformers import Trainer


import torch
from transformers import Trainer


ALLOWED_TOKENS = ['0','1','2','3','4','5','6','7','8','9', 'A','B','C','D']

import torch
from transformers import Trainer

BIG_NEG = -1e9

class AllowedVocabTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert tokenizer is not None, "需要传入 tokenizer"
        self.allowed_ids = list({
            tokenizer.convert_tokens_to_ids(tok) if tok in tokenizer.get_vocab()
            else tokenizer.encode(tok, add_special_tokens=False)[0]
            for tok in ['0','1','2','3','4','5','6','7','8','9','A','B','C','D']
        })

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        BIG_NEG = -1e9
        labels = inputs.get("labels")  # [bsz, seq_len]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [bsz, seq_len, vocab]

        # 屏蔽非允许 token
        mask = torch.full_like(logits, BIG_NEG)
        mask[:, :, self.allowed_ids] = 0
        masked_logits = logits + mask

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(masked_logits.view(-1, masked_logits.size(-1)), labels.view(-1))

        if return_outputs:
            return loss, outputs
        return loss


class VocabMaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs 里需要包含：input_ids, attention_mask, labels(长度同输入，只在最后位置是label id，其余为 -100),
        # 以及 task_name 或 per-sample 的 allowed_ids
        task_names = inputs.pop("task_names")  # List[str] 长度=bsz
        labels = inputs["labels"]

        outputs = model(**{k:v for k,v in inputs.items() if k!="labels" and k!="task_names"})
        logits = outputs.logits  # [bsz, seq_len, vocab]

        # 只在最后一个可见位置上训练（或你自己的 label 位置）。
        # 这里假设 labels 只有一个位置不是 -100。
        bsz, seq_len, vocab = logits.shape
        # 找到每个样本的 label 位置（非 -100 的地方）
        label_pos = (labels != -100).float().argmax(dim=1)  # [bsz]
        # 取出对应位置的 logits: [bsz, vocab]
        idx = torch.arange(bsz, device=logits.device)
        logits_last = logits[idx, label_pos, :]

        # 构造 batch 级别的 mask: [bsz, vocab]，允许的 token=0，不允许的 = BIG_NEG
        mask = torch.full_like(logits_last, BIG_NEG)
        for i, task in enumerate(task_names):
            allowed_ids = task_allowed_ids[task]  # 你在外部准备好的 dict（放到全局或 trainer.state）
            mask[i, allowed_ids] = 0.0

        masked_logits = logits_last + mask  # 屏蔽

        # 取出对应的 label id（每条样本只有一个标签）
        gold = labels[idx, label_pos]  # [bsz]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(masked_logits, gold)

        return (loss, outputs) if return_outputs else loss


def build_crashtype_prompt(vehicle_index, vehicle_summary, crashtype_class_dict):
    crash_type_options = json.dumps(crashtype_class_dict)
    prompt = f"""You are a crash analysis assistant.

        Your task is to assign a crash type ID to a specific vehicle involved in a traffic crash, based on the structured context and detailed textual description below.

        Use the following crash type definitions for classification:
        {crash_type_options}

        Crash classification must be based on the following inputs:

        - Vehicle index: {vehicle_index}
        - Vehicle Description (may include context about other involved vehicles):
        \"\"\"{vehicle_summary}\"\"\"

        Instructions:
        - Carefully identify and focus on vehicle {vehicle_index} in the text.
        - Consider not only this vehicle's motion and behavior but also how it interacted with other vehicles (e.g., which vehicle was backing, struck another, etc.).
        - Use both the structured crash context and relevant textual evidence to determine the most appropriate crash type ID.

        Respond with only one number or letter corresponding to the correct crash type from the options above. Do not include any explanation or extra text.
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
category_config_df = pd.read_excel("tests/crashtypeMask.xlsx", sheet_name="Sheet1")
config_crashtype_df = pd.read_excel("tests/crashtypeMask.xlsx", sheet_name="Sheet3")
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
def format_example(row, df_gv, config_crashtype_df, tokenizer, max_length=512):
    raw_summary = row['SUMMARY']
    case_id = row['CASEID']
    number_of_vehicles = int(row['VEHICLES'])

    examples = []
    
    for i in range(1, number_of_vehicles + 1):
        summary = replace_vehicle_reference(i, raw_summary)
        GT_crashcat, GT_crashconf, GT_crashtype = get_gt_CrashInfoperVeh(caseid=case_id, vehno=i, df_gv=df_gv)
        # print(f"Processing vehicle {i} in case {case_id}: GT_crashcat={GT_crashcat}, GT_crashconf={GT_crashconf}, GT_crashtype={GT_crashtype}")
        if pd.isna(GT_crashtype):
            print(f"[Warning] Missing ground truth crashtype for vehicle {i} in case {case_id}")
            continue
        
        crashconf = GT_crashconf
        if crashconf not in config_crashtype_df.columns:
            print(f"[Warning] crashconf '{crashconf}' not found in config_crashtype_df columns.")
            continue
        
        # 获取分类信息
        crashtype_class = config_crashtype_df[crashconf].iloc[4]
        crashtype_offset = int(config_crashtype_df[crashconf].iloc[1])

        # 构造 prompt
        prompt = build_crashtype_prompt(
            vehicle_index=i,
            vehicle_summary=summary,
            crashtype_class_dict=crashtype_class
        )

        label_id = int(GT_crashtype) - crashtype_offset
        if label_id < 10:
            label_str = str(label_id)
        else:
            label_str = chr(ord('A') + (label_id - 10))


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
        examples = format_example(row, df_gv, config_crashtype_df, tokenizer)
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
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

# === Trainer 配置 ===
training_args = TrainingArguments(
    output_dir="models/qwen2.5-finetune-crashtype-smth",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    dataloader_pin_memory=False,
)

# === 创建 Trainer 并训练 ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

trainer = AllowedVocabTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=lambda data: {
        "input_ids": torch.tensor([f["input_ids"] for f in data]).to(model.device),
        "attention_mask": torch.tensor([f["attention_mask"] for f in data]).to(model.device),
        "labels": torch.tensor([f["labels"] for f in data]).to(model.device),
    },
    tokenizer=tokenizer
)


trainer.train()
# tokenizer.save_pretrained(training_args.output_dir)