import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.utils.rnn import pad_sequence
def build_prompt(summary):
    return f"""You are a helpful assistant that classifies vehicle collisions into one of the following categories based on the description provided. 
    Please choose the most accurate collision type based on the definitions and clarifications below:

{{
  0: "Not Collision with Vehicle in Transport - The vehicle did not collide with another vehicle in motion.",
  1: "Rear-End - The front of one vehicle strikes the rear of another vehicle traveling in the same direction.",
  2: "Head-On - The front ends of two vehicles traveling in opposite directions collide.",
  4: "Angle - The front of one vehicle strikes the side of another at an angle (usually near intersections or crossing paths).",
  5: "Sideswipe, Same Direction - Both vehicles are moving in the same direction and **their sides make contact**, typically during lane changes.",
  6: "Sideswipe, Opposite Direction - Both vehicles are moving in **opposite directions** and their **sides make contact**, such as on narrow two-way roads.",
  9: "Unknown - The manner of collision cannot be determined from the description."
}}

⚠️ Clarification:
If the collision happens at or near an intersection, classify as 4 (Angle). 
If it does not occur near an intersection:
and both vehicles are traveling in the same direction, classify as 5 (Sideswipe, Same Direction).
and vehicles are traveling in opposite directions, classify as 6 (Sideswipe, Opposite Direction).

📌 Special instructions:
- If the collision involves only one vehicle and a non-vehicle object (e.g., animal, fence, tree), classify it as 0.
- If no collision is described or it is unclear whether any impact occurred, classify as 9.
- If multiple collisions occur (e.g., chain reaction), classify based on the **first** collision described in the summary.

Now read the following description carefully:

\"\"\"{summary}\"\"\"

What is the manner of collision?

Only respond with a single number from the list above. Do not add any explanation."""


collision_classes = {
    0: "Not Collision with Vehicle in Transport",
    1: "Rear-End",
    2: "Head-On",
    4: "Angle",
    5: "Sideswipe, Same Direction",
    6: "Sideswipe, Opposite Direction",
    9: "Unknown"
}

model_id = "Mistral"
file_path = "data/processed_data/case_info_2021.xlsx"

# === load model and Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id
# === Dataset ===
df = pd.read_excel(file_path, sheet_name=0)[["SUMMARY", "MANCOLL"]].dropna()
df = df.rename(columns={"SUMMARY": "summary", "MANCOLL": "label"})
dataset = Dataset.from_pandas(df)


def format_example(example):
    prompt = build_prompt(example["summary"])
    label = str(example['label'])

    prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    answer_ids = tokenizer(label, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids["input_ids"]
    attention_mask = prompt_ids["attention_mask"]

    # Mask non-answer tokens with -100
    labels = [-100] * len(input_ids)
    answer_start = len(input_ids) - len(answer_ids)
    if answer_start < 0:
        raise ValueError("Answer too long to fit in input_ids.")

    labels[answer_start:] = answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_dataset = dataset.map(format_example, remove_columns=["summary", "label"])

# === LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],  # Adjusted for Mistral model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# === Trainer  ===
training_args = TrainingArguments(
    output_dir="models/mistral-ft-MANCOLL",
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
