import pandas as pd
import torch
import torch.nn.functional as F
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/llm/')))
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from llm_loader_HPC import LLM_HPC
import matplotlib.pyplot as plt
import time

collision_classes = {
    0: "Not Collision with Vehicle in Transport",
    1: "Rear-End",
    2: "Head-On",
    4: "Angle",
    5: "Sideswipe, Same Direction",
    6: "Sideswipe, Opposite Direction",
    9: "Unknown"
}
# LLM = LLM_HPC(model_name="qwen2.5-7b-instruct-1m", provider="transformers")
# LLM = LLM_HPC(model_name="mistral", provider="transformers")
# LLM = LLM_HPC(model_name="mistral-7b", provider="mistral", max_new_tokens=2)
# LLM = LLM_HPC(model_name="deepseek-r1-Distill-Qwen-32B", provider="transformers")
# LLM = LLM_HPC(model_name="llama3-3b", provider="transformers", max_new_tokens=1)
# LLM = LLM_HPC(model_name="BERT", provider = "BERT")
# LLM = LLM_HPC(model_name="llama3-70b", provider="transformers", max_new_tokens=2)
# LLM = LLM_HPC(model_name="gpt-4o", provider = "openai", max_new_tokens=2)

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




def classify(summary):
    if LLM.provider == "BERT" or LLM.provider == "bert":
        prompt = summary
    else:
        prompt = build_prompt(summary)
    # print(f"Prompt: {prompt}")
    
    response = LLM.query(prompt)  # 你的大语言模型调用逻辑

    # ✅ 从 response 中提取第一个有效数字
    
    match = re.search(r'\b[0-9]\b', str(response))
    # print(f"Response: {response}")
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No valid class number found in response: {response}")

    # matches = re.findall(r'\b[0-9]\b', str(response))
    # if matches:
    #     return int(matches[-1])  # 选择最后一个数字
    # else:
    #     raise ValueError(f"No valid class number found in response: {response}")
    # matches = re.findall(r'\b[0-9]\b', str(response))
    
    # if not matches:
    #     raise ValueError(f"No valid class number found in response: {response}")
    
    # if len(matches) == 1:
    #     return int(matches[0])
    
    # # 多个数字时，优先选第一个不是 9 的
    # for digit in matches:
    #     if digit != '9':
    #         return int(digit)
    
    # # 如果全是9，只能选第一个
    # return int(matches[0])

def process_excel(file_path, output_path):
    df = pd.read_excel(file_path, sheet_name=0)

    if 'SUMMARY' not in df.columns:
        raise ValueError("The first sheet must contain a column named 'summary'.")

    df["collision_type"] = df["SUMMARY"].apply(classify)

    df.to_excel(output_path, index=False)
    # print(f"Done. Results saved to {output_path}")

# def compute_cross_entropy(true_labels, pred_labels, num_classes):
#     true_tensor = torch.tensor(true_labels, dtype=torch.long)
#     pred_tensor = torch.tensor(pred_labels, dtype=torch.long)
#     pred_one_hot = F.one_hot(pred_tensor, num_classes=num_classes).float()
#     loss = F.cross_entropy(pred_one_hot, true_tensor)
#     return loss.item()
def plot_accuracy_heatmap(y_true, y_pred, labels, output_path="confusion_heatmap.png"):

    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')  # 每行归一化
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # 设置样式
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.title("Normalized Confusion Matrix (Accuracy per Class)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path)
    print(f"✅ 图像已保存到: {output_path}")
    plt.show()
    plt.close()
    
    
from sklearn.metrics import accuracy_score, f1_score

def process_excel_row_by_row(file_path, output_path):
    df = pd.read_excel(file_path, sheet_name=0)

    if 'SUMMARY' not in df.columns or 'MANCOLL' not in df.columns:
        raise ValueError("The first sheet must contain 'SUMMARY' and 'MANCOLL' columns.")

    records = []
    y_true = []
    y_pred = []

    for idx, row in df.iterrows():
        if idx == 3500:
            break
        summary = row['SUMMARY']
        mancoll = int(row['MANCOLL'])
        # if mancoll!= 9:
        #     continue
        # print(f"Processing row {idx + 1}/{len(df)}...")
        # time.sleep(1)
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        try:
            pred = int(classify(summary))
        except (ValueError, TypeError):
            pred = 9  
        if LLM.provider == "BERT" or LLM.provider == "bert":
            mapping = {
                3: 4,
                4: 5,
                5: 6,
                6: 9
            }

            if pred in mapping:
                pred = mapping[pred]
            
        records.append({
            'SUMMARY': summary,
            'MANCOLL': mancoll,
            'collision_type': pred
        })
        y_true.append(mancoll)
        y_pred.append(pred)
    time_end = time.time()
    # 计算总指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 添加总指标作为一行写入结果
    records.append({
        'SUMMARY': 'METRICS (ALL)',
        'MANCOLL': f"Accuracy: {acc:.4f}",
        'collision_type': f"Macro F1-score: {f1:.4f}"
    })

    # ⚠️ 去除 MANCOLL == 9 后的指标
    filtered = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != 9]
    if filtered:
        filtered_y_true, filtered_y_pred = zip(*filtered)
        acc_excl9 = accuracy_score(filtered_y_true, filtered_y_pred)
        f1_excl9 = f1_score(filtered_y_true, filtered_y_pred, average='macro')

        # 添加额外指标行
        records.append({
            'SUMMARY': 'METRICS (EXCL 9)',
            'MANCOLL': f"Accuracy: {acc_excl9:.4f}",
            'collision_type': f"Macro F1-score: {f1_excl9:.4f}"
        })

        print(f"📊 Accuracy (excluding 9): {acc_excl9:.4f} | Macro F1-score (excluding 9): {f1_excl9:.4f}")
    else:
        print("⚠️ No data left after excluding MANCOLL == 9.")

    # 保存结果
    result_df = pd.DataFrame(records)
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    result_df.to_excel(output_path, index=False)

    print(f"✅ Done. Results saved to {output_path}")
    print(f"📊 Accuracy: {acc:.4f} | Macro F1-score: {f1:.4f}")
    plot_accuracy_heatmap(y_true, y_pred, list(collision_classes.keys()), output_path=f"reports/figures/confusion_heatmap_{output_path}.png")
    
    print(f"⏱️ Total processing time: {time_end - time_start:.2f} seconds")
    # plot_accuracy_heatmap(y_true, y_pred, list(collision_classes.keys()), output_path="reports/figures/confusion_heatmap_llama3-70b1000v2.png")
    # plot_accuracy_heatmap(y_true, y_pred, list(collision_classes.keys()), output_path="reports/figures/confusion_heatmap_deepseek-r1-Distill-Qwen-32B-1000.png")
    


    
if __name__ == "__main__":
    time_start = time.time()
    print("Starting classification process...")
    # process_excel("data/processed_data/case_info_2020.xlsx", "output_with_classification.xlsx")
    # process_excel_row_by_row("data/processed_data/case_info_2020.xlsx", "reports/mancolll-test/self-con/MANCOLL-classification_llama70b-3.xlsx")
    for i in range(600,601,400):
        
        print(f"Test with {i} samples")
        model_dir = f"models/bert-ft-MANCOLL-limit-sample-test/limit-samples-{i}-new/checkpoint-5epoch"
        # model_dir = f"models/llama3B-ft-MANCOLL-limit-sample/with-limit-samples-{i}/checkpoint-{int(i/2)}"
        print(f"Loading model from {model_dir}")
        LLM = LLM_HPC(model_name="BERT", provider = "BERT",model_dir=model_dir)
        # LLM = LLM_HPC(model_name="llama3-3b", provider="transformers", max_new_tokens=1, model_dir=model_dir)
        # process_excel_row_by_row("data/processed_data/case_info_2020.xlsx", "reports/mancolll-test/noise-test/with_25perc/llama3b-bias-20perc-1668.xlsx")
        process_excel_row_by_row("data/processed_data/case_info_2020.xlsx", f"reports/mancolll-test/noise-test/limit-sample/bert-samples-{i}.xlsx")
        # process_excel_row_by_row("data/processed_data/case_info_2020.xlsx", f"/mimer/NOBACKUP/groups/naiss2025-22-321/llama3-3b")
    
    # process_excel_row_by_row("data/processed_data/case_info_2020.xlsx", "reports/output_with_classification_llama3-70b1000v2.xlsx")
    # process_excel_row_by_row("data/processed_data/case_info_2020.xlsx", "reports/output_with_classification_deepseek-r1-Distill-Qwen-32B-1000.xlsx")
    
    