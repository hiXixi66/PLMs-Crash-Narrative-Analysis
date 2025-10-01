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
import ast
import json
from sklearn.metrics import accuracy_score, f1_score


# LLM = LLM_HPC(model_name="gpt-4o", provider = "openai", max_new_tokens=1)
LLM = LLM_HPC(model_name="qwen2.5-7b-instruct-1m", provider="transformers", max_new_tokens=2)
# LLM = LLM_HPC(model_name="llama3-70b", provider="transformers", max_new_tokens=2)
# LLM = LLM_HPC(model_name="deepseek-r1-Distill-Qwen-32B", provider="transformers")
from collections import defaultdict

def evaluate_vehicle_predictions(true_values, pred_values):

    true_dict = {
        (case, veh): (int(cat) if pd.notna(cat) else -1)
        for case, veh, cat in true_values
    }

    pred_dict = {
        (case, veh): (int(cat) if pd.notna(cat) else -1)
        for case, veh, cat in pred_values
    }
    
    total_true = len(true_dict)
    total_pred = len(pred_dict)

    matched = 0
    exact_match = 0
    cat_match = 0
    missed_cases = []
    wrong_preds = []

    for key, true_item in true_dict.items():
        if key in pred_dict:
            matched += 1
            pred_item = pred_dict[key]
            if true_item == pred_item:
                cat_match += 1
                exact_match += 1
        else:
            missed_cases.append(key)

    for key in pred_dict:
        if key not in true_dict:
            wrong_preds.append(key)
    precision = exact_match / total_pred if total_pred else 0
    recall = exact_match / total_true if total_true else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    result = {
    "Total Ground Truth Vehicles": total_true,
    "Total Predicted Vehicles": total_pred,
    "Matched (case_id + vehno)": matched,
    "Crash Category Accuracy": cat_match / matched if matched else 0,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1,
    "Missed Vehicles": missed_cases,
    "Wrong Extra Predictions": wrong_preds
    }

    return result


def classify(summary):
    prompt = build_prompt(summary)
    print(f"Prompt: {prompt}")
    
    response = LLM.query(prompt) 

    
    match = re.search(r'\b[0-9]\b', str(response))
    print(f"Response: {response}")
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No valid class number found in response: {response}")


def process_excel(file_path, output_path):
    df = pd.read_excel(file_path, sheet_name=0)

    if 'SUMMARY' not in df.columns:
        raise ValueError("The first sheet must contain a column named 'summary'.")

    df["collision_type"] = df["SUMMARY"].apply(classify)

    df.to_excel(output_path, index=False)
    print(f"Done. Results saved to {output_path}")
    

def plot_accuracy_heatmap(y_true, y_pred, labels, output_path="confusion_heatmap.png"):

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')  # 每行归一化
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

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
    


def append_row_to_excel(row_data, file_path):
    new_row_df = pd.DataFrame([row_data])

    # 如果文件存在，读取后追加；否则创建
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df

    # 保存到文件
    updated_df.to_excel(file_path, index=False)
    

def build_crashcat_prompt(vehicle_index, vehicle_summary, crashcat_class_dict):
    # if isinstance(crashcat_class_dict, str):
    #     crashcat_class_dict = ast.literal_eval(crashcat_class_dict)
    # rules_text = "\n".join([f"{k}: {v}" for k, v in crashcat_class_dict.items()])

    # crashcat_class_dict = json.loads(crashcat_class_dict)
    # rules_text = "\n".join([f"{k}: {v}" for k, v in crashcat_class_dict.items()])
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


def classify_crashcat(vehno, summary, crashcat_class):
    
    prompt = build_crashcat_prompt(vehno, summary,crashcat_class)
    response = LLM.query(prompt) 

    match = re.search(r'\d+', str(response))  
    # print(f"Response crashcat: {match}")
    if match:
        return int(match.group()[0]) 
    else:
        return 9 

    
def extract_vehicle_info_by_caseid(df_gv, caseid):

    df_filtered = df_gv[df_gv['CASEID'] == caseid]

    result = []
    for _, row in df_filtered.iterrows():
        case_id = row.get('CASEID')
        vehno = row.get('VEHNO')
        crashcat = row.get('CRASHCAT')
        # crashconf = row.get('CRASHCONF')
        # crashtype = row.get('CRASHTYPE')
        result.append((case_id, vehno, crashcat))

    return result


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
    
def list_crashcat_classes():
    """
    Returns the crash category description for a given crashcat code.

    Parameters:
        crashcat_code (int): Integer code representing crash category (1 to 6).

    Returns:
        str: Description of the crash category, or a warning message if code is invalid.
    """
    crashcat_class = {
        1: "Single Driver: Only one vehicle involved, such as run-off-road crashes, rollovers, or collisions with stationary objects (e.g., tree, pole, or barrier). No interaction with other moving vehicles.",
        2: "Same Trafficway, Same Direction: Two or more vehicles traveling in the same lane or direction on the same roadway (e.g., rear-end collisions, sideswipes while overtaking or merging).",
        3: "Same Trafficway, Opposite Direction: Vehicles traveling on the same roadway but in opposite directions (e.g., head-on collisions, sideswipes when passing an oncoming vehicle).",
        4: "Changing Trafficway, Vehicle Turning: A vehicle changes its path (e.g., by turning, merging, or crossing a centerline), resulting in a crash with another vehicle (e.g., left-turn across path, lane change conflicts).",
        5: "Intersecting Paths (Vehicle Damage): Crashes occurring at intersections or junctions where vehicles paths cross at an angle (e.g., T-bone crashes, intersection collisions), typically involving clear impact damage.",
        6: "Miscellaneous: Any crash type not covered by the other categories, including unusual scenarios (e.g., vehicle-to-pedestrian, off-road crashes involving moving vehicles, animal strikes, etc.)."
    }
    return crashcat_class


def vehicle_crashcat_classification(category_config_df, config_crashtype_df, file_path):
    df_gv = pd.read_excel(file_path, sheet_name='GV')
    df = pd.read_excel(file_path, sheet_name=0)
    # excel_path = "evaluation_flat_llama70b-v2.xlsx"
    excel_path = "crashcat_evaluation_flat_qwen-v1-721.xlsx"
    # excel_path = "evaluation_flat_qwen7b-noft-v2.xlsx"
    
    if 'SUMMARY' not in df.columns or 'CASEID' not in df.columns:
        raise ValueError("The first sheet must contain 'SUMMARY' and 'CASEID' columns.")

    records = []
    y_true = []
    y_pred = []
    
    crashcat_class = list_crashcat_classes()
    
    for idx, row in df.iterrows():
        if idx == 200:
            break
        summary = row['SUMMARY']
        case_id = row['CASEID']
        number_of_vehicles =int(row['VEHICLES'])
        print(f"Processing row {idx + 1}/{len(df)}...")
        pred = []
        
        for i in range(1, number_of_vehicles + 1):
            # Extract ground truth values
            GT_crashcat, _, _ =get_gt_CrashInfoperVeh(caseid=case_id, vehno=i, df_gv=df_gv)
            
            # Classify crash category
            crashcat = classify_crashcat(i,summary,crashcat_class)
    
            if int(crashcat) == 9:
                print(f"[Warning] Unable to classify crash category for vehicle {i}.")
                continue
  
            pred.append((case_id, i, crashcat))
    

            
        y_pred.append(pred)
        true_values = extract_vehicle_info_by_caseid(df_gv, case_id)
        y_true.append(true_values)
        print(f"True values for vehicle : {true_values}")
        print(f"Predicted values for vehicle : {pred}")
        eval_result = evaluate_vehicle_predictions(true_values, pred)

        gt_pred_pairs = []
        for i in range(max(len(true_values), len(pred))):
            gt = true_values[i] if i < len(true_values) else "-"
            predex = pred[i] if i < len(pred) else "-"
            gt_pred_pairs.append(f"{gt} → {predex}")
        gt_pred_str = "\n".join(gt_pred_pairs)

        flat_eval = {k: str(v) if isinstance(v, (list, tuple)) else v for k, v in eval_result.items()}

        row_data = {
            "case_id": case_id,
            "summary": summary,
            "GT → Pred": gt_pred_str,
            **flat_eval
        }

        append_row_to_excel(row_data, excel_path)


    
if __name__ == "__main__":
    category_config_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet1")
    config_crashtype_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet3")
    file_path = "data/processed_data/case_info_2020.xlsx"
    sheet_names = pd.ExcelFile(file_path).sheet_names
    print("Available sheets:", sheet_names)

    vehicle_crashcat_classification(category_config_df, config_crashtype_df, file_path)