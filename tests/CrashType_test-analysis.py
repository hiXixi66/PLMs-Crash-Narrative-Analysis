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



from collections import defaultdict


class CrashTypeClassifier:
    
    def __init__(self, llm,  config_crashtype_df):
        self.llm = llm
        # self.crashcat_class = crashcat_class_dict
        self.config_crashtype_df = config_crashtype_df
        
    def get_crashtype_class_dict(self, crash_conf):
        crashtype_class = self.config_crashtype_df[crash_conf].iloc[3]  # 或你实际使用的列

        try:
            # 修复未闭合字符串的问题
            # crashtype_class = crashtype_class.replace("another's", "another\\'s")
            # 或者更安全地统一转义所有 ' 字符
            # crashtype_class = crashtype_class.replace("'", "\\'")
            crashtype_class = "\n".join([line.strip() for line in crashtype_class.splitlines()])  # 去掉缩进
            return ast.literal_eval(crashtype_class)
        except Exception as e:
            print(f"[Error] Failed to parse crash type dict for {crash_conf}: {e}")
            print(f"[Raw string]:\n{crashtype_class}")
            return {}

    
    def get_config_crashtype_df(self):
        return self.config_crashtype_df
        
    def get_ground_truth_info(self, caseid, vehno, df_gv):
        # print(f"[Info] Searching for ground truth for CASEID {caseid}, VEHNO {vehno}...")

        row = df_gv[(df_gv['CASEID'].astype(str) == str(caseid)) & (df_gv['VEHNO'].astype(str) == str(vehno))]
        
        if not row.empty:
            # print(f"[Info] Found ground truth for CASEID {caseid}, VEHNO {vehno}.")
            crashcat = row.iloc[0].get('CRASHCAT', None)
            crashconf = row.iloc[0].get('CRASHCONF', None)
            crashtype = row.iloc[0].get('CRASHTYPE', None)
            return crashcat, crashconf, crashtype

        print(f"[Warning] Ground truth not found for CASEID {caseid}, VEHNO {vehno}.")
        return None, None, None

    def build_prompt(self, vehicle_index, vehicle_summary, crashtype_class_dict):
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
        # print(prompt)
        return prompt

    def classify(self, vehno, summary, crashtype_class_dict):
        prompt = self.build_prompt(vehno, summary, crashtype_class_dict)
        response = self.llm.query(prompt)
        print(f"[Info] LLM response for vehicle {vehno}: {response}")
        for letter, number in zip(['A', 'B', 'C', 'D'], ['10', '11', '12', '13']):
            response = response.replace(letter, number)

        numbers = re.findall(r'\b\d+\b', str(response))
        for num_str in numbers:
            num = int(num_str) % 100
            if 0 <= num < 100:
                return num
        return None

    def evaluate(self, true_values, pred_values):
        true_dict = {
            (case, veh): (int(cat) if pd.notna(cat) else -1, conf, int(t) if pd.notna(t) else -1)
            for case, veh, cat, conf, t in true_values
        }

        pred_dict = {
            (case, veh): (int(cat) if pd.notna(cat) else -1, conf, int(t) if pd.notna(t) else -1)
            for case, veh, cat, conf, t in pred_values
        }

        total_true = len(true_dict)
        total_pred = len(pred_dict)

        matched = 0
        exact_match = 0
        cat_match = 0
        conf_match = 0
        type_match = 0
        missed_cases = []
        wrong_preds = []

        correctness_dict = {}

        for key, true_item in true_dict.items():
            if key in pred_dict:
                matched += 1
                pred_item = pred_dict[key]

                cat_correct = true_item[0] == pred_item[0]
                conf_correct = true_item[1] == pred_item[1]
                type_correct = true_item[2] == pred_item[2]
                exact_correct = true_item == pred_item

                correctness_dict[key] = {
                    "cat_correct": cat_correct,
                    "conf_correct": conf_correct,
                    "type_correct": type_correct,
                    "exact_correct": exact_correct,
                    "true": true_item,
                    "pred": pred_item
                }

                cat_match += int(cat_correct)
                conf_match += int(conf_correct)
                type_match += int(type_correct)
                exact_match += int(exact_correct)
            else:
                missed_cases.append(key)
                correctness_dict[key] = {
                    "cat_correct": False,
                    "conf_correct": False,
                    "type_correct": False,
                    "exact_correct": False,
                    "true": true_item,
                    "pred": None
                }

        for key in pred_dict:
            if key not in true_dict:
                wrong_preds.append(key)
                correctness_dict[key] = {
                    "cat_correct": False,
                    "conf_correct": False,
                    "type_correct": False,
                    "exact_correct": False,
                    "true": None,
                    "pred": pred_dict[key]
                }

        precision = exact_match / total_pred if total_pred else 0
        recall = exact_match / total_true if total_true else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        result = {
            "Total Ground Truth Vehicles": total_true,
            "Total Predicted Vehicles": total_pred,
            "Matched (case_id + vehno)": matched,
            "Crash Category Accuracy": cat_match / matched if matched else 0,
            "Crash Config Accuracy": conf_match / matched if matched else 0,
            "Crash Type Accuracy": type_match / matched if matched else 0,
            "Exact Match Accuracy": exact_match / matched if matched else 0,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1,
            "Missed Vehicles": missed_cases,
            "Wrong Extra Predictions": wrong_preds,
            "Per Vehicle Evaluation": correctness_dict
        }

        return result
    
def replace_vehicle_reference(label_id: int, text: str) -> str:
    """
    Replace references like 'V5', 'V#5', 'Vehicle 5', or 'Vehicle #5'
    with 'the vehicle to be classified'.
    """
    pattern = fr'\b(V#{label_id}|V{label_id}|Vehicle #{label_id}|Vehicle {label_id})\b'
    return re.sub(pattern, 'the vehicle to be classified', text, flags=re.IGNORECASE)

def evaluate_vehicle_predictions(true_values, pred_values):

    true_dict = {
        (case, veh): (int(cat) if pd.notna(cat) else -1, conf, int(t) if pd.notna(t) else -1)
        for case, veh, cat, conf, t in true_values
    }

    pred_dict = {
        (case, veh): (int(cat) if pd.notna(cat) else -1, conf, int(t) if pd.notna(t) else -1)
        for case, veh, cat, conf, t in pred_values
    }
    
    total_true = len(true_dict)
    total_pred = len(pred_dict)

    matched = 0
    exact_match = 0
    cat_match = 0
    conf_match = 0
    type_match = 0
    missed_cases = []
    wrong_preds = []

    for key, true_item in true_dict.items():
        if key in pred_dict:
            matched += 1
            pred_item = pred_dict[key]
            if true_item[0] == pred_item[0]:
                cat_match += 1
            if true_item[1] == pred_item[1]:
                conf_match += 1
            if true_item[2] == pred_item[2]:
                type_match += 1
            if true_item == pred_item:
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
    "Crash Config Accuracy": conf_match / matched if matched else 0,
    "Crash Type Accuracy": type_match / matched if matched else 0,
    "Exact Match Accuracy": exact_match / matched if matched else 0,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1,
    "Missed Vehicles": missed_cases,
    "Wrong Extra Predictions": wrong_preds
    }

    return result


def process_excel(file_path, output_path):
    df = pd.read_excel(file_path, sheet_name=0)

    if 'SUMMARY' not in df.columns:
        raise ValueError("The first sheet must contain a column named 'summary'.")

    df["collision_type"] = df["SUMMARY"].apply(classify)

    df.to_excel(output_path, index=False)
    print(f"Done. Results saved to {output_path}")
    


def append_row_to_excel(row_data, file_path):
    new_row_df = pd.DataFrame([row_data])

    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df
        
    updated_df.to_excel(file_path, index=False)
    





    
def extract_vehicle_info_by_caseid(df_gv, caseid):

    df_filtered = df_gv[df_gv['CASEID'] == caseid]

    result = []
    for _, row in df_filtered.iterrows():
        case_id = row.get('CASEID')
        vehno = row.get('VEHNO')
        crashcat = row.get('CRASHCAT')
        crashconf = row.get('CRASHCONF')
        crashtype = row.get('CRASHTYPE')
        result.append((case_id, vehno, crashcat, crashconf, crashtype))

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
    





def vehicle_crashtype_classification_with_classifier(classifier,  config_crashtype_df, file_path, output_excel_path):
    df_gv = pd.read_excel(file_path, sheet_name='GV')
    df = pd.read_excel(file_path, sheet_name=0)

    if 'SUMMARY' not in df.columns or 'CASEID' not in df.columns:
        raise ValueError("The first sheet must contain 'SUMMARY' and 'CASEID' columns.")

    y_true = []
    y_pred = []
    save_path = output_excel_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for idx, row in df.iterrows():
        # if idx < 10:
        #     continue
        if idx == 2000:
            break
        raw_summary = row['SUMMARY']
        case_id = row['CASEID']
        number_of_vehicles = int(row['VEHICLES'])
        print(f"Processing row {idx + 1}/{len(df)}...")
        if 'results_df' not in locals():
            results_df = pd.DataFrame(columns=["gt_cat", "gt_conf", "gt_type", "pred_type", "correct"])

        
        if number_of_vehicles  == 1:
            pred = []
            for i in range(1, number_of_vehicles + 1):
                summary = replace_vehicle_reference(i,raw_summary) 
                gt_cat, gt_conf, gt_type = classifier.get_ground_truth_info(caseid=case_id, vehno=i, df_gv=df_gv)
                crashcat = gt_cat
                crashconf = gt_conf

                if pd.isna(crashcat) or int(crashcat) == 9:
                    print(f"[Warning] Invalid or unknown crash category for vehicle {i}.")
                    continue

                if crashconf is None or crashconf not in config_crashtype_df.columns:
                    print(f"[Warning] Invalid or missing crash configuration for vehicle {i}.")
                    continue
                
                if LLM.provider == "bert" or LLM.provider == "BERT":
                    print(f"[Info] Using BERT model for vehicle {i}.")
                    crashtype = LLM.query(summary)
                    crashtype = int(crashtype)
               
                else:
                    crashtype_class = config_crashtype_df[crashconf].iloc[4]
                    crashtype_offset = int(config_crashtype_df[crashconf].iloc[1])
                    # crashtype_prompt_context = config_crashtype_df[crashconf].iloc[4]

                    crashtype = classifier.classify(
                        vehno=i,
                        summary=summary,
                        crashtype_class_dict=crashtype_class
                    )
                    if crashtype is None:
                        print(f"[Warning] Unable to classify crash type for vehicle {i}.")
                        continue

                    crashtype += crashtype_offset
                correct_flag = 1 if crashtype == gt_type else 0

                # 追加一行到表格
                results_df.loc[len(results_df)] = {
                    "gt_cat": gt_cat,
                    "gt_conf": gt_conf,
                    "gt_type": gt_type,
                    "pred_type": crashtype,
                    "correct": correct_flag
                }
    
                results_df.to_csv(save_path, index=False)




            


def extract_vehicle_descriptions(summary):
    pattern = r"(Vehicle\s*\d+.*?)((?=Vehicle\s*\d+)|$)"
    matches = re.findall(pattern, summary, re.DOTALL)
    return [m[0].strip() for m in matches]

    
if __name__ == "__main__":
    # LLM = LLM_HPC(model_name="gpt-4o", provider = "openai", max_new_tokens=2)
    # LLM = LLM_HPC(model_name="qwen2.5-7b-instruct-1m", provider="transformers", max_new_tokens=1)
    # LLM = LLM_HPC(model_name="llama3-70b", provider="transformers", max_new_tokens=1)
    # LLM = LLM_HPC(model_name="deepseek-r1-Distill-Qwen-32B", provider="transformers")
    # LLM = LLM_HPC(model_name="llama3-8b", provider="transformers", max_new_tokens=1)
    LLM = LLM_HPC(model_name="BERT", provider = "BERT")
    
    config_crashtype_df = pd.read_excel("tests/crashtype.xlsx", sheet_name="Sheet3")
    classifier = CrashTypeClassifier(LLM, config_crashtype_df)
    
    
    file_path = "data/processed_data/case_info_2020.xlsx"
    sheet_names = pd.ExcelFile(file_path).sheet_names
    print("Available sheets:", sheet_names)
    output_excel_path = "reports/crash-type-test/results/crash-cons/crashtype_BERT-2900-1veh-3.xlsx"
    
    vehicle_crashtype_classification_with_classifier(
    classifier=classifier,
    config_crashtype_df=config_crashtype_df,
    file_path=file_path,
    output_excel_path= output_excel_path
    )
