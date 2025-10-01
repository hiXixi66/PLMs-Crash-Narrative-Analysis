import os
import pandas as pd
import chardet 
from xlsxwriter import Workbook
def detect_encoding(file_path, num_bytes=10000):
    """
    Detect the encoding of the specified file.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes) 
    result = chardet.detect(raw_data)
    return result['encoding']

def get_csv_headings(folder_path):
    headings = {}

    for file in os.listdir(folder_path):
        if file.endswith(".csv"): 
            file_path = os.path.join(folder_path, file)
            
            encodings = ['utf-8', 'ISO-8859-1', 'windows-1252', 'latin1']

            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc, nrows=5)
                    print(f"Successfully read file using encoding {enc}")
                    headings[file] = list(df.columns)
                    break 
                except Exception as e:
                    print(f"Cannot read file with encoding {enc}: {e}")
    
    return headings
    
def csv_to_excel(folder_path, output_excel, column_to_csv_mapping):
    """
    Extract column data from the specified CSV file and save it to Excel based on the specified column name-CSV file mapping.

    :param folder_path: CSV file folder path
    :param output_excel: Generated Excel file path
    :param column_to_csv_mapping: Dictionary, specify the columns to be extracted for each Sheet {sheet_name: (csv file name, [columns to be extracted])}
    """
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    
    loaded_csvs = {}
    
    for sheet_name, (csv_file, selected_columns) in column_to_csv_mapping.items():
        csv_path = os.path.join(folder_path, f"{csv_file}.csv")
        
        if not os.path.exists(csv_path):
            print(f"File {csv_file}.csv does not exist, skipping...")
            continue
        
        if csv_file not in loaded_csvs:
                
            encodings = ['utf-8', 'ISO-8859-1', 'windows-1252', 'latin1']

            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    loaded_csvs[csv_file] = df 
                    print(f"Successfully read file using encoding {enc}")
                    break  
                except Exception as e:
                    print(f"Cannot read file with encoding {enc}: {e}")
                    
        
        df = loaded_csvs[csv_file]
        if df is None:
            print(f"File {csv_file}.csv does not exist, skipping...")
            continue
        df_selected = df[selected_columns] if set(selected_columns).issubset(df.columns) else df[df.columns.intersection(selected_columns)]
        
        df_selected.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Processed {csv_file}.csv, saved in sheet: {sheet_name}")
    
    writer.close()
    print(f"Excel saved: {output_excel}")
    


column_to_csv_mapping = {
    "CRASH": ("CRASH", ["CASEID","CRASHYEAR","PSU","CASENO","CASENUMBER","CATEGORY","CRASHMONTH","DAYOFWEEK","CRASHTIME","EVENTS","VEHICLES","MANCOLL","SUMMARY"]),
    "EVENT":("EVENT",["CASEID","PSU","CASENO","CASENUMBER","CATEGORY","EVENTNO","VEHNUM","CLASS1","GAD1","CLASS2","GAD2"]),
    "GV":("GV",['CASEID', 'PSU', 'CASENO', 'CASENUMBER', 'CATEGORY', 'VEHNO','DVLONG','DVLAT','DVTOTAL','DVANGTHIS','CRASHTYPE','PREFHE',	'MODELYR','BODYTYPE','BODYCAT',	'SPEEDLIMIT','CRITEVENT','CRASHCONF','CRASHCAT']),
    "VPICDECODE":("VPICDECODE",['CASEID', 'PSU', 'CASENO', 'CASENUMBER', 'CATEGORY', 'VEHNO','VehicleType','ModelYear', 'Model','AdaptiveCruiseControl',
                                'AdaptiveCruiseControl','AntilockBrakeSystem','AutoPedestrianAlertingSound','DynamicBrakeSupport','ForwardCollisionWarning','LaneCenteringAssistance',
                                'LaneDepartureWarning', 'LaneKeepingAssistance','RearAutomaticEmergencyBraking','ActiveSafetySysNote']),
    "OCC":("OCC",['CASEID', 'PSU', 'CASENO', 'CASENUMBER', 'CATEGORY', 'VEHNO','OCCNO','SEATLOC','AGE', 'HEIGHT','WEIGHT','SEX','FETALMORT','ROLE','EYEWEAR','POSTURE','BELTUSE'])  ,
    "AVOID":("AVOID",['CASEID', 'PSU', 'CASENO', 'CASENUMBER', 'CATEGORY', 'VEHNO','EQUIP','AVAIL','ACTIVATE'])
}
for i in range(17, 24):
    folder_path = "data/raw-data/CISS_20"+str(i)+"_CSV_files"  
    csv_headings = get_csv_headings(folder_path)

    output_excel = "data/processed_data/case_info_20"+str(i)+".xlsx"

    csv_to_excel(folder_path, output_excel, column_to_csv_mapping)