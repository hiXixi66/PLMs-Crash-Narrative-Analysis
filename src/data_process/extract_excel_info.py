import pandas as pd


def extract_excel_info(file_path):

    xls = pd.ExcelFile(file_path)
    
    table_name = file_path.split("/")[-1]  
    
    sheet_names = xls.sheet_names

    sheet_headers = {}
    for sheet in sheet_names:
        df = xls.parse(sheet, nrows=0) 
        sheet_headers[sheet] = df.columns.tolist()
    
    return {
        "Table Name": table_name,
        "Number of Sheets": len(sheet_names),
        "Sheet Names": sheet_names,
        "Sheet Headers": sheet_headers
    }


file_path = "data/case_info.xlsx"  
excel_info = extract_excel_info(file_path)

# 打印提取的信息
print(f"Table Name: {excel_info['Table Name']}")
print(f"Number of Sheets: {excel_info['Number of Sheets']}")
print("Sheet Names and Headers:")
for sheet, headers in excel_info["Sheet Headers"].items():
    print(f"  - {sheet}: {headers}")
