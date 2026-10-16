
import pandas as pd

file_path = "online_retail_II.xlsx"

# Read the Excel workbook
excel_file = pd.ExcelFile(file_path)

print("Available sheets:", excel_file.sheet_names)

for sheet in excel_file.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    print(f"\n--- Sheet: {sheet} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
