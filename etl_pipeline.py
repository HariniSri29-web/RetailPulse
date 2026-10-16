
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "online_retail_II.xlsx"
OUTPUT_DIR = BASE_DIR / "etl_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "clean_retail_data.csv"
REPORT_FILE = OUTPUT_DIR / "etl_report.txt"

print("=" * 50)
print("RETAILPULSE - ETL PIPELINE")
print("=" * 50)

# 1. EXTRACT
print("\n1. Extracting raw Excel data...")

sheets = pd.read_excel(RAW_FILE, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

original_rows = len(df)
print("Rows extracted:", original_rows)

# 2. TRANSFORM
print("\n2. Cleaning and transforming data...")

df.columns = df.columns.str.strip()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)
df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)
df["Customer ID"] = df["Customer ID"].astype("string").str.strip()
df["StockCode"] = df["StockCode"].astype("string").str.strip()

# Remove rows that cannot support sales analysis.
df = df.dropna(
    subset=["Invoice", "StockCode", "InvoiceDate", "Quantity", "Price"]
)

# Add project-specific features.
df["TotalAmount"] = df["Quantity"] * df["Price"]
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["IsReturn"] = df["Quantity"] < 0
df["IsCancelledInvoice"] = (
    df["Invoice"].astype(str).str.upper().str.startswith("C")
)

# Keep original rows with returns for return analysis.
# Do not remove duplicates blindly: repeated invoice lines can be valid.
cleaned_rows = len(df)
removed_rows = original_rows - cleaned_rows

print("Rows after cleaning:", cleaned_rows)
print("Rows removed for missing/invalid core fields:", removed_rows)
print("Return records:", int(df["IsReturn"].sum()))
print("Cancelled invoice records:", int(df["IsCancelledInvoice"].sum()))

# 3. LOAD
print("\n3. Saving transformed data...")

df.to_csv(OUTPUT_FILE, index=False)

report = f"""
RETAILPULSE ETL REPORT
----------------------
Source file: {RAW_FILE.name}
Rows extracted: {original_rows}
Rows after cleaning: {cleaned_rows}
Rows removed: {removed_rows}
Return records: {int(df["IsReturn"].sum())}
Cancelled invoice records: {int(df["IsCancelledInvoice"].sum())}
Missing customer IDs: {int(df["Customer ID"].isna().sum())}
Output file: {OUTPUT_FILE}
"""

REPORT_FILE.write_text(report, encoding="utf-8")

print("\nETL completed successfully!")
print("Clean dataset:", OUTPUT_FILE)
print("ETL report:", REPORT_FILE)
