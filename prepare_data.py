
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "online_retail_II.xlsx"
OUTPUT_FILE = BASE_DIR / "retailpulse_prepared.csv"

print("Loading Excel sheets...")

# Load and combine both years
sheets = pd.read_excel(INPUT_FILE, sheet_name=None)

frames = []

for sheet_name, data in sheets.items():
    print(f"Loaded {sheet_name}: {len(data):,} rows")
    frames.append(data)

df = pd.concat(frames, ignore_index=True)

print("\nCombined shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip()

# Convert data types
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

# Custom column 1: transaction line amount
df["TotalAmount"] = df["Quantity"] * df["Price"]

# Custom column 2: year and month
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# Custom column 3: identify negative-quantity records
df["IsReturn"] = df["Quantity"] < 0

# Add an indicator for cancelled invoices
df["IsCancelledInvoice"] = (
    df["Invoice"].astype(str).str.startswith("C", na=False)
)

# Save prepared dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nPreparation completed!")
print("Final shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values in Customer ID:", df["Customer ID"].isna().sum())
print("Return records:", df["IsReturn"].sum())
print("Cancelled invoice records:", df["IsCancelledInvoice"].sum())
print("Saved to:", OUTPUT_FILE)
