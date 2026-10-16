
import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "eda_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading RetailPulse dataset...")

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["InvoiceDate"],
    low_memory=False
)

print("\n========== DATA OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\n========== DATA QUALITY ==========")
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Use positive-quantity, positive-price records
# for the initial sales analysis.
sales = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0) &
    (~df["IsCancelledInvoice"].astype(bool))
].copy()

print("\n========== SALES ANALYSIS ==========")
print("Eligible sales lines:", len(sales))
print("Sales line value:", round(sales["TotalAmount"].sum(), 2))

# 1. Monthly sales
monthly_sales = (
    sales.groupby("YearMonth", as_index=False)
    .agg(
        Revenue=("TotalAmount", "sum"),
        UnitsSold=("Quantity", "sum"),
        Transactions=("Invoice", "nunique")
    )
    .sort_values("YearMonth")
)

monthly_sales.to_csv(
    OUTPUT_DIR / "monthly_sales.csv", index=False
)

print("\nMonthly sales:")
print(monthly_sales.to_string(index=False))

# 2. Top 10 products by units sold
top_products = (
    sales.groupby("Description", as_index=False)
    .agg(
        UnitsSold=("Quantity", "sum"),
        Revenue=("TotalAmount", "sum")
    )
    .sort_values("UnitsSold", ascending=False)
    .head(10)
)

top_products.to_csv(
    OUTPUT_DIR / "top_products.csv", index=False
)

print("\n========== TOP 10 PRODUCTS ==========")
print(top_products.to_string(index=False))

# 3. Revenue by country
country_sales = (
    sales.groupby("Country", as_index=False)
    .agg(
        Revenue=("TotalAmount", "sum"),
        UnitsSold=("Quantity", "sum"),
        Transactions=("Invoice", "nunique")
    )
    .sort_values("Revenue", ascending=False)
)

country_sales.to_csv(
    OUTPUT_DIR / "country_sales.csv", index=False
)

print("\n========== TOP 10 COUNTRIES ==========")
print(country_sales.head(10).to_string(index=False))

# 4. Customer analysis
customer_sales = sales.dropna(subset=["Customer ID"])

customer_summary = (
    customer_sales.groupby("Customer ID", as_index=False)
    .agg(
        Revenue=("TotalAmount", "sum"),
        Orders=("Invoice", "nunique"),
        UnitsPurchased=("Quantity", "sum")
    )
    .sort_values("Revenue", ascending=False)
)

customer_summary.to_csv(
    OUTPUT_DIR / "customer_summary.csv", index=False
)

print("\n========== TOP 10 CUSTOMERS ==========")
print(customer_summary.head(10).to_string(index=False))

# 5. Returns and cancellations
returns = df[df["IsReturn"].astype(bool)]
cancelled = df[df["IsCancelledInvoice"].astype(bool)]

print("\n========== RETURNS & CANCELLATIONS ==========")
print("Negative-quantity lines:", len(returns))
print("Cancelled-invoice lines:", len(cancelled))
print("Negative-quantity value:", round(returns["TotalAmount"].sum(), 2))

# Save a concise report
with open(OUTPUT_DIR / "eda_report.txt", "w", encoding="utf-8") as report:
    report.write("RETAILPULSE - EDA SUMMARY\n")
    report.write(f"Total rows: {len(df)}\n")
    report.write(f"Total columns: {len(df.columns)}\n")
    report.write(f"Eligible sales lines: {len(sales)}\n")
    report.write(f"Eligible sales line value: {sales['TotalAmount'].sum():.2f}\n")
    report.write(f"Negative-quantity lines: {len(returns)}\n")
    report.write(f"Cancelled-invoice lines: {len(cancelled)}\n")

print("\nEDA completed successfully!")
print("Output folder:", OUTPUT_DIR.resolve())
