
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "inventory_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading RetailPulse data...")

df = pd.read_csv(
    DATA_FILE,
    usecols=["StockCode", "Description", "InvoiceDate", "Quantity", "Price"],
    parse_dates=["InvoiceDate"],
    low_memory=False
)

# Keep valid product sales; exclude returns and invalid quantities.
sales = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & df["StockCode"].notna()
    & df["InvoiceDate"].notna()
].copy()

sales["StockCode"] = sales["StockCode"].astype(str).str.strip()
sales = sales[sales["StockCode"] != ""]

# Use the most recent 90 days of available sales.
latest_date = sales["InvoiceDate"].max().normalize()
start_date = latest_date - pd.Timedelta(days=89)

recent_sales = sales[
    (sales["InvoiceDate"] >= start_date)
    & (sales["InvoiceDate"] < latest_date + pd.Timedelta(days=1))
].copy()

print("Analysis period:", start_date.date(), "to", latest_date.date())

# Calculate product-level sales and average daily demand.
product_summary = recent_sales.groupby("StockCode").agg(
    ProductName=("Description", "first"),
    UnitsSold=("Quantity", "sum"),
    Revenue=("Price", lambda x: 0)
).reset_index()

# Calculate revenue separately.
recent_sales["LineRevenue"] = recent_sales["Quantity"] * recent_sales["Price"]
revenue = recent_sales.groupby("StockCode")["LineRevenue"].sum()

product_summary["Revenue"] = (
    product_summary["StockCode"].map(revenue).fillna(0)
)

product_summary["AverageDailyDemand"] = (
    product_summary["UnitsSold"] / 90
)

# Example planning assumptions, since actual stock-on-hand data
# is not included in the original retail transaction dataset.
lead_time_days = 14
safety_stock_days = 7

product_summary["SuggestedSafetyStock"] = np.ceil(
    product_summary["AverageDailyDemand"] * safety_stock_days
).astype(int)

product_summary["SuggestedReorderLevel"] = np.ceil(
    product_summary["AverageDailyDemand"]
    * (lead_time_days + safety_stock_days)
).astype(int)

product_summary["SuggestedOrderQuantity"] = np.ceil(
    product_summary["AverageDailyDemand"] * 30
).astype(int)

def recommend_action(row):
    if row["UnitsSold"] <= 0:
        return "Review Product"
    if row["AverageDailyDemand"] < 0.1:
        return "Low Demand - Review"
    return "Stock Planning Required"

product_summary["Recommendation"] = product_summary.apply(
    recommend_action, axis=1
)

product_summary = product_summary.sort_values(
    "Revenue", ascending=False
)

output_file = OUTPUT_DIR / "inventory_recommendations.csv"
product_summary.to_csv(output_file, index=False)

print("\n========== INVENTORY SUMMARY ==========")
print("Products analyzed:", len(product_summary))
print("\nTop 10 products by revenue:")
print(
    product_summary[
        [
            "StockCode",
            "ProductName",
            "UnitsSold",
            "Revenue",
            "AverageDailyDemand",
            "SuggestedReorderLevel",
            "SuggestedOrderQuantity",
            "Recommendation"
        ]
    ].head(10).to_string(index=False)
)

print("\nInventory optimization completed!")
print("Saved to:", output_file.resolve())
