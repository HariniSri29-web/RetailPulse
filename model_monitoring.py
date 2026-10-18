
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "monitoring_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalUnits",
    "UniqueProducts",
    "AverageOrderValue"
]

print("Loading RetailPulse data for drift monitoring...")

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "Invoice", "StockCode", "InvoiceDate", "Quantity",
        "Price", "Customer ID", "TotalAmount",
        "IsCancelledInvoice"
    ],
    parse_dates=["InvoiceDate"],
    low_memory=False
)

cancelled = (
    df["IsCancelledInvoice"].astype(str).str.lower().eq("true")
)

purchases = df[
    df["Customer ID"].notna()
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~cancelled)
].copy()

purchases["Customer ID"] = purchases["Customer ID"].astype("int64")

# Build a historical reference customer-feature dataset.
# Build two comparable 90-day customer-feature snapshots.
last_date = purchases["InvoiceDate"].max().normalize()

current_snapshot_date = last_date
reference_snapshot_date = current_snapshot_date - pd.Timedelta(days=90)

# Each snapshot uses purchases from the 90 days before its snapshot date.
def build_customer_features(data, snapshot_date):
    window_start = snapshot_date - pd.Timedelta(days=90)

    window = data[
        (data["InvoiceDate"] >= window_start)
        & (data["InvoiceDate"] < snapshot_date)
    ].copy()

    features = window.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalUnits=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique")
    ).reset_index()

    features["Recency"] = (
        snapshot_date - features["LastPurchase"].dt.normalize()
    ).dt.days

    features["AverageOrderValue"] = (
        features["Monetary"]
        / features["Frequency"].clip(lower=1)
    )

    return features[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

reference = build_customer_features(
    purchases, reference_snapshot_date
)

current = build_customer_features(
    purchases, current_snapshot_date
)

if reference.empty or current.empty:
    raise ValueError("Reference or comparison data is empty.")

print("Reference snapshot:", reference_snapshot_date.date())
print("Current snapshot:", current_snapshot_date.date())
print("Reference customers:", len(reference))
print("Current customers:", len(current))

# Compare feature distributions using the Population Stability Index (PSI).
# PSI is a descriptive drift signal, not a formal hypothesis test.
def calculate_psi(reference_values, current_values, bins=10):
    ref = np.asarray(reference_values, dtype=float)
    cur = np.asarray(current_values, dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:
        return 0.0 if np.allclose(np.mean(ref), np.mean(cur)) else np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]

    ref_pct = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(cur), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

rows = []

for feature in FEATURE_COLUMNS:
    psi = calculate_psi(reference[feature], current[feature])

    if pd.isna(psi):
        status = "Review"
    elif psi < 0.1:
        status = "Low drift"
    elif psi < 0.25:
        status = "Moderate drift"
    else:
        status = "High drift"

    rows.append({
        "Feature": feature,
        "ReferenceMean": reference[feature].mean(),
        "CurrentMean": current[feature].mean(),
        "PSI": psi,
        "Status": status
    })

report = pd.DataFrame(rows)
report.to_csv(OUTPUT_DIR / "drift_report.csv", index=False)

print("\n========== DRIFT MONITORING REPORT ==========")
print(report.round(4).to_string(index=False))
print("\nReference customers:", len(reference))
print("Comparison customers:", len(current))
print("Report saved to:", OUTPUT_DIR / "drift_report.csv")
