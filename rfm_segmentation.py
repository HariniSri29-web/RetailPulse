
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "rfm_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading RetailPulse data...")

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "Invoice", "InvoiceDate", "Quantity",
        "Price", "Customer ID", "TotalAmount",
        "IsCancelledInvoice"
    ],
    parse_dates=["InvoiceDate"],
    low_memory=False
)

# Exclude cancelled invoices and non-purchase lines.
# Customers without an ID cannot be segmented individually.
df = df[
    df["Customer ID"].notna()
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~df["IsCancelledInvoice"].astype(bool))
].copy()

df["Customer ID"] = df["Customer ID"].astype("int64")

# Reference date: one day after the last transaction.
reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

print("Calculating RFM metrics...")

rfm = df.groupby("Customer ID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

rfm["Recency"] = (
    reference_date - rfm["LastPurchase"].dt.normalize()
).dt.days

rfm = rfm[
    (rfm["Frequency"] > 0)
    & (rfm["Monetary"] > 0)
].copy()

# Prepare features for clustering.
features = ["Recency", "Frequency", "Monetary"]

# Log transformation reduces the effect of extreme values.
X = rfm[features].clip(lower=0)
X = X.apply(lambda column: __import__("numpy").log1p(column))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training K-Means model...")

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = model.fit_predict(X_scaled)

# Save customer-level RFM metrics and cluster assignments.
rfm.to_csv(OUTPUT_DIR / "customer_rfm_segments.csv", index=False)

# Summarize each cluster for interpretation.
summary = rfm.groupby("Cluster").agg(
    Customers=("Customer ID", "nunique"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    TotalMonetary=("Monetary", "sum")
).round(2)

summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

print("\nCustomer count:", len(rfm))
print("\nCluster summary:")
print(summary)
print("\nRFM segmentation completed!")
print("Files saved in:", OUTPUT_DIR.resolve())
