
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "churn_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading RetailPulse dataset...")

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

# Identify cancelled invoices robustly.
cancelled_flag = (
    df["IsCancelledInvoice"].astype(str).str.lower().eq("true")
)

# Use genuine positive purchase lines.
purchases = df[
    df["Customer ID"].notna()
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~cancelled_flag)
].copy()

purchases["Customer ID"] = purchases["Customer ID"].astype("int64")

# Use the final 90 days as the outcome-observation window.
last_date = purchases["InvoiceDate"].max().normalize()
snapshot_date = last_date - pd.Timedelta(days=90)
label_end = snapshot_date + pd.Timedelta(days=90)

print("Feature snapshot:", snapshot_date.date())
print("Churn observation ends:", label_end.date())

# Only use purchases before the snapshot to create features.
history = purchases[purchases["InvoiceDate"] < snapshot_date].copy()

# Observe customer purchases during the following 90 days.
future = purchases[
    (purchases["InvoiceDate"] >= snapshot_date)
    & (purchases["InvoiceDate"] <= label_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
]

# Customer-level features: all calculated using past data only.
features = history.groupby("Customer ID").agg(
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
    features["Monetary"] / features["Frequency"].clip(lower=1)
)

# Churn label: no qualifying purchase during the next 90 days.
future_customers = set(future["Customer ID"].unique())

features["Churn"] = (
    ~features["Customer ID"].isin(future_customers)
).astype(int)

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalUnits",
    "UniqueProducts",
    "AverageOrderValue"
]

features[feature_columns] = features[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

features = features.dropna(subset=["Churn"])

print("\nCustomer records:", len(features))
print("Churn distribution:")
print(features["Churn"].value_counts())

if features["Churn"].nunique() < 2:
    raise ValueError(
        "Only one churn class was found. Check the dataset dates and label window."
    )

X = features[feature_columns]
y = features["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining churn model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========")
print(classification_report(
    y_test,
    predictions,
    target_names=["Active", "Churned"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

if y_test.nunique() == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

# Score all customers in this historical snapshot.
features["ChurnProbability"] = model.predict_proba(X)[:, 1]
features["PredictedChurn"] = (
    features["ChurnProbability"] >= 0.5
).astype(int)

features["RiskLevel"] = pd.cut(
    features["ChurnProbability"],
    bins=[-0.001, 0.4, 0.7, 1.0],
    labels=["Low", "Medium", "High"]
)

features.to_csv(
    OUTPUT_DIR / "customer_churn_predictions.csv",
    index=False
)

print("\nChurn prediction completed!")
print("Predictions saved to:", OUTPUT_DIR / "customer_churn_predictions.csv")
print("\nRisk-level counts:")
print(features["RiskLevel"].value_counts())
