
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "churn_outputs"
SHAP_DIR = BASE_DIR / "shap_outputs"

SHAP_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalUnits",
    "UniqueProducts",
    "AverageOrderValue"
]

print("Loading data for SHAP explainability...")

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

cancelled_flag = (
    df["IsCancelledInvoice"].astype(str).str.lower().eq("true")
)

purchases = df[
    df["Customer ID"].notna()
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~cancelled_flag)
].copy()

purchases["Customer ID"] = purchases["Customer ID"].astype("int64")

last_date = purchases["InvoiceDate"].max().normalize()
snapshot_date = last_date - pd.Timedelta(days=90)
label_end = snapshot_date + pd.Timedelta(days=90)

history = purchases[
    purchases["InvoiceDate"] < snapshot_date
].copy()

future = purchases[
    (purchases["InvoiceDate"] >= snapshot_date)
    & (
        purchases["InvoiceDate"]
        <= label_end + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
]

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

future_customers = set(future["Customer ID"].unique())

features["Churn"] = (
    ~features["Customer ID"].isin(future_customers)
).astype(int)

features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].replace(
    [np.inf, -np.inf], np.nan
)

X = features[FEATURE_COLUMNS]
y = features["Churn"]

if y.nunique() < 2:
    raise ValueError("Only one churn class exists; SHAP training stopped.")

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

print("Training model...")
model.fit(X_train, y_train)

# Explain the trained logistic regression model in transformed feature space.
imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
classifier = model.named_steps["classifier"]

X_train_transformed = scaler.transform(
    imputer.transform(X_train)
)
X_test_transformed = scaler.transform(
    imputer.transform(X_test)
)

# Use a limited background sample to keep computation manageable.
background = shap.sample(
    X_train_transformed,
    min(200, len(X_train_transformed)),
    random_state=42
)

print("Calculating SHAP values...")
explainer = shap.LinearExplainer(classifier, background)

sample_size = min(1000, len(X_test_transformed))
X_explain = X_test_transformed[:sample_size]
shap_values = explainer(X_explain)

# Feature importance: mean absolute SHAP contribution.
importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "MeanAbsoluteSHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values("MeanAbsoluteSHAP", ascending=False)

importance.to_csv(
    SHAP_DIR / "shap_feature_importance.csv",
    index=False
)

# Global feature importance chart.
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_explain,
    feature_names=FEATURE_COLUMNS,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    SHAP_DIR / "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# Detailed SHAP summary plot.
shap.summary_plot(
    shap_values.values,
    X_explain,
    feature_names=FEATURE_COLUMNS,
    show=False
)
plt.tight_layout()
plt.savefig(
    SHAP_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# Save individual feature contributions for the explained test customers.
individual = pd.DataFrame(
    shap_values.values,
    columns=[f"SHAP_{name}" for name in FEATURE_COLUMNS]
)

individual.insert(
    0, "Customer ID",
    features.loc[X_test.index[:sample_size], "Customer ID"].to_numpy()
)
individual["ActualChurn"] = y_test.iloc[:sample_size].to_numpy()
individual["PredictedChurnProbability"] = model.predict_proba(
    X_test.iloc[:sample_size]
)[:, 1]

individual.to_csv(
    SHAP_DIR / "individual_customer_explanations.csv",
    index=False
)

print("\nSHAP explainability completed!")
print("Feature importance:")
print(importance.to_string(index=False))
print("\nFiles saved in:", SHAP_DIR)
print("- shap_feature_importance.csv")
print("- shap_feature_importance.png")
print("- shap_summary.png")
print("- individual_customer_explanations.csv")
