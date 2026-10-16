
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
TRACKING_DIR = BASE_DIR / "mlruns"

print("Loading RetailPulse data...")

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "InvoiceDate", "Quantity", "Price", "IsCancelledInvoice"
    ],
    parse_dates=["InvoiceDate"],
    low_memory=False
)

cancelled = (
    df["IsCancelledInvoice"].astype(str).str.lower()
    .isin(["true", "1", "yes"])
)

sales = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~cancelled)
    & df["InvoiceDate"].notna()
].copy()

sales["Revenue"] = sales["Quantity"] * sales["Price"]

daily = (
    sales.groupby(sales["InvoiceDate"].dt.normalize())["Revenue"]
    .sum()
    .sort_index()
)

daily = daily.reindex(
    pd.date_range(daily.index.min(), daily.index.max(), freq="D"),
    fill_value=0
)
daily.name = "Revenue"

# Create time-based features without using future revenue.
data = pd.DataFrame({"Revenue": daily})
data["DayOfWeek"] = data.index.dayofweek
data["Month"] = data.index.month
data["DayOfYear"] = data.index.dayofyear
data["Lag7"] = data["Revenue"].shift(7)
data["RollingMean7"] = data["Revenue"].shift(1).rolling(7).mean()

data = data.dropna()

features = [
    "DayOfWeek", "Month", "DayOfYear", "Lag7", "RollingMean7"
]

# Keep the last 30 days for testing.
train = data.iloc[:-30]
test = data.iloc[-30:]

X_train, y_train = train[features], train["Revenue"]
X_test, y_test = test[features], test["Revenue"]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

TRACKING_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(TRACKING_DIR.resolve().as_uri())
mlflow.set_experiment("RetailPulse-Revenue-Prediction")

print("Training baseline model...")

with mlflow.start_run(run_name="RandomForest-Revenue-Baseline"):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("test_days", 30)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("MAE", float(mae))
    mlflow.log_metric("RMSE", float(rmse))
    mlflow.log_metric("R2", float(r2))

    results = pd.DataFrame({
        "Date": test.index,
        "ActualRevenue": y_test.values,
        "PredictedRevenue": predictions
    })

    results.to_csv(BASE_DIR / "revenue_model_evaluation.csv", index=False)

    mlflow.log_artifact(str(BASE_DIR / "revenue_model_evaluation.csv"))
    mlflow.sklearn.log_model(model, name="revenue_model")

    print("\nModel evaluation")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))
    print("\nExperiment logged successfully!")

print("Tracking data saved in:", TRACKING_DIR)
