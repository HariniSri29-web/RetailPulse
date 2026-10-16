
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "forecast_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading RetailPulse data...")

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "InvoiceDate", "Quantity", "Price",
        "TotalAmount", "IsCancelledInvoice"
    ],
    parse_dates=["InvoiceDate"],
    low_memory=False
)

# Identify cancelled invoices safely.
cancelled = (
    df["IsCancelledInvoice"].astype(str).str.lower().eq("true")
)

# Keep valid positive sales for revenue forecasting.
sales = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~cancelled)
    & df["InvoiceDate"].notna()
].copy()

# Aggregate revenue by calendar day.
daily = (
    sales.groupby(sales["InvoiceDate"].dt.normalize())["TotalAmount"]
    .sum()
    .sort_index()
)

# Include days with no recorded sales.
all_days = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D"
)

daily = daily.reindex(all_days, fill_value=0)

data = pd.DataFrame({
    "ds": daily.index,
    "y": daily.values
})

print("Historical days:", len(data))
print("First date:", data["ds"].min().date())
print("Last date:", data["ds"].max().date())

# Hold out the last 30 days for evaluation.
if len(data) <= 60:
    raise ValueError("Not enough daily data for a 30-day holdout.")

train = data.iloc[:-30].copy()
test = data.iloc[-30:].copy()

print("\nTraining evaluation model...")

evaluation_model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)

evaluation_model.fit(train)

test_future = test[["ds"]].copy()
test_forecast = evaluation_model.predict(test_future)

actual = test["y"].to_numpy()
predicted = test_forecast["yhat"].to_numpy()

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("\n========== HOLDOUT EVALUATION ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

evaluation_results = test[["ds", "y"]].copy()
evaluation_results["PredictedRevenue"] = predicted
evaluation_results["Error"] = (
    evaluation_results["y"] -
    evaluation_results["PredictedRevenue"]
)

evaluation_results.to_csv(
    OUTPUT_DIR / "forecast_evaluation.csv",
    index=False
)

# Refit using all historical data.
print("\nTraining final forecasting model...")

final_model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)

final_model.fit(data)

# Forecast the next 30 calendar days.
future = final_model.make_future_dataframe(
    periods=30,
    freq="D",
    include_history=True
)

forecast = final_model.predict(future)

forecast_output = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
].copy()

forecast_output.rename(
    columns={
        "ds": "Date",
        "yhat": "ForecastRevenue",
        "yhat_lower": "LowerEstimate",
        "yhat_upper": "UpperEstimate",
        "trend": "Trend"
    },
    inplace=True
)

forecast_output.to_csv(
    OUTPUT_DIR / "daily_revenue_forecast.csv",
    index=False
)

# Plot historical revenue and forecast.
fig = final_model.plot(forecast)
fig.set_size_inches(12, 6)
plt.title("RetailPulse - Daily Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Revenue (dataset currency)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "revenue_forecast.png",
    dpi=150
)
plt.close(fig)

# Plot trend and seasonal components.
fig = final_model.plot_components(forecast)
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "forecast_components.png",
    dpi=150
)
plt.close(fig)

print("\nNext 30 days forecast:")
print(forecast_output.tail(30).head(10).to_string(index=False))

print("\nDemand forecasting completed!")
print("Output folder:", OUTPUT_DIR.resolve())
print("Forecast: daily_revenue_forecast.csv")
print("Evaluation: forecast_evaluation.csv")
print("Chart: revenue_forecast.png")
