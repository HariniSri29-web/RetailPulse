
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# 1. CONFIGURATION
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv"
OUTPUT_DIR = BASE_DIR / "lstm_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 30
FORECAST_DAYS = 30
EPOCHS = 20
BATCH_SIZE = 32

np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# 2. LOAD AND PREPARE DAILY REVENUE
# =====================================================
print("Loading retail data...")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "InvoiceDate", "Quantity", "Price",
        "IsCancelledInvoice"
    ],
    low_memory=False
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)
df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

cancelled = (
    df["IsCancelledInvoice"].astype(str).str.lower()
    .isin(["true", "1", "yes"])
)

sales = df[
    df["InvoiceDate"].notna()
    & df["Quantity"].gt(0)
    & df["Price"].gt(0)
    & ~cancelled
].copy()

sales["Revenue"] = sales["Quantity"] * sales["Price"]
sales["Date"] = sales["InvoiceDate"].dt.normalize()

daily = (
    sales.groupby("Date")["Revenue"]
    .sum()
    .sort_index()
    .asfreq("D", fill_value=0)
)

daily = daily.astype(float)

if len(daily) <= WINDOW_SIZE + 10:
    raise ValueError("Not enough daily records to train the LSTM model.")

print(f"Historical days: {len(daily)}")
print(f"First date: {daily.index.min().date()}")
print(f"Last date: {daily.index.max().date()}")

# =====================================================
# 3. TIME-BASED TRAIN/TEST SPLIT
# =====================================================
values = daily.to_numpy().reshape(-1, 1)

split_index = int(len(values) * 0.80)

if split_index <= WINDOW_SIZE:
    raise ValueError("Insufficient training data for the selected window.")

scaler = MinMaxScaler()
scaler.fit(values[:split_index])

scaled_values = scaler.transform(values)

X_train, y_train = [], []

for i in range(WINDOW_SIZE, split_index):
    X_train.append(scaled_values[i - WINDOW_SIZE:i, 0])
    y_train.append(scaled_values[i, 0])

X_train = np.array(X_train, dtype=np.float32).reshape(
    -1, WINDOW_SIZE, 1
)
y_train = np.array(y_train, dtype=np.float32)

X_test, y_test = [], []

for i in range(split_index, len(scaled_values)):
    X_test.append(scaled_values[i - WINDOW_SIZE:i, 0])
    y_test.append(scaled_values[i, 0])

X_test = np.array(X_test, dtype=np.float32).reshape(
    -1, WINDOW_SIZE, 1
)
y_test = np.array(y_test, dtype=np.float32)

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")

# =====================================================
# 4. BUILD THE LSTM MODEL
# =====================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# =====================================================
# 5. TRAIN
# =====================================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

print("Training LSTM model...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

model.save(OUTPUT_DIR / "retailpulse_lstm.keras")

# =====================================================
# 6. EVALUATE ON UNSEEN TEST PERIOD
# =====================================================
predicted_scaled = model.predict(X_test, verbose=0)

y_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()

y_predicted = scaler.inverse_transform(
    predicted_scaled
).ravel()

mae = mean_absolute_error(y_actual, y_predicted)
rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

nonzero = y_actual != 0
if nonzero.any():
    mape = np.mean(
        np.abs(
            (y_actual[nonzero] - y_predicted[nonzero])
            / y_actual[nonzero]
        )
    ) * 100
else:
    mape = np.nan

metrics = pd.DataFrame([{
    "Model": "LSTM",
    "MAE": mae,
    "RMSE": rmse,
    "MAPE_percent_nonzero_days": mape,
    "TestDays": len(y_actual)
}])

metrics.to_csv(
    OUTPUT_DIR / "lstm_evaluation.csv", index=False
)

test_results = pd.DataFrame({
    "Date": daily.index[split_index:],
    "ActualRevenue": y_actual,
    "PredictedRevenue": y_predicted
})

test_results.to_csv(
    OUTPUT_DIR / "lstm_test_predictions.csv", index=False
)

print("\nLSTM Evaluation")
print(metrics.to_string(index=False))

# =====================================================
# 7. FORECAST THE NEXT 30 DAYS
# =====================================================
print(f"\nForecasting the next {FORECAST_DAYS} days...")

sequence = scaled_values[-WINDOW_SIZE:, 0].copy()
future_predictions = []

for _ in range(FORECAST_DAYS):
    model_input = sequence.reshape(1, WINDOW_SIZE, 1)

    next_scaled = float(
        model.predict(model_input, verbose=0)[0, 0]
    )

    future_predictions.append(next_scaled)
    sequence = np.append(sequence[1:], next_scaled)

future_revenue = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
).ravel()

# Revenue cannot be negative
future_revenue = np.maximum(future_revenue, 0)

future_dates = pd.date_range(
    start=daily.index[-1] + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "LSTMForecastRevenue": future_revenue
})

forecast_df.to_csv(
    OUTPUT_DIR / "lstm_future_forecast.csv",
    index=False
)

# =====================================================
# 8. SAVE FORECAST CHART
# =====================================================
plt.figure(figsize=(12, 6))

plt.plot(
    daily.index[-90:],
    daily.iloc[-90:].values,
    label="Historical Revenue"
)

plt.plot(
    future_dates,
    future_revenue,
    label="LSTM Forecast"
)

plt.title("RetailPulse - LSTM Daily Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "lstm_forecast.png",
    dpi=150
)
plt.close()

print("\nLSTM forecasting completed successfully!")
print(f"Output folder: {OUTPUT_DIR}")
print("Generated files:")
print("- lstm_evaluation.csv")
print("- lstm_test_predictions.csv")
print("- lstm_future_forecast.csv")
print("- lstm_forecast.png")
print("- retailpulse_lstm.keras")
