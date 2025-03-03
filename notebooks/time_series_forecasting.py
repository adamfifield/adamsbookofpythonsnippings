"""
Time Series Forecasting in Python

This script covers various essential time series forecasting techniques, including:
- Understanding time series data
- Preprocessing and feature engineering
- Checking for stationarity & differencing
- Building forecasting models (ARIMA, Prophet, LSTMs)
- Evaluating model performance
- Hyperparameter tuning
- Deploying forecasts
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Understanding Time Series Data
# ----------------------------

# Load dataset and convert to datetime index
df = pd.read_csv("timeseries_data.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Check missing timestamps
print(df.isnull().sum())

# Visualize time series data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["value"], label="Time Series Data")
plt.legend()
plt.title("Time Series Data Visualization")
plt.show()

# ----------------------------
# 2. Time Series Data Preprocessing
# ----------------------------

# Extract time-based features
df["year"] = df.index.year
df["month"] = df.index.month
df["day"] = df.index.day
df["weekday"] = df.index.weekday

# Fill missing values using forward fill
df.fillna(method="ffill", inplace=True)

# Normalize time series data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df["scaled_value"] = scaler.fit_transform(df[["value"]])

# ----------------------------
# 3. Stationarity & Differencing
# ----------------------------

# Perform Augmented Dickey-Fuller (ADF) test
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df["value"])
print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")

# Apply differencing to remove trends
df["diff_value"] = df["value"].diff().dropna()

# Apply log transformation for variance stabilization
df["log_value"] = np.log1p(df["value"])

# ----------------------------
# 4. Feature Engineering for Time Series
# ----------------------------

# Create lag features
df["lag_1"] = df["value"].shift(1)
df["lag_3"] = df["value"].shift(3)

# Rolling window features
df["rolling_mean"] = df["value"].rolling(window=3).mean()

# Generate cyclical features (sin/cos transformations for periodicity)
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

# ----------------------------
# 5. Time Series Forecasting Models
# ----------------------------

# ARIMA Model
from statsmodels.tsa.arima.model import ARIMA

model_arima = ARIMA(df["value"].dropna(), order=(5, 1, 0))
model_arima_fit = model_arima.fit()
df["arima_forecast"] = model_arima_fit.predict(start=len(df)-30, end=len(df)-1)

# Facebook Prophet
from prophet import Prophet

df_prophet = df.reset_index().rename(columns={"date": "ds", "value": "y"})
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=30)
forecast = model_prophet.predict(future)
df["prophet_forecast"] = forecast["yhat"].tail(30).values

# ----------------------------
# 6. Model Evaluation Metrics
# ----------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = df["value"].dropna().tail(30)
y_pred_arima = df["arima_forecast"].dropna()
y_pred_prophet = df["prophet_forecast"].dropna()

# Compute evaluation metrics
mae_arima = mean_absolute_error(y_true, y_pred_arima)
mse_arima = mean_squared_error(y_true, y_pred_arima)
r2_arima = r2_score(y_true, y_pred_arima)

mae_prophet = mean_absolute_error(y_true, y_pred_prophet)
mse_prophet = mean_squared_error(y_true, y_pred_prophet)
r2_prophet = r2_score(y_true, y_pred_prophet)

# ----------------------------
# 7. Hyperparameter Tuning for Forecasting Models
# ----------------------------

# Auto ARIMA for parameter tuning
from pmdarima import auto_arima

optimal_arima_model = auto_arima(df["value"].dropna(), seasonal=True, m=12, trace=True)

# Prophet hyperparameter tuning
model_prophet = Prophet(seasonality_mode="multiplicative", changepoint_prior_scale=0.5)
model_prophet.fit(df_prophet)

# ----------------------------
# 8. Deploying Time Series Forecasts
# ----------------------------

# Save ARIMA model
import joblib

joblib.dump(model_arima_fit, "arima_model.pkl")

# Save Prophet model
joblib.dump(model_prophet, "prophet_model.pkl")

# API for serving predictions using FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict/")
def predict(data: dict):
    df_input = pd.DataFrame([data])
    model = joblib.load("arima_model.pkl")
    prediction = model.forecast(steps=1)
    return {"forecast": float(prediction[0])}

# Run FastAPI server (for local testing)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# ----------------------------
# END OF SCRIPT
# ----------------------------
