# 📖 Time Series Forecasting

### **Description**  
This section covers **preprocessing time series data**, **checking for stationarity**, **feature engineering**, **training forecasting models (ARIMA, Prophet, LSTMs)**, **evaluating performance**, **hyperparameter tuning**, and **deploying time series forecasts**.

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Understanding Time Series Data**  
  - Ensure dataset has a properly formatted datetime index (`pd.to_datetime(df['date'])`).  
  - Check for missing timestamps and handle gaps appropriately.  
  - Identify trends, seasonality, and stationarity.  

- ✅ **Time Series Data Preprocessing**  
  - Convert timestamps into useful features (hour, day, month, quarter, year).  
  - Handle missing time steps using forward fill (`df.ffill()`) or interpolation.  
  - Normalize or scale time series data (`MinMaxScaler()`, `StandardScaler()`).  

- ✅ **Feature Engineering for Time Series**  
  - Create lag features (`df['lag1'] = df['value'].shift(1)`).  
  - Compute rolling window statistics (`df['rolling_mean'] = df['value'].rolling(3).mean()`).  
  - Generate cyclical features using sine/cosine transformations.  

- ✅ **Time Series Forecasting Models**  
  - Use **ARIMA** for statistical forecasting.  
  - Use **Facebook Prophet** for trend and seasonality-based forecasting.  
  - Train **LSTMs** or **GRUs** using TensorFlow/Keras for deep learning forecasting.  

- ✅ **Deploying Time Series Forecasts**  
  - Store trained models using `joblib.dump()`.  
  - Serve real-time predictions via `FastAPI`.  
  - Automate scheduled re-training using **Airflow** or **cron jobs**.  
