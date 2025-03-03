# 📖 Anomaly Detection

### **Description**  
This section covers **statistical methods (Z-score, IQR)**, **machine learning-based approaches (Isolation Forest, One-Class SVM, Local Outlier Factor, DBSCAN)**, **deep learning anomaly detection using Autoencoders**, **time series anomaly detection**, and **evaluation metrics for anomaly detection**.

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Statistical Methods for Anomaly Detection**  
  - Use **Z-score method** to identify outliers (`scipy.stats.zscore()`).  
  - Use **Interquartile Range (IQR)** to detect anomalies in continuous data.  
  - Be cautious of **false positives in small datasets**.  

- ✅ **Machine Learning-Based Anomaly Detection**  
  - Use **Isolation Forest (`sklearn.ensemble.IsolationForest`)** to detect anomalies based on data isolation.  
  - Use **One-Class SVM (`sklearn.svm.OneClassSVM`)** for anomaly detection in high-dimensional data.  
  - Apply **Local Outlier Factor (`sklearn.neighbors.LocalOutlierFactor`)** for density-based anomaly detection.  
  - Use **DBSCAN (`sklearn.cluster.DBSCAN`)** to find anomalies in spatial/clustered data.  

- ✅ **Deep Learning-Based Anomaly Detection**  
  - Train an **Autoencoder neural network** to learn normal patterns and identify anomalies based on reconstruction error.  
  - Use **mean squared error (MSE) thresholding** to classify anomalies.  
  - Consider **LSTMs or Transformer-based models** for sequential anomalies.  

- ✅ **Time Series Anomaly Detection**  
  - Compute **rolling mean and standard deviation** for anomaly detection (`df.rolling(window=5).mean()`).  
  - Use **seasonal decomposition** (`statsmodels.tsa.seasonal_decompose()`) for identifying trends and seasonal anomalies.  

- ✅ **Evaluating Anomaly Detection Models**  
  - Use `precision_score()`, `recall_score()`, and `f1_score()` for evaluating anomaly classifiers.  
  - Compare against **simple statistical methods (Z-score, IQR) as baselines**.  
  - Validate detection results using **domain expertise** or labeled anomalies.  
