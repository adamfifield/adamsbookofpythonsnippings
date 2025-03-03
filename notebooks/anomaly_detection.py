"""
Comprehensive Anomaly Detection in Python

This script covers various advanced anomaly detection techniques, including:
- Statistical methods (Z-score, IQR)
- Machine learning-based methods (Isolation Forest, One-Class SVM, Local Outlier Factor, DBSCAN)
- Deep learning methods (Autoencoders)
- Time series anomaly detection
- Evaluation metrics for anomaly detection
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Statistical Methods for Anomaly Detection
# ----------------------------

from scipy.stats import zscore

df = pd.read_csv("data.csv")

# Z-score method
df["zscore"] = zscore(df["value"])
df["is_anomaly_zscore"] = np.abs(df["zscore"]) > 3  # Identify anomalies

# Interquartile Range (IQR) method
Q1 = df["value"].quantile(0.25)
Q3 = df["value"].quantile(0.75)
IQR = Q3 - Q1
df["is_anomaly_iqr"] = (df["value"] < (Q1 - 1.5 * IQR)) | (df["value"] > (Q3 + 1.5 * IQR))

# ----------------------------
# 2. Machine Learning-Based Anomaly Detection
# ----------------------------

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_iforest"] = iso_forest.fit_predict(df[["value"]])

# One-Class SVM
oc_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")
df["anomaly_ocsvm"] = oc_svm.fit_predict(df[["value"]])

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df["anomaly_lof"] = lof.fit_predict(df[["value"]])

# DBSCAN Clustering for Outlier Detection
dbscan = DBSCAN(eps=3, min_samples=2)
df["anomaly_dbscan"] = dbscan.fit_predict(df[["value"]])

# ----------------------------
# 3. Deep Learning-Based Anomaly Detection (Autoencoder)
# ----------------------------

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = df[["value"]].shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train autoencoder
autoencoder.fit(df[["value"]], df[["value"]], epochs=50, batch_size=16, shuffle=True)

# Reconstruction error as anomaly score
reconstructions = autoencoder.predict(df[["value"]])
mse = np.mean(np.abs(df[["value"]] - reconstructions), axis=1)
df["anomaly_autoencoder"] = mse > np.percentile(mse, 95)

# ----------------------------
# 4. Time Series Anomaly Detection
# ----------------------------

df["rolling_mean"] = df["value"].rolling(window=5).mean()
df["rolling_std"] = df["value"].rolling(window=5).std()
df["is_anomaly_ts"] = np.abs(df["value"] - df["rolling_mean"]) > (2 * df["rolling_std"])

# ----------------------------
# 5. Evaluating Anomaly Detection Models
# ----------------------------

from sklearn.metrics import f1_score, precision_score, recall_score

true_labels = np.random.choice([1, -1], size=len(df))  # Dummy ground truth

f1 = f1_score(true_labels, df["anomaly_iforest"])
precision = precision_score(true_labels, df["anomaly_iforest"])
recall = recall_score(true_labels, df["anomaly_iforest"])
