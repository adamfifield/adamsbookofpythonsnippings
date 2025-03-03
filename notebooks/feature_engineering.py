"""
Feature Engineering in Python

This script covers various essential feature engineering techniques, including:
- Handling missing values
- Encoding categorical variables
- Scaling & normalization
- Feature transformation
- Feature selection
- Feature construction
- Handling outliers
- Dimensionality reduction
- Time series feature engineering
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Handling Missing Values
# ----------------------------

# Fill missing values with mean
df = pd.read_csv("data.csv")
df_filled = df.fillna(df.mean())

# Use sklearn's SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Drop columns with excessive missing values
df_dropped = df.dropna(axis=1, thresh=int(0.7 * len(df)))  # Keep columns with at least 70% non-null values

# ----------------------------
# 2. Encoding Categorical Variables
# ----------------------------

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["category_column"], drop_first=True)

# Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["category_column"] = le.fit_transform(df["category_column"])

# ----------------------------
# 3. Scaling & Normalization
# ----------------------------

# StandardScaler (zero mean, unit variance)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# MinMaxScaler (scales between 0 and 1)
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)

# RobustScaler (better for outliers)
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)

# ----------------------------
# 4. Feature Transformation
# ----------------------------

# Log transformation to reduce skewness
df["log_feature"] = np.log1p(df["numeric_column"])

# Polynomial feature expansion
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
df_poly = pd.DataFrame(poly.fit_transform(df), columns=poly.get_feature_names(df.columns))

# PCA for dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=["PC1", "PC2"])

# ----------------------------
# 5. Feature Selection
# ----------------------------

# SelectKBest (based on ANOVA F-test)
from sklearn.feature_selection import SelectKBest, f_classif

X = df.drop(columns=["target"])
y = df["target"]
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Variance Threshold (removing low variance features)
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(threshold=0.01)
X_filtered = var_thresh.fit_transform(X)

# ----------------------------
# 6. Feature Construction
# ----------------------------

# Create interaction terms
df["interaction_feature"] = df["feature1"] * df["feature2"]

# Create derived features
df["bmi"] = df["weight"] / (df["height"] ** 2)

# Extract useful date-time features
df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

# ----------------------------
# 7. Handling Outliers
# ----------------------------

# Capping outliers using percentile clipping
df["capped_column"] = df["numeric_column"].clip(lower=df["numeric_column"].quantile(0.05), upper=df["numeric_column"].quantile(0.95))

# Z-score method to remove extreme outliers
df["zscore"] = (df["numeric_column"] - df["numeric_column"].mean()) / df["numeric_column"].std()
df_no_outliers = df[df["zscore"].abs() < 3]

# ----------------------------
# 8. Dimensionality Reduction
# ----------------------------

# Reduce dimensionality using PCA
pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(X), columns=["PC1", "PC2", "PC3"])

# ----------------------------
# 9. Time Series Feature Engineering
# ----------------------------

# Create lag features
df["lag_1"] = df["value"].shift(1)

# Rolling window features
df["rolling_mean"] = df["value"].rolling(window=3).mean()

# Create cyclical features (sin/cos transformation)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
