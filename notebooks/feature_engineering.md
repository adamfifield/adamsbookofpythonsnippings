# 📖 Feature Engineering

### **Description**  
This section covers **handling missing values**, **encoding categorical variables**, **scaling and normalization**, **feature transformation**, **feature selection**, **constructing new features**, **handling outliers**, and **dimensionality reduction**. Feature engineering is a critical step in improving the performance of machine learning models.

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Handling Missing Values**  
  - Use `df.fillna()` with mean, median, or mode for imputation.  
  - Use `SimpleImputer` from `sklearn.impute` for systematic imputation.  
  - Consider dropping columns with excessive missing values (`df.drop()`).  

- ✅ **Encoding Categorical Variables**  
  - Use `pd.get_dummies(df['col'])` for one-hot encoding.  
  - Use `LabelEncoder()` from `sklearn.preprocessing` for ordinal encoding.  
  - Use `TargetEncoder()` for supervised encoding of categorical variables.  

- ✅ **Scaling & Normalization**  
  - Use `StandardScaler()` to standardize data (zero mean, unit variance).  
  - Use `MinMaxScaler()` to scale values between 0 and 1.  
  - Use `RobustScaler()` to handle outliers better.  

- ✅ **Feature Transformation**  
  - Use `np.log1p(df['col'])` to reduce skewness in data.  
  - Use `PolynomialFeatures()` from `sklearn.preprocessing` for polynomial features.  
  - Use `PCA()` from `sklearn.decomposition` to reduce dimensionality.  

- ✅ **Feature Selection**  
  - Use `SelectKBest()` from `sklearn.feature_selection` to select best features.  
  - Use `VarianceThreshold()` to remove low-variance features.  
  - Use `Recursive Feature Elimination (RFE)` for automatic feature selection.  

- ✅ **Feature Construction**  
  - Create interaction terms using `df['new_feature'] = df['feature1'] * df['feature2']`.  
  - Use domain knowledge to derive meaningful features (e.g., `df['bmi'] = df['weight'] / df['height']**2`).  
  - Extract useful date-time features (`df['hour'] = df['timestamp'].dt.hour`).  

- ✅ **Handling Outliers**  
  - Use `df.clip(lower, upper)` to cap extreme values.  
  - Use Z-score or IQR methods to remove extreme outliers.  
  - Apply transformations like `np.sqrt(df['col'])` or `np.log1p(df['col'])` to reduce impact.  

- ✅ **Dimensionality Reduction**  
  - Use `PCA()` for feature reduction in high-dimensional datasets.  
  - Use `t-SNE()` or `UMAP()` for visualization in 2D space.  
  - Apply `TruncatedSVD()` for sparse matrix feature selection.  

- ✅ **Time Series Feature Engineering**  
  - Extract lag features: `df['lag1'] = df['value'].shift(1)`.  
  - Create rolling window statistics using `df.rolling(window=3).mean()`.  
  - Generate cyclical features (sin/cos transformations for periodic data).  
