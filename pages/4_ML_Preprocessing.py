import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from code_executor import code_execution_widget

st.title("Machine Learning Preprocessing")

# Sidebar persistent code execution widget
code_execution_widget()

st.markdown("## 📌 Machine Learning Preprocessing Techniques")

# Expandable Section: Data Splitting
with st.expander("📂 Splitting Data into Training & Testing Sets", expanded=False):
    st.markdown("### Splitting Data for Training & Testing")
    st.code('''
from sklearn.model_selection import train_test_split

# Split data into training (80%) and testing (20%) sets
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
''', language="python")

# Expandable Section: Handling Missing Values
with st.expander("🚫 Handling Missing Data", expanded=False):
    st.markdown("### Imputing Missing Values with Mean")
    st.code('''
from sklearn.impute import SimpleImputer

# Replace missing values with column mean
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
''', language="python")

    st.markdown("### Imputing Categorical Data with Most Frequent Value")
    st.code('''
# Replace missing categorical values with most frequent category
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train["category_column"] = cat_imputer.fit_transform(X_train[["category_column"]])
X_test["category_column"] = cat_imputer.transform(X_test[["category_column"]])
''', language="python")

# Expandable Section: Encoding Categorical Variables
with st.expander("🔠 Encoding Categorical Data", expanded=False):
    st.markdown("### One-Hot Encoding (Dummy Variables)")
    st.code('''
from sklearn.preprocessing import OneHotEncoder

# Convert categorical column into multiple binary columns
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_features = encoder.fit_transform(X_train[["category_column"]])

# Convert back to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(["category_column"]))
X_train = X_train.drop(columns=["category_column"]).join(encoded_df)
''', language="python")

    st.markdown("### Label Encoding")
    st.code('''
from sklearn.preprocessing import LabelEncoder

# Convert categorical values into numerical labels
le = LabelEncoder()
X_train["category_column"] = le.fit_transform(X_train["category_column"])
X_test["category_column"] = le.transform(X_test["category_column"])
''', language="python")

# Expandable Section: Feature Scaling & Normalization
with st.expander("📏 Feature Scaling & Normalization", expanded=False):
    st.markdown("### Standardization (Z-score scaling)")
    st.code('''
from sklearn.preprocessing import StandardScaler

# Standardize features (mean = 0, std = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
''', language="python")

    st.markdown("### Min-Max Scaling")
    st.code('''
from sklearn.preprocessing import MinMaxScaler

# Scale features to range 0-1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
''', language="python")

# Expandable Section: Feature Selection
with st.expander("🛠️ Feature Selection", expanded=False):
    st.markdown("### Removing Low-Variance Features")
    st.code('''
from sklearn.feature_selection import VarianceThreshold

# Remove features with very little variance
selector = VarianceThreshold(threshold=0.01)
X_train_reduced = selector.fit_transform(X_train)
X_test_reduced = selector.transform(X_test)
''', language="python")

    st.markdown("### Selecting Features with Correlation Thresholding")
    st.code('''
# Remove features highly correlated with others
correlation_matrix = X_train.corr().abs()
upper_tri = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Drop features with high correlation
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
X_train_reduced = X_train.drop(columns=to_drop)
X_test_reduced = X_test.drop(columns=to_drop)
''', language="python")

# Expandable Section: Feature Engineering
with st.expander("🔧 Feature Engineering", expanded=False):
    st.markdown("### Creating Polynomial Features")
    st.code('''
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
''', language="python")

    st.markdown("### Log Transformation for Skewed Data")
    st.code('''
import numpy as np

# Apply log transformation to reduce skewness
X_train["log_feature"] = np.log1p(X_train["numeric_feature"])
X_test["log_feature"] = np.log1p(X_test["numeric_feature"])
''', language="python")

# Expandable Section: Saving Processed Data
with st.expander("💾 Saving Preprocessed Data", expanded=False):
    st.markdown("### Saving to CSV")
    st.code('''
# Save preprocessed dataset to CSV
X_train.to_csv("X_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
''', language="python")

    st.markdown("### Saving to Pickle Format")
    st.code('''
import joblib

# Save preprocessed data as pickle files
joblib.dump(X_train, "X_train_preprocessed.pkl")
joblib.dump(X_test, "X_test_preprocessed.pkl")
''', language="python")
