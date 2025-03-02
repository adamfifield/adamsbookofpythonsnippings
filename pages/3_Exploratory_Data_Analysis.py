import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from code_executor import code_execution_widget

st.title("Exploratory Data Analysis (EDA)")

# Sidebar persistent code execution widget
code_execution_widget()

st.markdown("## 📌 Exploratory Data Analysis (EDA) Techniques")

# Expandable Section: Understanding the Dataset
with st.expander("📂 Loading & Understanding the Dataset", expanded=False):
    st.markdown("### Loading Data")
    st.code('''
# Load a CSV file into a DataFrame
df = pd.read_csv("data.csv")
print(df.head())  # View first 5 rows
''', language="python")

    st.markdown("### Checking Data Information")
    st.code('''
# Get dataset info
print(df.info())

# Check summary statistics
print(df.describe())
''', language="python")

    st.markdown("### Checking for Duplicates")
    st.code('''
# Count duplicate rows
print(df.duplicated().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)
''', language="python")

# Expandable Section: Data Distribution & Summary Statistics
with st.expander("📊 Data Distribution & Summary Statistics", expanded=False):
    st.markdown("### Viewing Column Distributions")
    st.code('''
# Count unique values in a categorical column
print(df["category_column"].value_counts())

# View distribution of a numerical column
print(df["numeric_column"].describe())
''', language="python")

    st.markdown("### Detecting Outliers")
    st.code('''
# Check for outliers using interquartile range (IQR)
Q1 = df["numeric_column"].quantile(0.25)
Q3 = df["numeric_column"].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
outliers = df[(df["numeric_column"] < lower_bound) | (df["numeric_column"] > upper_bound)]
print(outliers)
''', language="python")

# Expandable Section: Data Visualization - Univariate Analysis
with st.expander("📈 Univariate Analysis - Visualizing Single Variables", expanded=False):
    st.markdown("### Histogram - Distribution of a Single Column")
    st.code('''
import matplotlib.pyplot as plt

# Plot histogram
df["numeric_column"].hist(bins=30, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Numeric Column")
plt.show()
''', language="python")

    st.markdown("### Boxplot - Detecting Outliers")
    st.code('''
import seaborn as sns

# Create a boxplot
sns.boxplot(x=df["numeric_column"])
plt.title("Boxplot of Numeric Column")
plt.show()
''', language="python")

# Expandable Section: Data Visualization - Multivariate Analysis
with st.expander("📊 Multivariate Analysis - Visualizing Relationships", expanded=False):
    st.markdown("### Scatter Plot - Relationship Between Two Variables")
    st.code('''
# Scatter plot of two numerical variables
plt.scatter(df["column_x"], df["column_y"], alpha=0.5)
plt.xlabel("Column X")
plt.ylabel("Column Y")
plt.title("Scatter Plot of Column X vs Column Y")
plt.show()
''', language="python")

    st.markdown("### Correlation Heatmap")
    st.code('''
import seaborn as sns

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
''', language="python")

# Expandable Section: Feature Engineering - Creating New Features
with st.expander("🛠️ Feature Engineering - Creating New Features", expanded=False):
    st.markdown("### Creating New Features Based on Existing Data")
    st.code('''
# Create a new column based on conditions
df["new_feature"] = df["existing_column"] * 2
print(df.head())
''', language="python")

    st.markdown("### Binning Numerical Data")
    st.code('''
# Convert a numerical column into categories (binning)
df["binned"] = pd.cut(df["numeric_column"], bins=3, labels=["Low", "Medium", "High"])
print(df["binned"].value_counts())
''', language="python")

# Expandable Section: Encoding Categorical Variables
with st.expander("🔠 Encoding Categorical Variables", expanded=False):
    st.markdown("### One-Hot Encoding")
    st.code('''
# Convert categorical columns into numerical dummy variables
df = pd.get_dummies(df, columns=["category_column"], drop_first=True)
print(df.head())
''', language="python")

    st.markdown("### Label Encoding")
    st.code('''
from sklearn.preprocessing import LabelEncoder

# Convert categorical labels into numerical values
le = LabelEncoder()
df["category_column"] = le.fit_transform(df["category_column"])
print(df.head())
''', language="python")

# Expandable Section: Feature Scaling & Normalization
with st.expander("📏 Feature Scaling & Normalization", expanded=False):
    st.markdown("### Min-Max Scaling")
    st.code('''
from sklearn.preprocessing import MinMaxScaler

# Scale data between 0 and 1
scaler = MinMaxScaler()
df["scaled_column"] = scaler.fit_transform(df[["numeric_column"]])
print(df.head())
''', language="python")

    st.markdown("### Standardization (Z-Score Scaling)")
    st.code('''
from sklearn.preprocessing import StandardScaler

# Standardize data (mean = 0, std = 1)
scaler = StandardScaler()
df["standardized_column"] = scaler.fit_transform(df[["numeric_column"]])
print(df.head())
''', language="python")

# Expandable Section: Saving Processed Data
with st.expander("💾 Saving Processed Data", expanded=False):
    st.markdown("### Saving Data After Cleaning & Transformation")
    st.code('''
# Save cleaned dataset to CSV
df.to_csv("cleaned_data.csv", index=False)
''', language="python")

    st.markdown("### Saving to Pickle Format")
    st.code('''
# Save DataFrame as a pickle file
df.to_pickle("cleaned_data.pkl")
''', language="python")
