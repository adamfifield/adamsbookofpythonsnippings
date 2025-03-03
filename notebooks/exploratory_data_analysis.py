"""
Exploratory Data Analysis (EDA) in Python

This script covers various essential EDA techniques, including:
- Understanding the dataset structure
- Checking for missing values
- Analyzing categorical and numerical variables
- Identifying outliers and anomalies
- Examining correlations
- Visualizing data distributions
- Detecting duplicate records
- Checking dataset balance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# ----------------------------
# 1. Understanding the Dataset Structure
# ----------------------------

# Load dataset
df = pd.read_csv("data.csv")

# Get dataset information
print(df.info())  # Column types and missing values
print(df.describe())  # Summary statistics
print(df.shape)  # Number of rows and columns

# ----------------------------
# 2. Checking for Missing Values
# ----------------------------

# Count missing values
print(df.isnull().sum())

# Drop missing values
df_cleaned = df.dropna()

# Fill missing values with column mean
df_filled = df.fillna(df.mean())

# ----------------------------
# 3. Analyzing Categorical Variables
# ----------------------------

# Get frequency count of categorical values
print(df['category_column'].value_counts())

# Convert categorical variables into dummies
df_encoded = pd.get_dummies(df, columns=['category_column'], drop_first=True)

# ----------------------------
# 4. Analyzing Numerical Variables
# ----------------------------

# Compute summary statistics
print(df['numeric_column'].mean(), df['numeric_column'].median(), df['numeric_column'].std())

# Visualize distribution with histogram
plt.hist(df['numeric_column'], bins=30, edgecolor='black')
plt.title("Histogram of Numeric Column")
plt.show()

# Boxplot for outlier detection
sns.boxplot(x=df['numeric_column'])
plt.title("Boxplot of Numeric Column")
plt.show()

# ----------------------------
# 5. Identifying Outliers and Anomalies
# ----------------------------

# Interquartile Range (IQR) method
Q1 = df['numeric_column'].quantile(0.25)
Q3 = df['numeric_column'].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[(df['numeric_column'] >= Q1 - 1.5 * IQR) & (df['numeric_column'] <= Q3 + 1.5 * IQR)]

# Z-score method for outlier detection
df['zscore'] = zscore(df['numeric_column'])
df_no_outliers = df[df['zscore'].abs() < 3]

# Scatter plot to detect anomalies
df.plot(kind='scatter', x='feature1', y='feature2', alpha=0.5)
plt.title("Scatter Plot for Anomaly Detection")
plt.show()

# ----------------------------
# 6. Examining Correlations
# ----------------------------

# Compute correlation matrix
print(df.corr())

# Visualize correlation matrix with heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# 7. Visualizing Data Distributions
# ----------------------------

# Pairplot of all numerical features
sns.pairplot(df)
plt.show()

# Histogram with seaborn
sns.histplot(df['numeric_column'], bins=30, kde=True)
plt.title("Histogram of Numeric Column")
plt.show()

# Boxplot for detecting skewness
sns.boxplot(x=df['numeric_column'])
plt.title("Boxplot of Numeric Column")
plt.show()

# ----------------------------
# 8. Detecting Duplicate Records
# ----------------------------

# Count duplicate rows
print(df.duplicated().sum())

# Remove duplicate records
df_no_duplicates = df.drop_duplicates()

# ----------------------------
# 9. Checking Dataset Balance (for classification)
# ----------------------------

# Count class distribution
print(df['target'].value_counts(normalize=True))

# Visualize class balance
sns.countplot(x='target', data=df)
plt.title("Class Distribution")
plt.show()
