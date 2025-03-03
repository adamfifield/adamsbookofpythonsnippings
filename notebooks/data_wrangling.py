"""
Data Wrangling in Python

This script covers various essential data wrangling operations, including:
- Loading data
- Handling missing values
- Data type conversions
- String operations
- Aggregation & Grouping
- Handling duplicates & outliers
- Merging & Joining datasets
- Reshaping data
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Data Importing & Initial Exploration
# ----------------------------

# Load CSV
df = pd.read_csv("data.csv")

# Load Excel
df_excel = pd.read_excel("data.xlsx", sheet_name="Sheet1", engine="openpyxl")

# Load JSON
df_json = pd.read_json("data.json")

# Inspect data
print(df.info())
print(df.describe())
print(df.head())

# Check missing values
print(df.isnull().sum())

# ----------------------------
# 2. Handling Missing Data
# ----------------------------

# Drop missing values
df_cleaned = df.dropna()
df_cleaned_subset = df.dropna(subset=['column1'])

# Fill missing values
df_filled = df.fillna(0)
df_filled_column = df['column1'].fillna(df['column1'].mean())

# Forward and backward fill
df_ffill = df.fillna(method='ffill')
df_bfill = df.fillna(method='bfill')

# ----------------------------
# 3. Data Type Conversions
# ----------------------------

# Convert column to numeric
df['column1'] = pd.to_numeric(df['column1'], errors='coerce')

# Convert to datetime
df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

# Convert categorical variables
df['category_column'] = df['category_column'].astype('category')
df_encoded = pd.get_dummies(df, columns=['category_column'])

# ----------------------------
# 4. String Operations
# ----------------------------

# Convert to lowercase
df['text_column'] = df['text_column'].str.lower()

# Remove special characters
df['text_column'] = df['text_column'].str.replace('[^a-zA-Z0-9]', '', regex=True)

# Extract text pattern using regex
df['phone_numbers'] = df['text_column'].str.extract(r'(\d{3}-\d{3}-\d{4})')

# Splitting and joining text
df[['first_name', 'last_name']] = df['full_name'].str.split(' ', expand=True)
df['recombined'] = df['first_name'] + ' ' + df['last_name']

# ----------------------------
# 5. Aggregation & Grouping
# ----------------------------

# Grouping and summarizing
df_grouped = df.groupby('category_column')['numeric_column'].mean()

# Pivot table
df_pivot = df.pivot_table(index='category_column', values='numeric_column', aggfunc='sum')

# Rolling average
df['rolling_avg'] = df['numeric_column'].rolling(window=3).mean()

# ----------------------------
# 6. Handling Duplicates & Outliers
# ----------------------------

# Remove duplicates
df_no_duplicates = df.drop_duplicates()

# Identify outliers using IQR
Q1 = df['numeric_column'].quantile(0.25)
Q3 = df['numeric_column'].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[(df['numeric_column'] >= Q1 - 1.5 * IQR) & (df['numeric_column'] <= Q3 + 1.5 * IQR)]

# Capping outliers
df['capped_column'] = df['numeric_column'].clip(lower=Q1, upper=Q3)

# ----------------------------
# 7. Data Merging & Joining
# ----------------------------

# Concatenation
df_concat = pd.concat([df, df], axis=0)

# Merging
df_merged = df.merge(df, on="common_column", how="inner")
df_left = df.merge(df, on="common_column", how="left")
df_outer = df.merge(df, on="common_column", how="outer")

# ----------------------------
# 8. Reshaping Data
# ----------------------------

# Melting (wide to long)
df_melted = df.melt(id_vars=['category_column'], value_vars=['numeric_column1', 'numeric_column2'])

# Pivoting (long to wide)
df_pivoted = df.pivot(index='category_column', columns='date_column', values='numeric_column')

# Stacking
df_stacked = df.set_index(['category_column', 'date_column']).stack()
