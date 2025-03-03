# 📖 Data Wrangling

### **Description**  
This section covers essential data wrangling techniques using `pandas`, a key library for data manipulation in Python. It includes methods for **loading and inspecting data**, **handling missing values**, **converting data types**, **string operations**, **aggregation and grouping**, **detecting and handling duplicates and outliers**, **merging datasets**, and **reshaping data**. These techniques are fundamental for cleaning and transforming raw data into a structured format suitable for analysis or mach...

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Inspect the dataset before applying transformations**  
  - Use `df.info()` to check column data types and null values.  
  - Use `df.describe()` for summary statistics.  
  - Use `df.head()` and `df.tail()` to inspect sample rows.  

- ✅ **Handle missing values carefully**  
  - Use `df.isnull().sum()` to count missing values per column.  
  - Use `df.dropna()` to remove rows with missing data (ensure this is appropriate).  
  - Use `df.fillna(value)` to replace missing values with a specific default.  
  - Consider forward (`method='ffill'`) or backward filling (`method='bfill'`) for time-series data.  

- ✅ **Ensure correct data types**  
  - Convert numeric columns explicitly using `pd.to_numeric(df['col'], errors='coerce')`.  
  - Convert date columns with `pd.to_datetime(df['date_col'], errors='coerce')`.  
  - Convert categorical columns using `df['col'] = df['col'].astype('category')`.  

- ✅ **Optimize text processing for large datasets**  
  - Use `df['col'].str.lower()` to normalize case.  
  - Use `df['col'].str.replace('[^a-zA-Z0-9]', '', regex=True)` to remove special characters.  
  - Use `df['col'].str.extract(r'pattern')` to extract patterns from text.  

- ✅ **Use efficient numerical operations**  
  - Avoid looping over rows; prefer vectorized operations like `df['new_col'] = df['col1'] + df['col2']`.  
  - Normalize numerical features for ML using `(df['col'] - df['col'].mean()) / df['col'].std()`.  

- ✅ **Handle duplicates and outliers properly**  
  - Use `df.duplicated().sum()` to check for duplicate rows.  
  - Use `df.drop_duplicates()` to remove duplicates.  
  - Use IQR-based filtering for outliers: `Q1 = df['col'].quantile(0.25)`, `Q3 = df['col'].quantile(0.75)`, `IQR = Q3 - Q1`.  
  - Use `df.clip(lower, upper)` to cap extreme values.  

- ✅ **Use correct merge strategies to avoid data loss**  
  - Use `df1.merge(df2, on='key', how='inner')` to merge datasets while keeping common records.  
  - Use `df1.merge(df2, on='key', how='outer')` to retain all records from both datasets.  
  - Use `df1.merge(df2, on='key', how='left')` to keep all records from the left table.  

- ✅ **Reshape data when necessary**  
  - Use `df.melt(id_vars=['id'], value_vars=['col1', 'col2'])` to convert wide data to long format.  
  - Use `df.pivot(index='id', columns='category', values='value')` to reshape long data into wide format.  
  - Use `df.set_index(['col1', 'col2']).stack()` to convert columns into row indices.  
