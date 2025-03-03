# 📖 Exploratory Data Analysis (EDA)

### **Description**  
This section covers key techniques for **understanding dataset structure**, **handling missing values**, **analyzing categorical and numerical variables**, **identifying outliers**, **examining correlations**, **visualizing distributions**, and **checking dataset balance**. These techniques provide critical insights before proceeding with modeling or deeper analysis.

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Understand the dataset structure**  
  - Use `df.info()` to check column types and null values.  
  - Use `df.describe()` for summary statistics.  
  - Use `df.shape` to get the number of rows and columns.  

- ✅ **Check for missing values**  
  - Use `df.isnull().sum()` to identify missing data.  
  - Use `df.dropna()` to remove missing values if necessary.  
  - Use `df.fillna(value)` to fill missing values with an appropriate default.  

- ✅ **Analyze categorical variables**  
  - Use `df['col'].value_counts()` to inspect category distributions.  
  - Use `df.groupby('col').size()` for frequency counts.  
  - Convert categorical columns using `pd.get_dummies()`.  

- ✅ **Analyze numerical variables**  
  - Use `df['col'].mean(), df['col'].median(), df['col'].std()` for basic statistics.  
  - Use `df.hist()` to visualize distributions.  
  - Use `df.boxplot(column='col')` to detect outliers.  

- ✅ **Identify outliers and anomalies**  
  - Use IQR method to detect outliers (`Q1`, `Q3`, `IQR` calculations).  
  - Use Z-score with `scipy.stats.zscore()` to identify extreme values.  
  - Use `df.plot(kind='scatter', x='feature1', y='feature2')` to spot anomalies.  

- ✅ **Examine correlations**  
  - Use `df.corr()` to compute correlation matrix.  
  - Use `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')` for visualization.  
  - Check for multicollinearity using variance inflation factor (VIF).  

- ✅ **Visualize data distributions**  
  - Use `sns.pairplot(df)` to plot relationships between features.  
  - Use `sns.histplot(df['col'], bins=30)` to visualize distributions.  
  - Use `sns.boxplot(x=df['col'])` to examine skewness and outliers.  

- ✅ **Detect duplicate records**  
  - Use `df.duplicated().sum()` to count duplicate rows.  
  - Use `df.drop_duplicates()` to remove them if necessary.  

- ✅ **Check dataset balance (for classification problems)**  
  - Use `df['target'].value_counts(normalize=True)` to check class distribution.  
  - Use `sns.countplot(x='target', data=df)` to visualize class balance.  
