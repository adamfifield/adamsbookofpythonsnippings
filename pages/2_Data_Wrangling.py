import streamlit as st
# from code_executor import code_execution_widget # In-Browser Code Executor disabled

st.title("Data Wrangling")

# Sidebar persistent code execution widget
#code_execution_widget()

st.markdown("## 📌 Data Wrangling Techniques")

# Expandable Section: Reading Data from Files
with st.expander("📂 Reading Data - CSV, JSON, Excel", expanded=False):
    st.markdown("### Reading CSV Files")
    st.code('''
import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv("data.csv")
print(df.head())
''', language="python")

    st.markdown("### Reading JSON Files")
    st.code('''
import pandas as pd

# Read a JSON file into a DataFrame
df = pd.read_json("data.json")
print(df.head())
''', language="python")

    st.markdown("### Reading Excel Files")
    st.code('''
import pandas as pd

# Read an Excel file into a DataFrame
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(df.head())
''', language="python")

# Expandable Section: Handling Missing Data
with st.expander("🚫 Handling Missing Data", expanded=False):
    st.markdown("### Checking for Missing Values")
    st.code('''
import pandas as pd

# Check for missing values
df = pd.read_csv("data.csv")
print(df.isnull().sum())
''', language="python")

    st.markdown("### Filling Missing Values")
    st.code('''
# Fill missing values with the column mean
df.fillna(df.mean(), inplace=True)
print(df.head())
''', language="python")

    st.markdown("### Dropping Missing Values")
    st.code('''
# Drop rows with missing values
df.dropna(inplace=True)
print(df.head())
''', language="python")

# Expandable Section: Filtering & Selecting Data
with st.expander("🔍 Filtering & Selecting Data", expanded=False):
    st.markdown("### Selecting Columns")
    st.code('''
# Select a single column
print(df["column_name"])

# Select multiple columns
print(df[["column1", "column2"]])
''', language="python")

    st.markdown("### Filtering Rows")
    st.code('''
# Filter rows where column value is greater than 50
filtered_df = df[df["column_name"] > 50]
print(filtered_df)
''', language="python")

# Expandable Section: Data Transformation
with st.expander("🔄 Data Transformation", expanded=False):
    st.markdown("### Applying Functions to Columns")
    st.code('''
# Apply a function to transform a column
df["new_column"] = df["existing_column"].apply(lambda x: x * 2)
print(df.head())
''', language="python")

    st.markdown("### Renaming Columns")
    st.code('''
# Rename specific columns
df.rename(columns={"old_name": "new_name"}, inplace=True)
print(df.head())
''', language="python")

    st.markdown("### Changing Data Types")
    st.code('''
# Convert a column to integer
df["column_name"] = df["column_name"].astype(int)
print(df.dtypes)
''', language="python")

# Expandable Section: Grouping & Aggregation
with st.expander("📊 Grouping & Aggregation", expanded=False):
    st.markdown("### Grouping by a Column")
    st.code('''
# Group by a column and calculate the mean
grouped = df.groupby("category_column")["value_column"].mean()
print(grouped)
''', language="python")

    st.markdown("### Multiple Aggregations")
    st.code('''
# Perform multiple aggregations at once
aggregated = df.groupby("category_column").agg({"value_column": ["mean", "sum", "count"]})
print(aggregated)
''', language="python")

# Expandable Section: Merging & Joining Data
with st.expander("🔗 Merging & Joining DataFrames", expanded=False):
    st.markdown("### Merging Two DataFrames")
    st.code('''
# Merge two DataFrames on a common column
df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
df2 = pd.DataFrame({"id": [1, 2, 3], "score": [85, 90, 78]})

merged_df = pd.merge(df1, df2, on="id")
print(merged_df)
''', language="python")

    st.markdown("### Different Types of Joins")
    st.code('''
# Perform an outer join
merged_df = pd.merge(df1, df2, on="id", how="outer")
print(merged_df)
''', language="python")

# Expandable Section: Reshaping Data
with st.expander("📐 Reshaping Data", expanded=False):
    st.markdown("### Pivot Tables")
    st.code('''
# Create a pivot table
pivot = df.pivot_table(index="category", values="value", aggfunc="sum")
print(pivot)
''', language="python")

    st.markdown("### Melting (Unpivoting) Data")
    st.code('''
# Convert wide-format data to long-format
melted = df.melt(id_vars=["id"], value_vars=["column1", "column2"])
print(melted)
''', language="python")

# Expandable Section: Working with Dates & Time
with st.expander("⏳ Working with Dates & Time", expanded=False):
    st.markdown("### Converting Strings to Datetime")
    st.code('''
# Convert a string column to datetime
df["date_column"] = pd.to_datetime(df["date_column"])
print(df.dtypes)
''', language="python")

    st.markdown("### Extracting Date Components")
    st.code('''
# Extract year, month, and day from a datetime column
df["year"] = df["date_column"].dt.year
df["month"] = df["date_column"].dt.month
df["day"] = df["date_column"].dt.day
print(df.head())
''', language="python")

# Expandable Section: Exporting Data
with st.expander("💾 Exporting Data", expanded=False):
    st.markdown("### Saving to CSV")
    st.code('''
# Save DataFrame to a CSV file
df.to_csv("output.csv", index=False)
''', language="python")

    st.markdown("### Saving to Excel")
    st.code('''
# Save DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)
''', language="python")
