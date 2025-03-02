import streamlit as st
import pandas as pd
import dask.dataframe as dd
from multiprocessing import Pool
import concurrent.futures
from code_executor import code_execution_widget

st.title("Big Data & Parallel Processing")

# Sidebar persistent code execution widget (temporarily disabled)
# code_execution_widget()

st.markdown("## 📌 Big Data Handling & Parallel Processing")

# Expandable Section: Working with Large Datasets using Dask
with st.expander("📊 Handling Large Datasets with Dask", expanded=False):
    st.markdown("### Using Dask for Scalable Pandas Operations")
    st.code('''
import dask.dataframe as dd

# Read large CSV with Dask
df = dd.read_csv("large_dataset.csv")

# Perform operations lazily (Dask computes only when needed)
df_mean = df.groupby("category_column")["value_column"].mean().compute()
print(df_mean)
''', language="python")

# Expandable Section: Parallel Processing with multiprocessing
with st.expander("⚡ Parallel Processing with multiprocessing", expanded=False):
    st.markdown("### Using multiprocessing for Parallel Execution")
    st.code('''
from multiprocessing import Pool

def square(x):
    return x * x

# Run function in parallel using multiple processes
with Pool(4) as p:
    results = p.map(square, [1, 2, 3, 4, 5])

print(results)  # [1, 4, 9, 16, 25]
''', language="python")

# Expandable Section: Parallel Processing with concurrent.futures
with st.expander("🚀 Using concurrent.futures for Asynchronous Execution", expanded=False):
    st.markdown("### Running Tasks in Parallel")
    st.code('''
import concurrent.futures

def square(x):
    return x * x

# Execute tasks asynchronously using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(square, [1, 2, 3, 4, 5]))

print(results)  # [1, 4, 9, 16, 25]
''', language="python")

# Expandable Section: Distributed Computing with Dask
with st.expander("🌐 Distributed Computing with Dask", expanded=False):
    st.markdown("### Running Computations Across Multiple Machines")
    st.code('''
from dask.distributed import Client

# Set up Dask client for distributed computing
client = Client(n_workers=4)
print(client)

# Perform distributed computation
df = dd.read_csv("large_dataset.csv")
df["new_col"] = df["existing_col"] * 2
df.compute()
''', language="python")

# Expandable Section: Processing Large Data with PySpark
with st.expander("🔥 Big Data Processing with PySpark", expanded=False):
    st.markdown("### Reading Large Data with PySpark")
    st.code('''
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("BigDataProcessing").getOrCreate()

# Read CSV into Spark DataFrame
df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)

# Show schema and first few rows
df.printSchema()
df.show(5)
''', language="python")

    st.markdown("### Performing Spark Data Transformations")
    st.code('''
# Group by and aggregate data
df_grouped = df.groupBy("category_column").agg({"value_column": "mean"})
df_grouped.show()
''', language="python")

# Expandable Section: Saving Processed Data Efficiently
with st.expander("💾 Saving & Exporting Large Data", expanded=False):
    st.markdown("### Saving Data with Dask")
    st.code('''
# Save large DataFrame efficiently
df.to_parquet("output_data.parquet", engine="pyarrow")
''', language="python")

    st.markdown("### Saving Data with PySpark")
    st.code('''
# Save DataFrame in Parquet format (efficient for big data)
df.write.parquet("output_data.parquet")
''', language="python")
