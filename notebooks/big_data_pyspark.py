"""
Big Data with PySpark

This script covers techniques for:
- Setting up Spark and loading big data
- Applying transformations and Spark SQL queries
- Optimizing performance for large-scale datasets
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ----------------------------
# 1. Setting Up PySpark
# ----------------------------

spark = SparkSession.builder.appName("BigDataProcessing").getOrCreate()

# ----------------------------
# 2. Loading and Transforming Data
# ----------------------------

df = spark.read.csv("big_data.csv", header=True, inferSchema=True)
df = df.withColumn("new_column", col("existing_column") * 2)

# ----------------------------
# 3. Running SQL Queries
# ----------------------------

df.createOrReplaceTempView("data_table")
spark.sql("SELECT new_column, COUNT(*) FROM data_table GROUP BY new_column").show()
