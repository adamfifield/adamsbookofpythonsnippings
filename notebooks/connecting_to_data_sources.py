"""
Connecting to External Data Sources in Python

This script covers various ways to connect to external data sources, including:
- Reading local files (CSV, Excel, JSON, Parquet)
- Databases (PostgreSQL, MySQL, SQLite, MongoDB, Redis)
- Cloud storage services (AWS S3, Google Cloud, Azure Blob)
- Web APIs and web scraping
- Streaming data (Kafka, RabbitMQ)
- Other sources (Google Sheets, FTP/SFTP)
"""

# ----------------------------
# 1. Reading Local Files
# ----------------------------

# Read CSV file
import pandas as pd

df_csv = pd.read_csv("data.csv")  # Basic usage
df_csv_utf8 = pd.read_csv("data.csv", encoding="utf-8")  # Specifying encoding

# Read Excel file
import pandas as pd

df_excel = pd.read_excel("data.xlsx", engine="openpyxl")  # Read first sheet
df_excel_specific = pd.read_excel("data.xlsx", sheet_name="Sheet1")  # Specific sheet

# Read JSON file
import pandas as pd
import json

df_json = pd.read_json("data.json")  # Read JSON into DataFrame
with open("data.json", "r") as file:
    json_data = json.load(file)  # Load raw JSON

# Read Parquet file
import pandas as pd

df_parquet = pd.read_parquet("data.parquet", engine="pyarrow")

# ----------------------------
# 2. Database Connections (SQL & NoSQL)
# ----------------------------

# PostgreSQL
import psycopg2

pg_conn = psycopg2.connect(dbname="mydb", user="user", password="pass", host="localhost", port="5432")
pg_cursor = pg_conn.cursor()
pg_cursor.execute("SELECT * FROM my_table;")
pg_results = pg_cursor.fetchall()

# MySQL
import pymysql

mysql_conn = pymysql.connect(host="localhost", user="user", password="pass", database="mydb")
mysql_cursor = mysql_conn.cursor()
mysql_cursor.execute("SELECT * FROM my_table;")
mysql_results = mysql_cursor.fetchall()

# SQLite
import sqlite3

sqlite_conn = sqlite3.connect("database.db")
df_sqlite = pd.read_sql_query("SELECT * FROM my_table;", sqlite_conn)

# MongoDB
import pymongo

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["mydatabase"]
mongo_collection = mongo_db["mycollection"]
mongo_results = mongo_collection.find_one()

# Redis
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)
redis_client.set("key", "value")
redis_value = redis_client.get("key")

# ----------------------------
# 3. Cloud Storage
# ----------------------------

# AWS S3
import boto3

s3_client = boto3.client("s3")
s3_client.download_file("my-bucket", "data.csv", "downloaded_data.csv")

# Google Cloud Storage
from google.cloud import storage

gcs_client = storage.Client()
bucket = gcs_client.bucket("my-bucket")
blob = bucket.blob("data.csv")
blob.download_to_filename("downloaded_data.csv")

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient

azure_blob_service = BlobServiceClient.from_connection_string("your_connection_string")
azure_container = azure_blob_service.get_container_client("container-name")
azure_blob = azure_container.get_blob_client("data.csv")
with open("downloaded_data.csv", "wb") as file:
    file.write(azure_blob.download_blob().readall())

# ----------------------------
# 4. Web APIs & Web Scraping
# ----------------------------

# REST API (GET request)
import requests

response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
json_data = response.json()

# Web Scraping with BeautifulSoup
import requests
from bs4 import BeautifulSoup

html = requests.get("https://example.com").text
soup = BeautifulSoup(html, "html.parser")
title = soup.find("title").text

# ----------------------------
# 5. Streaming Data & Message Queues
# ----------------------------

# Kafka (Consumer)
from confluent_kafka import Consumer

consumer = Consumer({'bootstrap.servers': 'localhost:9092', 'group.id': 'mygroup'})
consumer.subscribe(['my_topic'])
msg = consumer.poll(1.0)  # Poll for messages

# RabbitMQ
import pika

rabbitmq_conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = rabbitmq_conn.channel()
channel.queue_declare(queue='hello')
channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')

# ----------------------------
# 6. Other Data Sources
# ----------------------------

# Google Sheets
import gspread

gc = gspread.service_account(filename="credentials.json")
sh = gc.open("MySheet")
worksheet = sh.sheet1
sheet_data = worksheet.get_all_records()

# FTP/SFTP
import paramiko

sftp = paramiko.SSHClient()
sftp.set_missing_host_key_policy(paramiko.AutoAddPolicy())
sftp.connect("sftp.example.com", username="user", password="password")
sftp_client = sftp.open_sftp()
sftp_client.get("remote_file.txt", "local_file.txt")
