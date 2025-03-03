"""
Model Deployment in Python

This script covers various essential model deployment techniques, including:
- Saving & loading models
- Building an API for model inference
- Serving models in production
- Performance optimization
- Monitoring & logging
- Scaling & load balancing
- Security best practices
"""

import pickle
import joblib

# ----------------------------
# 1. Saving & Loading Models
# ----------------------------

# Save model using joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
joblib.dump(model, "model.pkl")

# Load model
loaded_model = joblib.load("model.pkl")

# Save model using pickle
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model using pickle
with open("model.pkl", "rb") as f:
    loaded_model_pickle = pickle.load(f)

# ----------------------------
# 2. Building an API for Model Inference
# ----------------------------

# Create a FastAPI server to serve model predictions
from fastapi import FastAPI
import uvicorn
import pandas as pd

app = FastAPI()

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = loaded_model.predict(df)
    return {"prediction": int(prediction[0])}

# Run FastAPI server (for local testing)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# ----------------------------
# 3. Serving Models in Production
# ----------------------------

# Example: Storing a Dockerfile separately
dockerfile_content = """
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# ----------------------------
# 4. Performance Optimization
# ----------------------------

# Use Redis caching to store predictions and speed up inference
import redis
import json

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def cache_prediction(input_data, prediction):
    redis_client.set(json.dumps(input_data), json.dumps(prediction))

def get_cached_prediction(input_data):
    cached = redis_client.get(json.dumps(input_data))
    return json.loads(cached) if cached else None

# ----------------------------
# 5. Monitoring & Logging
# ----------------------------

import logging

logging.basicConfig(filename="model_api.log", level=logging.INFO)

def log_request(data, prediction):
    logging.info(f"Input: {data}, Prediction: {prediction}")

# ----------------------------
# 6. Scaling & Load Balancing
# ----------------------------

# Example: Using Gunicorn to serve FastAPI efficiently
gunicorn_command = "gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app"

# Kubernetes YAML deployment file stored separately
kubernetes_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-api
  template:
    metadata:
      labels:
        app: model-api
    spec:
      containers:
      - name: model-api
        image: my_model_api_image
        ports:
        - containerPort: 8000
"""

# ----------------------------
# 7. Security Best Practices
# ----------------------------

# Secure API with authentication (example using API key validation)
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException

API_KEY = "mysecureapikey"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

@app.post("/secure_predict/")
def secure_predict(data: dict, api_key: str = Depends(verify_api_key)):
    df = pd.DataFrame([data])
    prediction = loaded_model.predict(df)
    return {"prediction": int(prediction[0])}
