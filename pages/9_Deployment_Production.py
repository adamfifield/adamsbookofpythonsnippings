import streamlit as st
#import joblib
#import pickle
#import requests
#from fastapi import FastAPI
#import uvicorn
#from code_executor import code_execution_widget # In-Browser Code Executor disabled

st.title("Deployment & Productionizing Models")

# Sidebar persistent code execution widget (temporarily disabled)
# code_execution_widget()

st.markdown("## 📌 Deploying Machine Learning Models")

# Expandable Section: Saving & Loading Models
with st.expander("💾 Saving & Loading Trained Models", expanded=False):
    st.markdown("### Saving a Model with Joblib")
    st.code('''
import joblib

# Save trained model
joblib.dump(model, "trained_model.pkl")

# Load model
model = joblib.load("trained_model.pkl")
''', language="python")

    st.markdown("### Saving a Model with Pickle")
    st.code('''
import pickle

# Save model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)
''', language="python")

# Expandable Section: Deploying with FastAPI
with st.expander("🚀 Deploying a Model with FastAPI", expanded=False):
    st.markdown("### Creating a FastAPI Endpoint")
    st.code('''
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("trained_model.pkl")

@app.get("/predict/")
def predict(feature: float):
    prediction = model.predict(np.array([[feature]]))
    return {"prediction": prediction.tolist()}
''', language="python")

    st.markdown("### Running the FastAPI Server")
    st.code('''
# Run FastAPI app (from command line)
uvicorn.run(app, host="0.0.0.0", port=8000)
''', language="python")

# Expandable Section: Making API Requests
with st.expander("🔗 Making API Requests to Deployed Models", expanded=False):
    st.markdown("### Sending Requests to FastAPI Endpoint")
    st.code('''
import requests

# Send a GET request to the FastAPI server
response = requests.get("http://127.0.0.1:8000/predict/?feature=5.5")
print(response.json())
''', language="python")

# Expandable Section: Deploying with Streamlit
with st.expander("📡 Deploying a Model with Streamlit", expanded=False):
    st.markdown("### Creating a Simple Web App for Predictions")
    st.code('''
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("trained_model.pkl")

st.title("ML Model Deployment")

# User input for prediction
feature = st.number_input("Enter a feature value", min_value=0.0, max_value=100.0, step=0.1)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[feature]]))
    st.write(f"Prediction: {prediction[0]}")
''', language="python")

# Expandable Section: Deploying with Docker
with st.expander("🐳 Containerizing with Docker", expanded=False):
    st.markdown("### Creating a Dockerfile")
    st.code('''
# Use an official Python runtime as base
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
''', language="dockerfile")

    st.markdown("### Building and Running a Docker Container")
    st.code('''
# Build the Docker image
docker build -t ml_model .

# Run the container
docker run -p 8000:8000 ml_model
''', language="bash")

# Expandable Section: Cloud Deployment
with st.expander("☁️ Deploying to Cloud Platforms", expanded=False):
    st.markdown("### Deploying with AWS Lambda")
    st.code('''
# Create a Lambda function with a trained ML model using AWS SDK
''', language="python")

    st.markdown("### Deploying with Google Cloud Run")
    st.code('''
# Deploy FastAPI model to Google Cloud Run
gcloud run deploy my-ml-api --image gcr.io/my-project/my-ml-api --platform managed
''', language="bash")

    st.markdown("### Deploying with Azure Functions")
    st.code('''
# Deploy ML model as an Azure Function
az functionapp create --resource-group myGroup --consumption-plan-location eastus --name my-ml-api --storage-account mystorage --runtime python
''', language="bash")
