import streamlit as st
from pathlib import Path

st.title("Python Reference Manual")

# Define notebooks and their descriptions
notebooks = [
    ("📂 Connecting to External Data Sources", "Learn how to connect to APIs, PostgreSQL, and MongoDB.", "connecting_to_data_sources.ipynb"),
    ("🛠️ Data Wrangling", "Clean and preprocess data effectively using Pandas.", "data_wrangling.ipynb"),
    ("📊 Exploratory Data Analysis", "Perform statistical analysis and visualizations.", "exploratory_data_analysis.ipynb"),
    ("🏆 Feature Engineering & Selection", "Enhance dataset features for better model performance.", "feature_engineering.ipynb"),
    ("🎯 ML Preprocessing", "Prepare datasets for machine learning models.", "ml_preprocessing.ipynb"),
    ("🤖 Training & Evaluating ML Models", "Train and evaluate regression and classification models.", "ml_training_evaluation.ipynb"),
    ("⏳ Time Series Data Preparation", "Handle time-based datasets for forecasting.", "time_series_analysis.ipynb"),
    ("🚀 Deploying ML Models", "Use FastAPI & Streamlit to serve ML models.", "deploy_ml_models.ipynb"),
    ("🔍 Model Interpretability", "Explain ML model predictions using SHAP & LIME.", "model_interpretability.ipynb"),
    ("🏋️ Fine-Tuning Deep Learning", "Optimize pretrained models using transfer learning.", "fine_tuning_deep_learning.ipynb"),
    ("🧠 LLM Inference & Fine-Tuning", "Run & fine-tune Large Language Models (GPT, BERT).", "llm_fine_tuning.ipynb"),
    ("🔥 Big Data with PySpark", "Process massive datasets efficiently using PySpark.", "big_data_pyspark.ipynb"),
]

st.title("📂 Quick-Start Jupyter Notebooks")

st.markdown("These downloadable Jupyter notebooks contain ready-to-use templates for common data science and machine learning workflows.")

# Create a single-column structured list
for title, description, file in notebooks:
    file_path = Path(f"notebooks/{file}")  # Ensure correct file path

    with st.container():
        st.write(f"**{title}** - {description}")
        st.download_button(
            label="📥 Download",
            data=open(file_path, "rb").read(),
            file_name=file,
            mime="application/octet-stream"
        )
        st.markdown("---")  # Separator for clarity
