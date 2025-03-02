import streamlit as st
from pathlib import Path

st.title("Adam's Book of Python Snippings")
st.markdown("---")

st.header("📂 Quick-Start Jupyter Notebooks")

st.markdown("*These Jupyter notebooks contain structured reference implementations for various data science, machine learning, and AI workflows. Each entry includes direct access to the source files.*")
st.write("")
st.write("")
st.write("")

# Define notebooks and their descriptions (formatted as keyword lists)
notebooks = [
    ("Connecting to External Data Sources", "pandas, requests, SQLAlchemy, pymongo, REST APIs, OAuth, pagination, authentication.", "connecting_to_data_sources"),
    ("Data Wrangling", "pandas, NumPy, missing values, imputation, duplicates, outliers, pivot, melt.", "data_wrangling"),
    ("Exploratory Data Analysis", "pandas, NumPy, Matplotlib, Seaborn, histograms, boxplots, correlation heatmaps.", "exploratory_data_analysis"),
    ("Feature Engineering & Selection", "Scikit-Learn, one-hot encoding, label encoding, feature scaling, feature selection.", "feature_engineering"),
    ("ML Preprocessing", "Scikit-Learn, categorical encoding, normalization, train-test split, imputation.", "ml_preprocessing"),
    ("Training & Evaluating ML Models", "Scikit-Learn, classification, regression, cross-validation, GridSearchCV, metrics.", "ml_training_evaluation"),
    ("Time Series Data Preparation", "pandas, Statsmodels, rolling window, stationarity (ADF test), differencing.", "time_series_analysis"),
    ("Deploying ML Models", "FastAPI, Streamlit, joblib, REST API, model inference, security.", "deploy_ml_models"),
    ("Model Interpretability", "SHAP, LIME, feature importance, PDPs, decision trees.", "model_interpretability"),
    ("Fine-Tuning Deep Learning", "PyTorch, TensorFlow, ResNet-50, freezing/unfreezing layers, dropout.", "fine_tuning_deep_learning"),
    ("LLM Inference & Fine-Tuning", "Hugging Face, Transformers, GPT, BERT, LoRA, quantization.", "llm_fine_tuning"),
    ("Big Data with PySpark", "PySpark, Spark DataFrames, Parquet, CSV, schema definitions, aggregations.", "big_data_pyspark"),
]

# Create two main columns
col1, col2 = st.columns(2)

# Define a min height for consistency
MIN_HEIGHT = 0  # Adjust this based on testing

for idx, (title, keywords, file_stem) in enumerate(notebooks):
    ipynb_path = Path(f"notebooks/{file_stem}.ipynb")
    py_path = Path(f"notebooks/{file_stem}.py")

    # Alternate between columns
    with (col1 if idx % 2 == 0 else col2):
        # Use a container with min height padding
        with st.container():
            st.markdown(f"###### 📘 {title}")  # H5 Heading
            st.write(keywords)

            # Nested columns for buttons
            btn_col1, btn_col2 = st.columns([0.3, 0.7])
            with btn_col1:
                st.download_button("📥 .ipynb", open(ipynb_path, "rb").read(), f"{file_stem}.ipynb")
            with btn_col2:
                st.download_button("📥 .py", open(py_path, "rb").read(), f"{file_stem}.py")

            # Padding to ensure uniform height
            st.markdown(f"<div style='min-height: {MIN_HEIGHT}px'></div>", unsafe_allow_html=True)

        st.markdown("---")  # Separator for clarity
