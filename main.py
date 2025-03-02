import streamlit as st
from pathlib import Path

st.title("Adam's Book of Python Snippings")

st.header("📂 Quick-Start Jupyter Notebooks")  # H2 Heading

st.markdown("These Jupyter notebooks contain structured reference implementations for various data science, machine learning, and AI workflows. Each entry includes direct access to the source files.")

# Define notebooks and their descriptions (formatted as keyword lists)
notebooks = [
    ("Connecting to External Data Sources",
     "pandas, requests, SQLAlchemy, pymongo, REST APIs, OAuth, pagination, authentication, query optimization, error handling.",
     "connecting_to_data_sources"),

    ("Data Wrangling",
     "pandas, NumPy, missing values, imputation, dropping, duplicates, pivot, melt, outliers, IQR, log transformation.",
     "data_wrangling"),

    ("Exploratory Data Analysis",
     "pandas, NumPy, Matplotlib, Seaborn, descriptive statistics, skewness, correlation heatmaps, boxplots, histograms, outliers.",
     "exploratory_data_analysis"),

    ("Feature Engineering & Selection",
     "pandas, Scikit-Learn, one-hot encoding, label encoding, MinMaxScaler, StandardScaler, interaction terms, feature selection, mutual information, decision trees.",
     "feature_engineering"),

    ("ML Preprocessing",
     "Scikit-Learn, pandas, missing value handling, mean/mode imputation, standardization, normalization, train-test split, stratified sampling, categorical encoding.",
     "ml_preprocessing"),

    ("Training & Evaluating ML Models",
     "Scikit-Learn, Linear Regression, Logistic Regression, Decision Trees, Random Forest, cross-validation, GridSearchCV, accuracy, precision, recall, F1-score, MSE, R².",
     "ml_training_evaluation"),

    ("Time Series Data Preparation",
     "pandas, NumPy, Statsmodels, datetime extraction, rolling window, stationarity (ADF test), differencing, feature engineering, ARIMA.",
     "time_series_analysis"),

    ("Deploying ML Models",
     "FastAPI, Streamlit, joblib, model serialization, REST API, HTTP requests, model inference, API endpoints, security, scalability.",
     "deploy_ml_models"),

    ("Model Interpretability",
     "SHAP, LIME, feature importance, partial dependence plots (PDPs), explainability, visualization, decision trees, interpretability strategies.",
     "model_interpretability"),

    ("Fine-Tuning Deep Learning",
     "PyTorch, TensorFlow, transfer learning, ResNet-50, freezing/unfreezing layers, dropout, weight decay, learning rate adjustment, differential training.",
     "fine_tuning_deep_learning"),

    ("LLM Inference & Fine-Tuning",
     "Hugging Face, Transformers, GPT, BERT, text generation, tokenization, dataset preparation, LoRA fine-tuning, quantization, model deployment.",
     "llm_fine_tuning"),

    ("Big Data with PySpark",
     "PySpark, Spark DataFrames, Parquet, CSV, schema definitions, caching, repartitioning, aggregations, distributed computing, performance optimization.",
     "big_data_pyspark"),
]

# Create a single-column structured list
for title, description, file_stem in notebooks:
    ipynb_path = Path(f"notebooks/{file_stem}.ipynb")  # Path for .ipynb file
    py_path = Path(f"notebooks/{file_stem}.py")  # Path for .py file

    with st.container():
        st.subheader(f"📘 {title}")  # H3 Heading
        st.write(description)

        # Align buttons in a row
        col1, col2 = st.columns([0.15, 0.85])  # Adjust width proportions for alignment
        with col1:
            st.download_button(
                label="📥 .ipynb",
                data=open(ipynb_path, "rb").read(),
                file_name=f"{file_stem}.ipynb",
                mime="application/octet-stream"
            )
        with col2:
            st.download_button(
                label="📥 .py",
                data=open(py_path, "rb").read(),
                file_name=f"{file_stem}.py",
                mime="application/octet-stream"
            )

        st.markdown("---")  # Separator for clarity
