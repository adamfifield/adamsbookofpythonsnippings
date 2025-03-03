import streamlit as st
from pathlib import Path

st.title("Adam's Book of Python Snippings")

st.markdown(
    "This is your quick-start page for different data science workflows in python. Select a topic or workflow from"
    "the list below and you will be given a checklist of items to consider, caveats and steps to perform, followed"
    " by a block of python code with ready-to-use examples."
)

# Define available topics with corresponding file paths
notebooks = {
    "Connecting to External Data Sources": {
        "md_file": "connecting_external_data.md",
        "py_file": "connecting_to_data_sources.py",
    },
    "Data Wrangling": {
        "md_file": "data_wrangling.md",
        "py_file": "data_wrangling.py",
    },
    "Exploratory Data Analysis (EDA)": {
        "md_file": "exploratory_data_analysis.md",
        "py_file": "exploratory_data_analysis.py",
    },
    "Anomaly Detection": {
        "md_file": "anomaly_detection.md",
        "py_file": "anomaly_detection.py",
    },
    "Feature Engineering": {
        "md_file": "feature_engineering.md",
        "py_file": "feature_engineering.py",
    },
    "Model Training & Evaluation": {
        "md_file": "model_training_evaluation.md",
        "py_file": "model_training_evaluation.py",
    },
    "Model Deployment": {
        "md_file": "model_deployment.md",
        "py_file": "model_deployment.py",
    },
    "Time Series Forecasting": {
        "md_file": "time_series_forecasting.md",
        "py_file": "time_series_forecasting.py",
    },
    "Natural Language Processing (NLP)": {
        "md_file": "natural_language_processing.md",
        "py_file": "natural_language_processing.py",
    },
    "Reinforcement Learning": {
        "md_file": "reinforcement_learning.md",
        "py_file": "reinforcement_learning.py",
    },
    "Computer Vision": {
        "md_file": "computer_vision.md",
        "py_file": "computer_vision.py",
    },
    "Recommendation Systems": {
        "md_file": "recommendation_systems.md",
        "py_file": "recommendation_systems.py",
    },
    "Graph-Based Machine Learning": {
        "md_file": "graph_ml.md",
        "py_file": "graph_ml.py",
    },
    "Fine-Tuning Deep Learning Models": {
        "md_file": "fine_tuning_dl.md",
        "py_file": "fine_tuning_dl.py",
    },
    "Fine-Tuning LLMs": {
        "md_file": "llm_fine_tuning.md",
        "py_file": "llm_fine_tuning.py",
    },
    "Big Data with PySpark": {
        "md_file": "big_data_pyspark.md",
        "py_file": "big_data_pyspark.py",
    },
    "Model Interpretability & Explainability": {
        "md_file": "model_interpretability.md",
        "py_file": "model_interpretability.py",
    }
}


# Dropdown for selecting a topic
selected_topic = st.selectbox("Select a topic to begin:", [""] + list(notebooks.keys()))

# Display content only when a selection is made
if selected_topic:
    topic_files = notebooks[selected_topic]
    md_file_path = Path(f"notebooks/{topic_files['md_file']}")
    py_file_path = Path(f"notebooks/{topic_files['py_file']}")

    # Show Markdown checklist if file exists
    if md_file_path.exists():
        with open(md_file_path, "r") as md_file:
            st.markdown(md_file.read())

    else:
        st.warning(f"Checklist file `{topic_files['md_file']}` not found.")

    # Show full Python script if file exists
    if py_file_path.exists():
        with open(py_file_path, "r") as py_file:
            code_content = py_file.read()

        st.subheader("📜 Full Python Code")
        st.code(code_content, language="python")

    else:
        st.warning(f"Python file `{topic_files['py_file']}` not found.")
