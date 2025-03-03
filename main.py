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
