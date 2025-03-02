import streamlit as st

# Set page title
st.set_page_config(page_title="Python Reference Manual", layout="wide")

# Sidebar Navigation
st.sidebar.title("📖 Sections")
sections = {
    "🐍 Python Basics & Best Practices": "1_Python_Basics",
    "🛠️ Data Wrangling & Transformation": "2_Data_Wrangling",
    "📊 Exploratory Data Analysis": "3_Exploratory_Data_Analysis",
    "🔄 ML Data Preprocessing": "4_ML_Preprocessing",
    "🤖 Machine Learning": "5_Machine_Learning",
    "🧠 Deep Learning": "6_Deep_Learning",
    "📜 NLP & Hugging Face": "7_NLP_HuggingFace",
    "🚀 Big Data & Parallel Processing": "8_Big_Data_Parallel_Processing",
    "🛠 Deployment & Productionizing ML Models": "9_Deployment_Production",
    "⚡ Advanced Topics": "10_Advanced_Topics"
}

selected_section = st.sidebar.radio("Select a Section:", list(sections.keys()))

# Redirect to selected section
st.sidebar.write(f"➡️ Navigate to: [**{selected_section}**](pages/{sections[selected_section]}.py)")

# Main header
st.title("📖 Python Reference Manual")
st.write("This interactive manual contains commonly used Python snippets for data science, machine learning, and AI.")

st.write("Use the **sidebar** for quick navigation between sections.")

st.markdown("---")
st.markdown("🔹 *Created with Streamlit*")
