import streamlit as st
import io
import sys

def code_execution_widget():
    st.sidebar.markdown("## 🖥️ Try Out the Code")

    code = st.sidebar.text_area("Enter Python code:", '''# Example: Modify or paste your own code here
print("Hello, Streamlit!")
''', height=200)

    if st.sidebar.button("Run Code"):
        output_buffer = io.StringIO()
        sys.stdout = output_buffer  # Redirect stdout to capture print output

        try:
            exec(code)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

        sys.stdout = sys.__stdout__  # Reset stdout
        output = output_buffer.getvalue()  # Get captured output

        if output:
            st.sidebar.text_area("Output:", output, height=150)
        else:
            st.sidebar.success("Code executed successfully! (No print output)")
