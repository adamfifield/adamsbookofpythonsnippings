import streamlit as st
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from code_executor import code_execution_widget

st.title("Natural Language Processing (NLP) with Hugging Face")

# Sidebar persistent code execution widget (temporarily disabled)
# code_execution_widget()

st.markdown("## 📌 NLP with Hugging Face Transformers")

# Expandable Section: Introduction to NLP & Hugging Face
with st.expander("📖 Introduction to NLP & Hugging Face", expanded=False):
    st.markdown("### What is NLP?")
    st.write("Natural Language Processing (NLP) enables machines to understand and process human language. Hugging Face provides pre-trained models for various NLP tasks.")

    st.markdown("### Installing Hugging Face Transformers")
    st.code('''
# Install the Hugging Face Transformers library
!pip install transformers
''', language="bash")

# Expandable Section: Sentiment Analysis
with st.expander("😊 Sentiment Analysis with Pretrained Models", expanded=False):
    st.markdown("### Using a Pretrained Sentiment Analysis Model")
    st.code('''
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze sentiment of a sentence
result = sentiment_analyzer("I love using Hugging Face!")
print(result)
''', language="python")

# Expandable Section: Text Generation with GPT-based Models
with st.expander("📝 Text Generation with GPT", expanded=False):
    st.markdown("### Using GPT-2 for Text Generation")
    st.code('''
from transformers import pipeline

# Load text generation pipeline
text_generator = pipeline("text-generation", model="gpt2")

# Generate text from a prompt
result = text_generator("Once upon a time", max_length=50)
print(result)
''', language="python")

# Expandable Section: Named Entity Recognition (NER)
with st.expander("🔍 Named Entity Recognition (NER)", expanded=False):
    st.markdown("### Extracting Named Entities from Text")
    st.code('''
from transformers import pipeline

# Load named entity recognition pipeline
ner = pipeline("ner")

# Identify entities in text
result = ner("Elon Musk founded SpaceX in 2002.")
print(result)
''', language="python")

# Expandable Section: Text Summarization
with st.expander("📜 Text Summarization", expanded=False):
    st.markdown("### Summarizing Long Texts")
    st.code('''
from transformers import pipeline

# Load text summarization pipeline
summarizer = pipeline("summarization")

# Summarize text
text = "Hugging Face provides state-of-the-art NLP models for various tasks, including text classification, question answering, and summarization."
summary = summarizer(text, max_length=30, min_length=10)
print(summary)
''', language="python")

# Expandable Section: Machine Translation
with st.expander("🌍 Machine Translation", expanded=False):
    st.markdown("### Translating Text Between Languages")
    st.code('''
from transformers import pipeline

# Load translation pipeline (English to French)
translator = pipeline("translation_en_to_fr")

# Translate text
result = translator("Hello, how are you?")
print(result)
''', language="python")

# Expandable Section: Fine-Tuning a Hugging Face Model
with st.expander("🛠️ Fine-Tuning a Transformer Model", expanded=False):
    st.markdown("### Training a Custom Text Classifier")
    st.code('''
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize input text
inputs = tokenizer("This is an example text.", return_tensors="pt")

# Get model output
outputs = model(**inputs)
print(outputs.logits)
''', language="python")

# Expandable Section: Saving & Loading Models
with st.expander("💾 Saving & Loading Hugging Face Models", expanded=False):
    st.markdown("### Saving a Fine-Tuned Model")
    st.code('''
# Save model and tokenizer
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")
''', language="python")

    st.markdown("### Loading a Saved Model")
    st.code('''
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("my_model")
tokenizer = AutoTokenizer.from_pretrained("my_model")
''', language="python")
