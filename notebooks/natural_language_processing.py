"""
Natural Language Processing (NLP) in Python

This script covers various essential NLP techniques, including:
- Text preprocessing
- Stopwords removal & lemmatization
- Feature extraction & vectorization
- Named Entity Recognition (NER) & POS tagging
- Text classification & sentiment analysis
- Topic modeling
- Text summarization
- Word embeddings & transformer models
- Deploying NLP models
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Text Preprocessing
# ----------------------------

import re
import nltk
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = nltk.word_tokenize(text)  # Tokenize text
    return tokens

text_sample = "Natural Language Processing is amazing! #AI #MachineLearning"
processed_text = preprocess_text(text_sample)
print(processed_text)

# ----------------------------
# 2. Stopwords Removal & Lemmatization
# ----------------------------

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

cleaned_text = clean_tokens(processed_text)
print(cleaned_text)

# ----------------------------
# 3. Feature Extraction & Vectorization
# ----------------------------

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = ["Machine learning is fun.", "Natural language processing is interesting."]
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(documents)
print(X_bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(documents)
print(X_tfidf.toarray())

# ----------------------------
# 4. Named Entity Recognition (NER) & POS Tagging
# ----------------------------

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Barack Obama was the 44th president of the United States.")

# POS Tagging
for token in doc:
    print(f"{token.text} - {token.pos_}")

# Named Entity Recognition (NER)
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")

# ----------------------------
# 5. Text Classification & Sentiment Analysis
# ----------------------------

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = pd.DataFrame({
    "text": ["I love this!", "This is terrible!", "Absolutely amazing!", "Worst experience ever."],
    "label": [1, 0, 1, 0]
})

X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)

# ----------------------------
# 6. Topic Modeling (LDA)
# ----------------------------

from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer()
X_topics = vectorizer.fit_transform(documents)

lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_model.fit(X_topics)

# ----------------------------
# 7. Text Summarization
# ----------------------------

from gensim.summarization.summarizer import summarize

text = ("Natural language processing enables machines to understand human language. "
        "It allows applications like chatbots, voice assistants, and sentiment analysis.")

summary = summarize(text, ratio=0.5)
print(summary)

# ----------------------------
# 8. Word Embeddings & Transformer Models
# ----------------------------

from gensim.models import Word2Vec

sentences = [["machine", "learning", "is", "great"], ["natural", "language", "processing", "rocks"]]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

print(word2vec_model.wv["machine"])

# ----------------------------
# 9. Deploying NLP Models
# ----------------------------

from fastapi import FastAPI
import joblib

app = FastAPI()

@app.post("/classify/")
def classify(text: str):
    model = joblib.load("text_classifier.pkl")
    prediction = model.predict([text])
    return {"prediction": int(prediction[0])}

# Run FastAPI server (for local testing)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# ----------------------------
# END OF SCRIPT
# ----------------------------
