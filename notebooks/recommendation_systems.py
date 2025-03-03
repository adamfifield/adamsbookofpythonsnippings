"""
Recommendation Systems in Python

This script covers various recommendation system techniques, including:
- Collaborative filtering
- Content-based filtering
- Hybrid recommendation systems
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Collaborative Filtering
# ----------------------------

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin("ml-100k")
algo = SVD()

cross_validate(algo, data, cv=5)

# ----------------------------
# 2. Content-Based Filtering
# ----------------------------

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
movies = pd.DataFrame({"title": ["Movie1", "Movie2"], "description": ["Sci-fi action", "Romantic drama"]})

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["description"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# ----------------------------
# 3. Hybrid Recommendation Systems
# ----------------------------

class HybridRecommender:
    def __init__(self, collaborative_model, content_similarity_matrix):
        self.collaborative_model = collaborative_model
        self.content_similarity_matrix = content_similarity_matrix

    def recommend(self, user_id, item_id):
        cf_score = self.collaborative_model.predict(user_id, item_id).est
        content_score = np.mean(self.content_similarity_matrix[item_id])
        return (cf_score + content_score) / 2
