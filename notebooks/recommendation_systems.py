"""
Comprehensive Recommendation Systems in Python

This script covers various recommendation system techniques, including:
- User-based and item-based collaborative filtering
- Content-based filtering with text similarity
- Hybrid recommendation systems combining multiple approaches
- Neural networks for recommendation models
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. User-Based and Item-Based Collaborative Filtering
# ----------------------------

from surprise import SVD, KNNBasic, Dataset, Reader
from surprise.model_selection import cross_validate

ratings_dict = {"user_id": [1, 1, 2, 2, 3, 3], "item_id": [101, 102, 101, 103, 102, 103], "rating": [5, 4, 4, 3, 5, 2]}
df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

# Train SVD (Matrix Factorization)
algo_svd = SVD()
cross_validate(algo_svd, data, cv=5)

# Item-based collaborative filtering
algo_knn = KNNBasic(sim_options={"user_based": False})
cross_validate(algo_knn, data, cv=5)

# ----------------------------
# 2. Content-Based Filtering
# ----------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({"title": ["Movie1", "Movie2", "Movie3"], "description": ["Sci-fi action", "Romantic drama", "Fantasy adventure"]})

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
