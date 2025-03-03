"""
Model Training & Evaluation in Python

This script covers various essential model training and evaluation techniques, including:
- Splitting data into training and test sets
- Choosing the right model
- Training machine learning models
- Evaluating model performance using metrics
- Handling imbalanced datasets
- Understanding feature importance
- Identifying overfitting & underfitting
- Hyperparameter tuning
"""

import pandas as pd
import numpy as np

# ----------------------------
# 1. Data Splitting
# ----------------------------

# Split data into training and test sets
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use cross-validation for better evaluation
from sklearn.model_selection import KFold, StratifiedKFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# 2. Choosing the Right Model
# ----------------------------

# Select appropriate model for the task
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Classification model
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Regression model
reg = RandomForestRegressor(n_estimators=100, random_state=42)

# ----------------------------
# 3. Model Training
# ----------------------------

# Train classification model
clf.fit(X_train, y_train)

# Train regression model
reg.fit(X_train, y_train)

# ----------------------------
# 4. Model Evaluation (Classification Metrics)
# ----------------------------

# Import classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Predict on test set
y_pred = clf.predict(X_test)

# Compute classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Print classification report
print(classification_report(y_test, y_pred))

# ----------------------------
# 5. Model Evaluation (Regression Metrics)
# ----------------------------

# Import regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict on test set
y_pred_reg = reg.predict(X_test)

# Compute regression metrics
mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

# ----------------------------
# 6. Handling Imbalanced Data
# ----------------------------

# Use SMOTE to generate synthetic samples
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model on resampled data
clf.fit(X_resampled, y_resampled)

# ----------------------------
# 7. Feature Importance & Model Explainability
# ----------------------------

# Retrieve feature importance from a tree-based model
feature_importances = clf.feature_importances_

# Compute permutation importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(clf, X_test, y_test, random_state=42)

# ----------------------------
# 8. Evaluating Overfitting & Underfitting
# ----------------------------

# Check training vs test accuracy
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

# ----------------------------
# 9. Hyperparameter Tuning
# ----------------------------

# GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_

# ----------------------------
# END OF SCRIPT
# ----------------------------
