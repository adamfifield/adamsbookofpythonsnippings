import streamlit as st

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

#from code_executor import code_execution_widget # In-Browser Code Executor disabled

st.title("Machine Learning Model Training")

# Sidebar persistent code execution widget
#code_execution_widget()

st.markdown("## 📌 Machine Learning Model Training & Evaluation")

# Expandable Section: Data Preparation
with st.expander("📂 Preparing Data for Model Training", expanded=False):
    st.markdown("### Splitting Data into Train/Test Sets")
    st.code('''
# Split data into train and test sets
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
''', language="python")

# Expandable Section: Training Regression Models
with st.expander("📈 Training Regression Models", expanded=False):
    st.markdown("### Linear Regression")
    st.code('''
from sklearn.linear_model import LinearRegression

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
''', language="python")

# Expandable Section: Training Classification Models
with st.expander("🔍 Training Classification Models", expanded=False):
    st.markdown("### Logistic Regression")
    st.code('''
from sklearn.linear_model import LogisticRegression

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
''', language="python")

    st.markdown("### Decision Tree Classifier")
    st.code('''
from sklearn.tree import DecisionTreeClassifier

# Train a decision tree classifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy}")
''', language="python")

    st.markdown("### Random Forest Classifier")
    st.code('''
from sklearn.ensemble import RandomForestClassifier

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy}")
''', language="python")

# Expandable Section: Model Evaluation & Cross-Validation
with st.expander("📊 Model Evaluation & Cross-Validation", expanded=False):
    st.markdown("### Cross-Validation")
    st.code('''
from sklearn.model_selection import cross_val_score

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
''', language="python")

    st.markdown("### Confusion Matrix & Classification Report")
    st.code('''
from sklearn.metrics import confusion_matrix, classification_report

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Display classification report
print(classification_report(y_test, y_pred))
''', language="python")

# Expandable Section: Hyperparameter Tuning
with st.expander("⚙️ Hyperparameter Tuning", expanded=False):
    st.markdown("### Grid Search for Hyperparameter Tuning")
    st.code('''
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
''', language="python")

    st.markdown("### Randomized Search for Faster Tuning")
    st.code('''
from sklearn.model_selection import RandomizedSearchCV

# Perform randomized search
random_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=10, cv=5, scoring="accuracy")
random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")
''', language="python")

# Expandable Section: Saving Trained Models
with st.expander("💾 Saving Trained Models", expanded=False):
    st.markdown("### Save Model Using Joblib")
    st.code('''
import joblib

# Save trained model to file
joblib.dump(model, "trained_model.pkl")
''', language="python")

    st.markdown("### Load Model for Predictions")
    st.code('''
# Load trained model from file
model = joblib.load("trained_model.pkl")

# Make predictions with the loaded model
y_pred = model.predict(X_test)
''', language="python")
