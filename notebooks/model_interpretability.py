"""
Model Interpretability & Explainability

This script covers techniques for:
- Feature importance methods (Permutation, Tree-based, SHAP)
- SHAP (SHapley Additive Explanations) for model interpretability
- LIME (Local Interpretable Model-Agnostic Explanations)
- Counterfactual explanations using DiCE
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import dice_ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Dataset & Train Model
# ----------------------------

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 2. Feature Importance (Tree-Based & Permutation)
# ----------------------------

from eli5.sklearn import PermutationImportance
import eli5

# Tree-based feature importance
feature_importance = model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]

print("Feature Importance (Tree-based):")
for i in sorted_indices[:10]:
    print(f"{X.columns[i]}: {feature_importance[i]:.4f}")

# Permutation importance
perm_importance = PermutationImportance(model, scoring="accuracy").fit(X_test, y_test)
print("Permutation Importance:")
eli5.show_weights(perm_importance, feature_names=X.columns.tolist())

# ----------------------------
# 3. SHAP Explanations
# ----------------------------

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

# SHAP Force Plot (First Instance)
shap.force_plot(explainer.expected_value[1], shap_values[1].values, X_test.iloc[1], matplotlib=True)

# ----------------------------
# 4. LIME Explanations
# ----------------------------

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns.tolist(), class_names=["Malignant", "Benign"], discretize_continuous=True)

# Explain one instance
exp = explainer.explain_instance(X_test.iloc[1].values, model.predict_proba, num_features=5)
exp.show_in_notebook()

# ----------------------------
# 5. Counterfactual Explanations (DiCE)
# ----------------------------

dice_data = dice_ml.Data(dataframe=pd.concat([X_train, pd.Series(y_train, name="target")], axis=1), continuous_features=list(X.columns), outcome_name="target")

# Model Wrapper
m = dice_ml.Model(model=model, backend="sklearn")

exp_gen = dice_ml.Dice(dice_data, m)
query_instance = X_test.iloc[1:2]  # Pick one sample for counterfactual generation
dice_exp = exp_gen.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")

# Display counterfactual examples
dice_exp.visualize_as_dataframe()
