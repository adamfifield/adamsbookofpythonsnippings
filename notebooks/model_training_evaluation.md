# 📖 Model Training & Evaluation

### **Description**  
This section covers **splitting data into training and test sets**, **choosing appropriate machine learning models**, **training models**, **evaluating performance using metrics**, **handling imbalanced datasets**, **analyzing feature importance**, **checking for overfitting/underfitting**, and **hyperparameter tuning**.

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Data Splitting**  
  - Use `train_test_split()` from `sklearn.model_selection` to divide data into training and testing sets.  
  - Consider stratified sampling (`stratify=y`) for classification problems.  
  - Use cross-validation (`KFold`, `StratifiedKFold`) to ensure robust evaluation.  

- ✅ **Choosing the Right Model**  
  - For regression, use `LinearRegression()`, `RandomForestRegressor()`, `XGBoost()`, etc.  
  - For classification, use `LogisticRegression()`, `RandomForestClassifier()`, `XGBoostClassifier()`, etc.  
  - Consider baseline models (`DummyClassifier`, `DummyRegressor`) for comparison.  

- ✅ **Model Training**  
  - Use `model.fit(X_train, y_train)` to train models.  
  - Tune hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.  
  - Save trained models using `joblib.dump()` or `pickle`.  

- ✅ **Performance Metrics**  
  - For classification:  
    - Use `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`, `roc_auc_score()`.  
    - Use `classification_report()` for summary.  
  - For regression:  
    - Use `mean_squared_error()`, `mean_absolute_error()`, `r2_score()`.  

- ✅ **Handling Imbalanced Data**  
  - Use `SMOTE()` from `imblearn.over_sampling` to generate synthetic samples.  
  - Try different class weighting strategies (`class_weight='balanced'` in classifiers).  
  - Use precision-recall curve instead of accuracy for imbalanced problems.  

- ✅ **Feature Importance & Model Explainability**  
  - Use `model.feature_importances_` for tree-based models.  
  - Use `permutation_importance()` from `sklearn.inspection` for feature impact.  
  - Use `SHAP` and `LIME` for model explainability.  

- ✅ **Evaluating Overfitting & Underfitting**  
  - Check the difference between train/test scores.  
  - Use learning curves to visualize performance over increasing training data.  
  - Use dropout & regularization (for neural networks).  

- ✅ **Hyperparameter Tuning**  
  - Use `GridSearchCV` for exhaustive parameter search.  
  - Use `RandomizedSearchCV` for faster tuning.  
  - Use Bayesian optimization (`optuna`) for more efficient tuning.  
