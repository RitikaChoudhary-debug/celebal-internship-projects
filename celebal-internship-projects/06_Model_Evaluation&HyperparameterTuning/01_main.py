# src/main.py

"""
Model Evaluation and Hyperparameter Tuning Project
Description: This script trains multiple machine learning models,
evaluates them using performance metrics, and applies hyperparameter tuning
to identify the best-performing model.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

# Suppress warning messages for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------
# 1. Load and Prepare Dataset
# ---------------------------
data = load_breast_cancer()  # Binary classification dataset
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 2. Train Initial ML Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC()
}

# Define evaluation function to compute key metrics
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

# Evaluate base models without tuning
print(" Initial Model Evaluation:")
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(evaluate_model(name, model, X_test, y_test))

# --------------------------------------------
# 3. Hyperparameter Tuning for Selected Models
# --------------------------------------------

# GridSearchCV for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='f1'
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# RandomizedSearchCV for Support Vector Machine
param_dist_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
random_svm = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist_svm,
    n_iter=6,
    cv=5,
    scoring='f1',
    random_state=42
)
random_svm.fit(X_train, y_train)
best_svm = random_svm.best_estimator_

# ---------------------------------------------
# 4. Evaluation After Hyperparameter Tuning
# ---------------------------------------------
print("\n After Hyperparameter Tuning:")
results.append(evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test))
results.append(evaluate_model("SVM (Tuned)", best_svm, X_test, y_test))

# ---------------------------------------------
# 5. Identify and Display the Best Model
# ---------------------------------------------
df_results = pd.DataFrame(results)
best_model = df_results.sort_values(by="F1 Score", ascending=False).iloc[0]

print("\n Best Model Summary:")
print(best_model.to_string(index=True))
