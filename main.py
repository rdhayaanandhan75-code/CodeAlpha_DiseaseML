# ==========================================================
# Disease Prediction from Medical Data
# Breast Cancer Prediction using Machine Learning
#
# Algorithms:
# 1. Logistic Regression
# 2. Support Vector Machine (SVM)
# 3. Random Forest
# 4. XGBoost (Optional)
#
# Dataset:
# Breast Cancer Dataset (Scikit-Learn)
# ==========================================================

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------
# Load Dataset
# ---------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("=" * 60)
print("DISEASE PREDICTION USING MACHINE LEARNING")
print("=" * 60)

print("\nDataset Shape :", X.shape)
print("Features      :", len(data.feature_names))
print("Classes       :", data.target_names)

# ---------------------------------------
# Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# Feature Scaling
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------
# Models
# ---------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=5000),

    "Support Vector Machine":
        SVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

# ---------------------------------------
# Add XGBoost (Optional)
# ---------------------------------------

try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

except ImportError:
    print("\nXGBoost is not installed.")
    print("Skipping XGBoost...\n")

# ---------------------------------------
# Training & Evaluation
# ---------------------------------------

results = []

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    if name in ["Logistic Regression", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

# ---------------------------------------
# Final Comparison
# ---------------------------------------

comparison = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(comparison.sort_values(by="Accuracy", ascending=False))
