🩺 Disease Prediction from Medical Data

Breast Cancer Prediction using Machine Learning

Task 4 of the CodeAlpha Machine Learning Internship — predicting the possibility of disease (breast cancer) from structured medical data using classical and ensemble ML algorithms.

📌 Overview

This project builds and compares multiple classification models to predict whether a tumor is malignant or benign using the well-known Breast Cancer Wisconsin Dataset. The goal is to demonstrate how machine learning can assist in early disease detection by learning patterns from diagnostic measurements such as cell radius, texture, perimeter, and area.

The project trains 4 different algorithms, evaluates each with multiple performance metrics, and ranks them by accuracy — giving a clear, side-by-side comparison of which model performs best for this medical diagnosis task.


🎯 Objective

Predict the possibility of disease (breast cancer) based on patient diagnostic data, using classification techniques and evaluating performance using industry-standard metrics.


🗂 Dataset

DetailDescriptionSourceScikit-learn Breast Cancer Dataset (built-in, no download needed)Samples569 patientsFeatures30 numeric features (radius, texture, perimeter, area, smoothness, etc.)Target ClassesMalignant (0) and Benign (1)TypeStructured / tabular medical data


🧠 Algorithms Used

#AlgorithmScaling Required1Logistic Regression✅ Yes2Support Vector Machine (SVM)✅ Yes3Random Forest Classifier❌ No4XGBoost (optional, auto-skipped if not installed)❌ No


⚙️ Project Pipeline

Load Dataset  →  Train-Test Split (80/20)  →  Feature Scaling
      →  Train Multiple Models  →  Evaluate Metrics
      →  Confusion Matrix + Classification Report
      →  Final Model Comparison Table

Steps in detail:


Data Loading — Fetches the breast cancer dataset directly from sklearn.datasets.
Train-Test Split — 80% training, 20% testing, with stratify=y to preserve class balance.
Feature Scaling — StandardScaler is applied for distance/gradient-sensitive models (Logistic Regression, SVM).
Model Training — Each algorithm is trained on the appropriate (scaled or raw) data.
Evaluation — Every model is scored on Accuracy, Precision, Recall, and F1-Score, plus a full Confusion Matrix and Classification Report.
Comparison — All results are compiled into a single ranked DataFrame, sorted by accuracy.



📊 Evaluation Metrics

Each model is evaluated using:


✅ Accuracy — Overall correctness of predictions
✅ Precision — How many predicted "malignant" cases were actually malignant
✅ Recall — How many actual malignant cases were correctly identified
✅ F1-Score — Harmonic mean of Precision & Recall
✅ Confusion Matrix — Breakdown of TP / TN / FP / FN
✅ Classification Report — Per-class precision, recall, F1, and support



⚠️ In medical diagnosis, Recall is often the most critical metric — missing a malignant case (false negative) is far more costly than a false alarm.




🚀 How to Run

1. Clone the repository

bashgit clone https://github.com/rdhayaanandhan75-code/codealpha_machinelearning.git
cd codealpha_machinelearning/CodeAlpha_DiseasePrediction

2. Install dependencies

bashpip install -r requirements.txt

3. Run the script

bashpython disease_prediction.py


💡 XGBoost is optional. If it isn't installed, the script automatically detects this and skips it without breaking — no extra configuration needed.




📦 Requirements

pandas
scikit-learn
xgboost      # optional


📈 Sample Output Format

============================================================
DISEASE PREDICTION USING MACHINE LEARNING
============================================================

Dataset Shape : (569, 30)
Features      : 30
Classes       : ['malignant' 'benign']

============================================================
Logistic Regression
============================================================
Accuracy : 0.9737
Precision: 0.9722
Recall   : 0.9859
F1 Score : 0.9790
...

======================================================================
FINAL MODEL COMPARISON
======================================================================
               Algorithm  Accuracy  Precision   Recall  F1 Score
     Logistic Regression    0.9737     0.9722   0.9859    0.9790
                     SVM    0.9649     ...
           Random Forest    0.9561     ...
                 XGBoost    0.9474     ...

(Exact numbers will vary slightly by run/random seed.)


🏆 Key Takeaways


Simple linear models like Logistic Regression can perform remarkably well on well-separated, well-engineered medical datasets.
Feature scaling matters — models like SVM and Logistic Regression are sensitive to feature magnitude, while tree-based models (Random Forest, XGBoost) are not.
A multi-metric comparison gives a much more trustworthy picture of model quality than accuracy alone — especially in healthcare, where false negatives carry real risk.



🛠 Tech Stack


Language: Python 3.9+
Libraries: pandas, scikit-learn, xgboost (optional)
Environment: Jupyter Notebook / Python script



👤 Author

Developed as part of the CodeAlpha Machine Learning Internship — Task 4: Disease Prediction from Medical Data.

📌 This project is submitted under the CodeAlpha internship program and follows the required naming convention: CodeAlpha_DiseasePrediction.


📎 Related


🔗 CodeAlpha Website
🔗 Other tasks in this repo: CodeAlpha_CreditScoringModel
