"""
train_models.py
----------------
Trains 5 classification models on the Telco Customer Churn dataset,
evaluates each with 6 metrics, and saves the trained models + scaler +
preprocessing metadata + a RAW-format test split (for the Streamlit app).

Dataset : Telco Customer Churn (IBM Sample Data Sets)
Source  : Originally published on Kaggle by IBM
          (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Task    : Binary classification - will a customer churn (Yes/No)?
Shape   : 7,043 customers, 19 usable raw features (+ customerID, + target)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------
raw = pd.read_csv(os.path.join(ROOT, "data", "telco_raw.csv"))
print(f"Raw dataset shape: {raw.shape[0]} instances, {raw.shape[1]} columns")

# ---------------------------------------------------------------------
# 2. Basic cleaning
# ---------------------------------------------------------------------
raw = raw.drop(columns=["customerID"])
# TotalCharges has a handful of blank strings for brand-new customers (tenure=0)
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
raw["TotalCharges"] = raw["TotalCharges"].fillna(raw["TotalCharges"].median())

TARGET = "Churn"
raw[TARGET] = raw[TARGET].map({"Yes": 1, "No": 0})

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [c for c in raw.columns if c not in NUMERIC_COLS + [TARGET]]

print(f"Usable raw features: {len(NUMERIC_COLS) + len(CATEGORICAL_COLS)}"
      f" ({len(NUMERIC_COLS)} numeric + {len(CATEGORICAL_COLS)} categorical)")
print(f"Class balance:\n{raw[TARGET].value_counts()}\n")


def encode_features(df, categorical_cols, numeric_cols, train_columns=None):
    """One-hot encode categorical columns; align to a fixed training column set."""
    encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    full = pd.concat([df[numeric_cols].reset_index(drop=True),
                       encoded.reset_index(drop=True)], axis=1)
    if train_columns is not None:
        full = full.reindex(columns=train_columns, fill_value=0)
    return full


# ---------------------------------------------------------------------
# 3. Train / test split on the RAW (pre-encoding) frame
#    -> keeps the held-out test set human-readable for the Streamlit app
# ---------------------------------------------------------------------
train_raw, test_raw = train_test_split(
    raw, test_size=0.2, stratify=raw[TARGET], random_state=RANDOM_STATE
)

# Save RAW-format test split (with target column) -> this is what gets
# uploaded to the Streamlit app, matching the assignment's "test data" ask.
test_raw.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"Saved test_data.csv with {len(test_raw)} rows (raw, human-readable format).")

# ---------------------------------------------------------------------
# 4. Encode features (fit column set on TRAIN only, then align test to it)
# ---------------------------------------------------------------------
X_train_full = encode_features(train_raw, CATEGORICAL_COLS, NUMERIC_COLS)
TRAIN_COLUMNS = list(X_train_full.columns)
X_test_full = encode_features(test_raw, CATEGORICAL_COLS, NUMERIC_COLS, train_columns=TRAIN_COLUMNS)

y_train = train_raw[TARGET].values
y_test = test_raw[TARGET].values

print(f"Encoded feature matrix shape: {X_train_full.shape}")

# ---------------------------------------------------------------------
# 5. Scale features (used by Logistic Regression / kNN)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full.values)
X_test_scaled = scaler.transform(X_test_full.values)

# ---------------------------------------------------------------------
# 6. Define & train models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
}
SCALED_MODELS = {"Logistic Regression", "kNN"}

results = []
for name, model in models.items():
    Xtr = X_train_scaled if name in SCALED_MODELS else X_train_full.values
    Xte = X_test_scaled if name in SCALED_MODELS else X_test_full.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    print(f"\n=== {name} ===")
    print(metrics)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(HERE, f"{fname}.pkl"))

# ---------------------------------------------------------------------
# 7. Save scaler + preprocessing metadata (needed by the Streamlit app)
# ---------------------------------------------------------------------
joblib.dump(scaler, os.path.join(HERE, "scaler.pkl"))
meta = {
    "numeric_cols": NUMERIC_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "train_columns": TRAIN_COLUMNS,
    "target": TARGET,
}
with open(os.path.join(HERE, "preprocess_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ---------------------------------------------------------------------
# 8. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)
print("\n\n=== FINAL COMPARISON TABLE ===")
print(results_df.to_string(index=False))
