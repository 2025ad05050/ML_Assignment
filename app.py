"""
app.py
------
Streamlit app to demonstrate 5 classification models trained on the
Telco Customer Churn dataset.

Features:
  a. CSV upload (raw-format test data only)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix + classification report
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Telco Churn — Model Comparison", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}
SCALED_MODELS = {"Logistic Regression", "kNN"}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    with open(f"{MODEL_DIR}/preprocess_meta.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(f"{MODEL_DIR}/{fname}") for name, fname in MODEL_FILES.items()}
    return scaler, meta, models


scaler, meta, models = load_artifacts()
NUMERIC_COLS = meta["numeric_cols"]
CATEGORICAL_COLS = meta["categorical_cols"]
TRAIN_COLUMNS = meta["train_columns"]
TARGET = meta["target"]


def preprocess(df):
    """Apply the same cleaning + one-hot encoding used during training."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    has_target = TARGET in df.columns
    y = None
    if has_target:
        # Handle both 'Yes'/'No' strings and already-encoded 0/1
        if df[TARGET].dtype == object:
            y = df[TARGET].map({"Yes": 1, "No": 0}).values
        else:
            y = df[TARGET].values

    encoded = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    X = pd.concat([df[NUMERIC_COLS].reset_index(drop=True),
                    encoded.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return X, y, has_target


st.title("📞 Telco Customer Churn — Model Comparison App")
st.write(
    "This app demonstrates 5 classification models (Logistic Regression, "
    "Decision Tree, kNN, Naive Bayes, Random Forest) trained on the "
    "**Telco Customer Churn** dataset (19 raw features, 7,043 customers, "
    "binary classification: will the customer churn?)."
)

# ---------------------------------------------------------------------
# a. Dataset upload (test data only, raw format)
# ---------------------------------------------------------------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV in the original Telco Churn column format "
    "(a 'Churn' column is optional but required for metrics)",
    type=["csv"],
)
use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=uploaded_file is None)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif use_sample:
    df_raw = pd.read_csv("test_data.csv")
else:
    df_raw = None

# ---------------------------------------------------------------------
# b. Model selection dropdown
# ---------------------------------------------------------------------
st.sidebar.header("2. Select Model")
selected_model_name = st.sidebar.selectbox("Choose a classification model", list(models.keys()))
model = models[selected_model_name]

if df_raw is not None:
    st.subheader(f"Predictions using: {selected_model_name}")

    X, y_true, has_target = preprocess(df_raw)
    X_input = scaler.transform(X.values) if selected_model_name in SCALED_MODELS else X.values

    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]

    result_df = df_raw.copy()
    result_df["Prediction"] = np.where(y_pred == 1, "Churn", "No Churn")
    result_df["Churn Probability"] = np.round(y_proba, 4)

    st.write("**Sample predictions:**")
    st.dataframe(result_df.head(20), use_container_width=True)

    # -------------------------------------------------------------
    # c. Evaluation metrics (only if ground-truth 'Churn' column given)
    # -------------------------------------------------------------
    if has_target:
        st.subheader("📊 Evaluation Metrics")
        eval_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        cols = st.columns(6)
        for col, (name, val) in zip(cols, eval_metrics.items()):
            col.metric(name, f"{val:.4f}")

        # ---------------------------------------------------------
        # d. Confusion matrix + classification report
        # ---------------------------------------------------------
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=150)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax,
            annot_kws={"size": 9}, cbar=False
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        cm_col, _ = st.columns([1, 2])  # confine the plot to ~1/3 of the wide layout
        with cm_col:
            st.pyplot(fig, use_container_width=False)

        st.subheader("📄 Classification Report")
        report = classification_report(
            y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    else:
        st.info("Upload a CSV with a 'Churn' column to see evaluation metrics and confusion matrix.")

    # -------------------------------------------------------------
    # Bonus: compare ALL models on the same uploaded data
    # -------------------------------------------------------------
    if has_target:
        st.subheader("📈 Compare All Models on This Data")
        rows = []
        for name, mdl in models.items():
            Xi = scaler.transform(X.values) if name in SCALED_MODELS else X.values
            yp = mdl.predict(Xi)
            ypr = mdl.predict_proba(Xi)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_true, yp), 4),
                "AUC": round(roc_auc_score(y_true, ypr), 4),
                "Precision": round(precision_score(y_true, yp), 4),
                "Recall": round(recall_score(y_true, yp), 4),
                "F1": round(f1_score(y_true, yp), 4),
                "MCC": round(matthews_corrcoef(y_true, yp), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("👈 Upload a CSV file or check 'Use bundled test_data.csv' from the sidebar to get started.")

st.markdown("---")
st.caption("ML Assignment 2 — Classification Model Comparison App")
