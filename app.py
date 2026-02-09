import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=" ",
    layout="centered"
)

st.title(" Heart Disease Prediction System")
st.caption("Academic ML Project - Heart Disease Prediction")
st.write("Machine Learning Based Medical Decision Support")

# ---------------- LOAD MODELS ----------------
trained_models = {
    "Logistic Regression": joblib.load("lr_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "Naive Bayes": joblib.load("nb_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl")
}

# Load scaler
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(trained_models.keys())
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Heart Dataset (CSV - Test Data Only)",
    type=["csv"]
)

# IMPORTANT: define y globally to avoid NameError
y = None

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader(" Uploaded Data Preview")
    st.dataframe(data.head())

    # Separate features and target (if present)
    if "target" in data.columns:
        X = data.drop("target", axis=1)
        y = data["target"]
    else:
        X = data

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    model = trained_models[model_name]
    y_pred = model.predict(X_scaled)

    data["Prediction"] = y_pred

    st.subheader(" Prediction Results")
    st.dataframe(data)

    # ---------------- EVALUATION ----------------
    if y is not None:

        st.subheader(" Model Performance Summary")

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        # AUC needs probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = None

        metrics_df = pd.DataFrame({
            "ML Model Name": [model_name],
            "Accuracy": [accuracy],
            "AUC": [auc],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
            "MCC": [mcc]
        })

        st.dataframe(metrics_df)

        # -------- Classification Report --------
        st.subheader(" Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # -------- Confusion Matrix --------
        st.subheader(" Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        st.success(f"Accuracy on Uploaded Data: {accuracy:.2f}")

    # ---------------- SUMMARY ----------------
    st.subheader(" Prediction Summary")
    st.write("Patients with Heart Disease:", (y_pred == 1).sum())
    st.write("Healthy Patients:", (y_pred == 0).sum())

else:
    st.info("Please upload a CSV file to start prediction.")
