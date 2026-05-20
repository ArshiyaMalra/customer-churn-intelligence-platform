from utils.load_css import load_css
load_css()

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

from utils.strategy_mapping import (
    categorize_churn_reason,
    get_reason_based_strategy
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_logistic_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names


model, feature_names = load_artifacts()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def assign_risk_tier(prob):
    if prob >= 0.80:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"


def get_primary_driver(row):
    """
    Lightweight rule-based primary driver selection for batch mode.
    This keeps the page fast and avoids computing SHAP for every row.
    """
    if row.get("Contract_Month-to-month", 0) == 1:
        return "Month-to-Month Contract"

    if row.get("PaymentMethod_Electronic check", 0) == 1:
        return "Electronic Check Payment Method"

    if row.get("TechSupport_No", 0) == 1:
        return "No Technical Support"

    if row.get("OnlineSecurity_No", 0) == 1:
        return "No Online Security Service"

    if row.get("InternetService_Fiber optic", 0) == 1:
        return "Fiber Optic Internet Service"

    if row.get("MonthlyCharges", 0) > 80:
        return "High Monthly Charges"

    if row.get("tenure", 0) < 12:
        return "Low Customer Tenure"

    return "General Churn Risk"


# =========================================================
# PAGE HEADER
# =========================================================
st.title("📂 Batch Churn Prediction")
st.markdown("""
Upload a CSV file containing multiple customer records and generate
churn predictions, risk tiers, and retention recommendations in bulk.
""")

# =========================================================
# TEMPLATE DOWNLOAD
# =========================================================
with st.expander("📥 Download Sample Input File"):
    try:
        with open("data/sample_batch_input.csv", "rb") as f:
            st.download_button(
                label="Download Sample CSV",
                data=f,
                file_name="sample_batch_input.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.info("Sample file will be created in the next step.")

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =========================================================
# PROCESS FILE
# =========================================================
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(batch_df.head())

        # Validate columns
        missing_columns = [
            col for col in feature_names
            if col not in batch_df.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded file is missing required encoded columns."
            )
            st.write("First few missing columns:")
            st.write(missing_columns[:10])
            st.stop()

        # Ensure correct column order
        X = batch_df[feature_names]

        # Predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)

        # Append outputs
        results = batch_df.copy()
        results["Churn Probability"] = probabilities
        results["Predicted Outcome"] = [
            "Likely to Churn" if p == 1 else "Likely to Stay"
            for p in predictions
        ]
        results["Risk Tier"] = [
            assign_risk_tier(p)
            for p in probabilities
        ]

        # Primary Driver and Strategy
        primary_drivers = []
        strategies = []

        for _, row in results.iterrows():
            driver = get_primary_driver(row)
            primary_drivers.append(driver)

            category = categorize_churn_reason(driver)
            strategy = get_reason_based_strategy(category)
            strategies.append(strategy)

        results["Primary Churn Driver"] = primary_drivers
        results["Retention Strategy"] = strategies

        # Display summary
        st.markdown("---")
        st.subheader("📊 Batch Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Customers",
                len(results)
            )

        with col2:
            st.metric(
                "Likely to Churn",
                int((results["Predicted Outcome"] ==
                     "Likely to Churn").sum())
            )

        with col3:
            st.metric(
                "Average Churn Probability",
                f"{results['Churn Probability'].mean() * 100:.2f}%"
            )

        # Show results
        st.subheader("📋 Prediction Results")
        st.dataframe(results.head(20))

        # Download results
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv_data,
            file_name="batch_churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")