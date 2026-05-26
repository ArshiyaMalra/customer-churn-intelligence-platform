from utils.load_css import load_css
load_css()

import streamlit as st
import pandas as pd
import joblib

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

    scaler = joblib.load("models/final_scaler.pkl")

    feature_names = joblib.load("models/feature_names.pkl")

    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

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

    if row.get("Contract_Month-to-month", 0) == 1:
        return "Month-to-Month Contract"

    elif row.get("MonthlyCharges", 0) > 80:
        return "High Monthly Charges"

    elif row.get("tenure", 0) < 12:
        return "Low Customer Tenure"

    elif row.get("InternetService_Fiber optic", 0) == 1:
        return "Fiber Optic Service Issues"

    elif row.get("TechSupport_No", 0) == 1:
        return "No Technical Support"

    elif row.get("OnlineSecurity_No", 0) == 1:
        return "No Online Security"

    elif row.get("PaymentMethod_Electronic check", 0) == 1:
        return "Electronic Check Payment"

    else:
        return "General Churn Risk"


# =========================================================
# PAGE HEADER
# =========================================================
st.title("Batch Churn Prediction")

st.markdown("""
Upload a customer dataset to generate bulk churn predictions,
risk prioritization, and retention recommendations at scale.
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

        st.info("Sample file not found.")


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

        # -------------------------------------------------
        # Validate Columns
        # -------------------------------------------------
        missing_columns = [

            col for col in feature_names
            if col not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded CSV is missing required encoded columns."
            )

            st.write("Missing columns:")
            st.write(missing_columns[:10])

            st.stop()

        # -------------------------------------------------
        # Correct Column Order
        # -------------------------------------------------
        X = batch_df[feature_names]

        # -------------------------------------------------
        # SCALE INPUTS
        # -------------------------------------------------
        X_scaled = scaler.transform(X)

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------
        probabilities = model.predict_proba(X_scaled)[:, 1]

        predictions = model.predict(X_scaled)

        # -------------------------------------------------
        # Build Results
        # -------------------------------------------------
        results = batch_df.copy()

        results["Churn Probability"] = probabilities

        results["Predicted Outcome"] = [

            "Likely to Churn"
            if pred == 1
            else "Likely to Stay"

            for pred in predictions
        ]

        results["Risk Tier"] = [

            assign_risk_tier(prob)

            for prob in probabilities
        ]

        # -------------------------------------------------
        # Primary Driver + Strategy
        # -------------------------------------------------
        primary_drivers = []

        strategies = []

        for _, row in results.iterrows():

            driver = get_primary_driver(row)

            strategy = get_reason_based_strategy(driver)

            primary_drivers.append(driver)

            strategies.append(strategy)

        results["Primary Churn Driver"] = primary_drivers

        results["Retention Strategy"] = strategies

        # -------------------------------------------------
        # Summary Metrics
        # -------------------------------------------------
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
                int(
                    (results["Predicted Outcome"]
                     == "Likely to Churn").sum()
                )
            )

        with col3:

            st.metric(
                "Average Churn Probability",
                f"{results['Churn Probability'].mean() * 100:.2f}%"
            )
        # -------------------------------------------------
        # Risk Tier Distribution
        # -------------------------------------------------
        st.markdown("---")

        st.subheader("📈 Risk Tier Distribution")

        risk_counts = (
            results["Risk Tier"]
            .value_counts()
        )

        st.bar_chart(risk_counts)

        # -------------------------------------------------
        # Results Table
        # -------------------------------------------------
        st.subheader("📋 Prediction Results")

        display_columns = [

            "Churn Probability",
            "Predicted Outcome",
            "Risk Tier",
            "Primary Churn Driver",
            "Retention Strategy"
        ]

        available_columns = [

            col for col in display_columns
            if col in results.columns
        ]

        st.dataframe(
            results[available_columns].head(20),
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # Download CSV
        # -------------------------------------------------
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv_data,
            file_name="batch_churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Error processing file: {e}")