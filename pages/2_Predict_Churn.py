from utils.load_css import load_css

load_css()

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from utils.pdf_generator import generate_prediction_report

from utils.strategy_mapping import (
    categorize_churn_reason,
    get_reason_based_strategy
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Predict Churn",
    page_icon="🔮",
    layout="wide"
)

# ==================================================
# LOAD ARTIFACTS
# ==================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_logistic_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    encoded_df = pd.read_csv("data/telco_churn_encoded_full.csv")
    return model, explainer, encoded_df


model, explainer, encoded_df = load_artifacts()

# Feature-only dataframe
feature_df = encoded_df.drop(columns=["Churn"], errors="ignore")

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def assign_risk_tier(prob):
    if prob >= 0.80:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"


# ==================================================
# PAGE HEADER
# ==================================================
st.title("🔮 Customer Churn Prediction")
st.markdown(
    """
    Enter customer details to estimate churn probability and receive
    personalized retention recommendations.
    """
)

# ==================================================
# INPUT FORM
# ==================================================
with st.form("prediction_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=10.0
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    submit = st.form_submit_button(
        "Predict Churn",
        type="primary"
    )

# ==================================================
# PREDICTION LOGIC
# ==================================================
if submit:
    try:
        # --------------------------------------------------
        # Start with a real non-churn customer profile
        # --------------------------------------------------
        base_row = encoded_df[encoded_df["Churn"] == 0].iloc[[0]].copy()
        input_df = base_row.drop(columns=["Churn"], errors="ignore")

        # --------------------------------------------------
        # Update numeric features
        # --------------------------------------------------
        if "tenure" in input_df.columns:
            input_df.loc[input_df.index[0], "tenure"] = tenure

        if "MonthlyCharges" in input_df.columns:
            input_df.loc[input_df.index[0], "MonthlyCharges"] = monthly_charges

        if "TotalCharges" in input_df.columns:
            input_df.loc[input_df.index[0], "TotalCharges"] = total_charges

        # --------------------------------------------------
        # Update Contract columns
        # --------------------------------------------------
        contract_columns = [
            "Contract_Month-to-month",
            "Contract_One year",
            "Contract_Two year"
        ]

        for col in contract_columns:
            if col in input_df.columns:
                input_df.loc[input_df.index[0], col] = 0

        selected_contract_col = f"Contract_{contract}"
        if selected_contract_col in input_df.columns:
            input_df.loc[input_df.index[0], selected_contract_col] = 1

        # --------------------------------------------------
        # Update Internet Service columns
        # --------------------------------------------------
        internet_columns = [
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No"
        ]

        for col in internet_columns:
            if col in input_df.columns:
                input_df.loc[input_df.index[0], col] = 0

        selected_internet_col = f"InternetService_{internet_service}"
        if selected_internet_col in input_df.columns:
            input_df.loc[input_df.index[0], selected_internet_col] = 1

        # --------------------------------------------------
        # Update Tech Support columns
        # --------------------------------------------------
        if "TechSupport_Yes" in input_df.columns:
            input_df.loc[input_df.index[0], "TechSupport_Yes"] = (
                1 if tech_support == "Yes" else 0
            )

        if "TechSupport_No" in input_df.columns:
            input_df.loc[input_df.index[0], "TechSupport_No"] = (
                1 if tech_support == "No" else 0
            )

        # --------------------------------------------------
        # Update Online Security columns
        # --------------------------------------------------
        if "OnlineSecurity_Yes" in input_df.columns:
            input_df.loc[input_df.index[0], "OnlineSecurity_Yes"] = (
                1 if online_security == "Yes" else 0
            )

        if "OnlineSecurity_No" in input_df.columns:
            input_df.loc[input_df.index[0], "OnlineSecurity_No"] = (
                1 if online_security == "No" else 0
            )

        # --------------------------------------------------
        # Update Payment Method columns
        # --------------------------------------------------
        payment_columns = [
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)"
        ]

        for col in payment_columns:
            if col in input_df.columns:
                input_df.loc[input_df.index[0], col] = 0

        selected_payment_col = f"PaymentMethod_{payment_method}"
        if selected_payment_col in input_df.columns:
            input_df.loc[input_df.index[0], selected_payment_col] = 1

        # --------------------------------------------------
        # Update Paperless Billing columns
        # --------------------------------------------------
        if "PaperlessBilling_Yes" in input_df.columns:
            input_df.loc[input_df.index[0], "PaperlessBilling_Yes"] = (
                1 if paperless_billing == "Yes" else 0
            )

        if "PaperlessBilling_No" in input_df.columns:
            input_df.loc[input_df.index[0], "PaperlessBilling_No"] = (
                1 if paperless_billing == "No" else 0
            )

        # Ensure column order matches training data
        input_df = input_df[feature_df.columns]

        # --------------------------------------------------
        # Model Prediction
        # --------------------------------------------------
        churn_probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        risk_tier = assign_risk_tier(churn_probability)

        # --------------------------------------------------
        # SHAP Values
        # --------------------------------------------------
        shap_values = explainer(input_df)

        if len(shap_values.values.shape) == 3:
            customer_shap = shap_values.values[0, :, 1]
        else:
            customer_shap = shap_values.values[0]

        # --------------------------------------------------
        # Focus on business-relevant features
        # --------------------------------------------------
        priority_features = [
            "Contract_Month-to-month",
            "PaymentMethod_Electronic check",
            "TechSupport_No",
            "OnlineSecurity_No",
            "InternetService_Fiber optic",
            "MonthlyCharges",
            "tenure",
            "PaperlessBilling_Yes"
        ]

        candidates = []

        for i, feature in enumerate(input_df.columns):
            if feature in priority_features:
                candidates.append((feature, abs(customer_shap[i])))

        if candidates:
            top_feature_raw = max(candidates, key=lambda x: x[1])[0]
        else:
            top_index = abs(customer_shap).argmax()
            top_feature_raw = input_df.columns[top_index]

        # --------------------------------------------------
        # Centralized logic
        # --------------------------------------------------
        primary_driver = categorize_churn_reason(top_feature_raw)
        retention_strategy = get_reason_based_strategy(primary_driver)

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Predicted Outcome",
                "Likely to Churn" if prediction == 1 else "Likely to Stay"
            )

        with col3:
            st.metric(
                "Risk Tier",
                risk_tier
            )

        # Risk message
        if risk_tier == "Critical":
            st.error(
                "This customer is at critical risk of churn and requires immediate intervention."
            )
        elif risk_tier == "High":
            st.warning("This customer is at high risk of churn.")
        elif risk_tier == "Medium":
            st.info("This customer has moderate churn risk.")
        else:
            st.success("This customer has low churn risk.")

        # Primary Driver
        st.markdown("### 📌 Primary Churn Driver")
        st.info(primary_driver)

        # Recommended Strategy
        st.markdown("### 🎯 Recommended Retention Strategy")
        st.success(retention_strategy)

        # ----------------------------------------------
        # Generate PDF Report
        # ----------------------------------------------
        prediction_label = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        report_path = "customer_churn_report.pdf"

        generate_prediction_report(
            file_path=report_path,
            churn_probability=churn_probability,
            prediction_label=prediction_label,
            risk_tier=risk_tier,
            primary_driver=primary_driver,
            retention_strategy=retention_strategy
        )

        with open(report_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=pdf_file,
                file_name="customer_churn_report.pdf",
                mime="application/pdf"
            )


        # --------------------------------------------------
        # SHAP Explanation Chart
        # --------------------------------------------------
        st.markdown("### 📈 SHAP Explanation")

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(
            shap_values[0],
            max_display=10,
            show=False
        )
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")