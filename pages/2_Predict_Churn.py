from utils.load_css import load_css
load_css()

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils.pdf_generator import generate_prediction_report

from utils.strategy_mapping import (
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

    model = joblib.load(
        "models/final_logistic_model.pkl"
    )

    scaler = joblib.load(
        "models/final_scaler.pkl"
    )

    explainer = joblib.load(
        "models/shap_explainer.pkl"
    )

    encoded_df = pd.read_csv(
        "data/telco_churn_encoded_full.csv"
    )

    feature_names = joblib.load(
        "models/feature_names.pkl"
    )

    return (
        model,
        scaler,
        explainer,
        encoded_df,
        feature_names
    )


(
    model,
    scaler,
    explainer,
    encoded_df,
    feature_names
) = load_artifacts()

# ==================================================
# FEATURE DATAFRAME
# ==================================================
feature_df = encoded_df.drop(
    columns=["Churn"],
    errors="ignore"
)

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
st.title("Customer Churn Prediction")

st.markdown(
    """
    Enter customer details to estimate churn probability
    and receive personalized retention recommendations.
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

        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

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

    # ==================================================
    # AUTO TOTAL CHARGES
    # ==================================================
    total_charges = round(
        tenure * monthly_charges,
        2
    )

    st.info(
        f"Estimated Total Charges: ${total_charges:,.2f}"
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

        # ==================================================
        # BASE CUSTOMER PROFILE
        # ==================================================
        base_row = encoded_df[
            encoded_df["Churn"] == 0
        ].iloc[[0]].copy()

        input_df = base_row.drop(
            columns=["Churn"],
            errors="ignore"
        )

        # ==================================================
        # NUMERIC FEATURES
        # ==================================================
        numeric_updates = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        for col, value in numeric_updates.items():

            if col in input_df.columns:

                input_df.loc[
                    input_df.index[0],
                    col
                ] = value

        # ==================================================
        # CONTRACT FEATURES
        # ==================================================
        contract_columns = [
            "Contract_Month-to-month",
            "Contract_One year",
            "Contract_Two year"
        ]

        for col in contract_columns:

            if col in input_df.columns:

                input_df.loc[
                    input_df.index[0],
                    col
                ] = 0

        selected_contract = (
            f"Contract_{contract}"
        )

        if selected_contract in input_df.columns:

            input_df.loc[
                input_df.index[0],
                selected_contract
            ] = 1

        # ==================================================
        # INTERNET FEATURES
        # ==================================================
        internet_columns = [
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No"
        ]

        for col in internet_columns:

            if col in input_df.columns:

                input_df.loc[
                    input_df.index[0],
                    col
                ] = 0

        selected_internet = (
            f"InternetService_{internet_service}"
        )

        if selected_internet in input_df.columns:

            input_df.loc[
                input_df.index[0],
                selected_internet
            ] = 1

        # ==================================================
        # SUPPORT FEATURES
        # ==================================================
        binary_fields = {

            "TechSupport_Yes":
                1 if tech_support == "Yes" else 0,

            "TechSupport_No":
                1 if tech_support == "No" else 0,

            "OnlineSecurity_Yes":
                1 if online_security == "Yes" else 0,

            "OnlineSecurity_No":
                1 if online_security == "No" else 0,

            "PaperlessBilling_Yes":
                1 if paperless_billing == "Yes" else 0
        }

        for col, value in binary_fields.items():

            if col in input_df.columns:

                input_df.loc[
                    input_df.index[0],
                    col
                ] = value

        # ==================================================
        # PAYMENT METHOD
        # ==================================================
        payment_columns = [
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)"
        ]

        for col in payment_columns:

            if col in input_df.columns:

                input_df.loc[
                    input_df.index[0],
                    col
                ] = 0

        selected_payment = (
            f"PaymentMethod_{payment_method}"
        )

        if selected_payment in input_df.columns:

            input_df.loc[
                input_df.index[0],
                selected_payment
            ] = 1

        # ==================================================
        # ENSURE CORRECT COLUMN ORDER
        # ==================================================
        input_df = input_df[
            feature_names
        ]

        # ==================================================
        # SCALE INPUT
        # ==================================================
        input_scaled = scaler.transform(
            input_df
        )

        input_scaled_df = pd.DataFrame(
            input_scaled,
            columns=feature_names
        )

        # ==================================================
        # MODEL PREDICTION
        # ==================================================
        churn_probability = (
            model.predict_proba(
                input_scaled
            )[0][1]
        )

        prediction = (
            1 if churn_probability >= 0.40 else 0
        )

        risk_tier = assign_risk_tier(
            churn_probability
        )

        # ==================================================
        # SHAP VALUES
        # ==================================================
        shap_values = explainer(
            input_scaled_df
        )

        # ==================================================
        # EXTRACT SHAP VALUES FOR CHURN CLASS
        # ==================================================
        if len(shap_values.values.shape) == 3:

            customer_shap = (
                shap_values.values[0, :, 1]
            )

            customer_base_value = (
                shap_values.base_values[0, 1]
            )

        else:

            customer_shap = (
                shap_values.values[0]
            )

            customer_base_value = (
                shap_values.base_values[0]
            )

        # ==================================================
        # SMART BUSINESS DRIVER LOGIC
        # ==================================================
        feature_impact_map = {}

        for i, feature in enumerate(feature_names):

            shap_score = customer_shap[i]

            # Ignore weak impacts
            if shap_score <= 0.15:
                continue

            # Ignore engineered features
            if feature in [
                "ChargesPerTenure",
                "TotalCharges"
            ]:
                continue

            # ------------------------------------------
            # Month-to-month contract
            # ------------------------------------------
            if (
                feature == "Contract_Month-to-month"
                and input_df.iloc[0][feature] == 1
            ):

                feature_impact_map[
                    "Month-to-Month Contract"
                ] = shap_score

            # ------------------------------------------
            # Electronic payment risk
            # ------------------------------------------
            elif (
                feature == "PaymentMethod_Electronic check"
                and input_df.iloc[0][feature] == 1
            ):

                feature_impact_map[
                    "Payment Friction"
                ] = shap_score

            # ------------------------------------------
            # Technical support risk
            # ------------------------------------------
            elif (
                feature == "TechSupport_No"
                and input_df.iloc[0][feature] == 1
            ):

                current = feature_impact_map.get(
                    "Lack of Support Services",
                    0
                )

                feature_impact_map[
                    "Lack of Support Services"
                ] = current + shap_score

            # ------------------------------------------
            # Online security risk
            # ------------------------------------------
            elif (
                feature == "OnlineSecurity_No"
                and input_df.iloc[0][feature] == 1
            ):

                current = feature_impact_map.get(
                    "Lack of Support Services",
                    0
                )

                feature_impact_map[
                    "Lack of Support Services"
                ] = current + shap_score

            # ------------------------------------------
            # Fiber optic dissatisfaction
            # ------------------------------------------
            elif (
                feature == "InternetService_Fiber optic"
                and input_df.iloc[0][feature] == 1
            ):

                feature_impact_map[
                    "Fiber Optic Service Issues"
                ] = shap_score

            # ------------------------------------------
            # High monthly charges
            # ------------------------------------------
            elif (
                feature == "MonthlyCharges"
                and monthly_charges >= 80
            ):

                feature_impact_map[
                    "High Monthly Charges"
                ] = shap_score

            # ------------------------------------------
            # Low tenure
            # ------------------------------------------
            elif (
                feature == "tenure"
                and tenure <= 12
            ):

                feature_impact_map[
                    "Low Customer Tenure"
                ] = shap_score

        # ==================================================
        # FINAL DRIVER SELECTION
        # ==================================================
        if len(feature_impact_map) > 0:

            primary_driver = max(
                feature_impact_map,
                key=feature_impact_map.get
            )

        else:

            primary_driver = (
                "General Churn Risk"
            )

        # ==================================================
        # RETENTION STRATEGY
        # ==================================================
        retention_strategy = (
            get_reason_based_strategy(
                primary_driver
            )
        )

        # ==================================================
        # DISPLAY RESULTS
        # ==================================================
        st.markdown("---")

        st.subheader(
            "📊 Prediction Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Predicted Outcome",
                (
                    "Likely to Churn"
                    if prediction == 1
                    else "Likely to Stay"
                )
            )

        with col3:

            st.metric(
                "Risk Tier",
                risk_tier
            )

        # ==================================================
        # RISK MESSAGE
        # ==================================================
        if risk_tier == "Critical":

            st.error(
                "This customer is at critical risk of churn and requires immediate intervention."
            )

        elif risk_tier == "High":

            st.warning(
                "This customer is at high risk of churn."
            )

        elif risk_tier == "Medium":

            st.info(
                "This customer has moderate churn risk."
            )

        else:

            st.success(
                "This customer has low churn risk."
            )

        # ==================================================
        # PRIMARY DRIVER
        # ==================================================
        st.markdown(
            "### 📌 Primary Churn Driver"
        )

        st.info(primary_driver)

        # ==================================================
        # RETENTION STRATEGY
        # ==================================================
        st.markdown(
            "### 🎯 Recommended Retention Strategy"
        )

        st.success(
            retention_strategy
        )

        # ==================================================
        # PDF REPORT
        # ==================================================
        prediction_label = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        report_path = (
            "customer_churn_report.pdf"
        )

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

        # ==================================================
        # CREATE SHAP EXPLANATION OBJECT
        # ==================================================
        explanation = shap.Explanation(

            values=customer_shap,

            base_values=customer_base_value,

            data=input_scaled_df.iloc[0],

            feature_names=feature_names
        )


        # ==================================================
        # TOP POSITIVE CHURN FACTORS
        # ==================================================
        st.markdown("---")

        st.subheader(
            "🔍 Top Factors Increasing Churn Risk"
        )

        feature_impacts = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Impact": customer_shap

        })

        positive_impacts = feature_impacts[
            (
                feature_impacts["SHAP Impact"] > 0
            )
            &
            (
                ~feature_impacts["Feature"].isin([
                    "ChargesPerTenure",
                    "TotalCharges"
                ])
            )
        ].sort_values(
            "SHAP Impact",
            ascending=False
        ).head(5)

        positive_impacts["SHAP Impact"] = (
            positive_impacts["SHAP Impact"]
            .round(4)
        )

        st.dataframe(
            positive_impacts,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"Error during prediction: {e}"
        )