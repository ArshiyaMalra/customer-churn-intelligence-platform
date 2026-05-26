from utils.load_css import load_css

load_css()

import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

st.set_page_config(
    page_title="SHAP Analysis",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# LOAD ARTIFACTS
# =====================================================
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

    feature_names = joblib.load(
        "models/feature_names.pkl"
    )

    return (
        model,
        scaler,
        explainer,
        feature_names
    )


(
    model,
    scaler,
    explainer,
    feature_names
) = load_artifacts()

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_sample_data():

    df = pd.read_csv(
        "data/telco_churn_encoded_full.csv"
    )

    if "Churn" in df.columns:

        df = df.drop(
            columns=["Churn"]
        )

    return df


X = load_sample_data()

# =====================================================
# CLEAN FEATURE NAMES
# =====================================================
feature_name_mapping = {

    "tenure": "Customer Tenure",

    "MonthlyCharges": "Monthly Charges",

    "TotalCharges": "Total Charges",

    "ChargesPerTenure": "Charges Per Tenure",

    "Contract_Month-to-month":
        "Month-to-Month Contract",

    "Contract_One year":
        "One-Year Contract",

    "Contract_Two year":
        "Two-Year Contract",

    "OnlineSecurity_No":
        "No Online Security",

    "TechSupport_No":
        "No Tech Support",

    "InternetService_Fiber optic":
        "Fiber Optic Internet",

    "PaymentMethod_Electronic check":
        "Electronic Check Payment",

    "PaperlessBilling_Yes":
        "Paperless Billing"
}

# =====================================================
# PAGE HEADER
# =====================================================
st.title("SHAP Explainability Dashboard")

st.markdown("""
This dashboard presents global feature importance using
SHAP (SHapley Additive exPlanations).

The analysis helps identify the key business factors
influencing customer churn predictions.
""")

# =====================================================
# SAMPLE DATA
# =====================================================
sample_size = min(300, len(X))

X_sample = X.sample(
    sample_size,
    random_state=42
)

# Ensure proper feature order
X_sample = X_sample[
    feature_names
]

# =====================================================
# SCALE DATA BEFORE SHAP
# =====================================================
X_scaled = scaler.transform(
    X_sample
)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names
)

# =====================================================
# GENERATE SHAP VALUES
# =====================================================
shap_values = explainer(
    X_scaled_df
)

# =====================================================
# GLOBAL FEATURE IMPORTANCE
# =====================================================
st.subheader(
    "📊 Global Feature Importance"
)

st.markdown("""
The chart below shows the most influential features
affecting churn predictions across customers.
""")

# Cleaner figure size
plt.figure(figsize=(5, 3))

# SHAP Bar Plot
shap.plots.bar(
    shap_values,
    max_display=8,
    show=False
)

# Improve readability
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()

fig = plt.gcf()

left, center, right = st.columns([2.5, 3, 2.5])

with center:
    st.pyplot(
    fig,
    clear_figure=True,
    use_container_width=False
)

plt.close(fig)

# =====================================================
# TOP FEATURES TABLE
# =====================================================
st.markdown("---")

st.subheader(
    "Top 10 Most Important Features"
)

# Handle SHAP dimensions safely
if len(shap_values.values.shape) == 3:

    mean_abs_shap = np.abs(
        shap_values.values[:, :, 1]
    ).mean(axis=0)

else:

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": [
        feature_name_mapping.get(
            feature,
            feature
        )
        for feature in feature_names
    ],

    "Mean |SHAP Value|": mean_abs_shap

}).sort_values(

    "Mean |SHAP Value|",
    ascending=False

).head(10)

importance_df[
    "Mean |SHAP Value|"
] = importance_df[
    "Mean |SHAP Value|"
].round(4)

st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# INTERPRETATION
# =====================================================
st.markdown("---")

st.subheader(
    "Interpretation"
)

st.markdown("""
SHAP analysis helps explain how different customer attributes
influence churn predictions generated by the machine learning model.

### Key Observations

- Customer tenure and pricing-related variables show strong influence on churn behavior.
- Contract type significantly affects customer retention patterns.
- Service-related factors such as technical support and online security also contribute to churn risk.
- Certain payment methods and internet service types influence customer retention outcomes.

Features with larger mean SHAP values contribute more strongly
to the model’s prediction decisions.

This analysis improves transparency by helping identify the
most influential business drivers behind customer churn.
""")

