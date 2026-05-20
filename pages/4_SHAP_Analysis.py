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

# -----------------------------------------------------
# Load artifacts
# -----------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_logistic_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, explainer, feature_names

model, explainer, feature_names = load_artifacts()

# -----------------------------------------------------
# Load sample data
# -----------------------------------------------------
@st.cache_data
def load_sample_data():
    df = pd.read_csv("data/telco_churn_encoded_full.csv")
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df

X = load_sample_data()

st.title("🔍 SHAP Explainability Dashboard")
st.markdown("""
This dashboard presents global feature importance using
SHAP (SHapley Additive exPlanations).
""")

# -----------------------------------------------------
# Compute SHAP values on a sample
# -----------------------------------------------------
sample_size = min(300, len(X))
X_sample = X.sample(sample_size, random_state=42)

# Ensure correct column order
X_sample = X_sample[feature_names]

# Generate SHAP values
shap_values = explainer(X_sample)

# -----------------------------------------------------
# Global SHAP Bar Plot
# -----------------------------------------------------
st.subheader("📊 Global Feature Importance")

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
st.pyplot(fig, clear_figure=True)
plt.close(fig)

# -----------------------------------------------------
# Top Features Table
# -----------------------------------------------------
st.markdown("---")
st.subheader("📋 Top 10 Most Important Features")

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP Value|": mean_abs_shap
}).sort_values("Mean |SHAP Value|",
               ascending=False).head(10)

st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)

# -----------------------------------------------------
# Interpretation
# -----------------------------------------------------
st.markdown("---")
st.subheader("📈 Interpretation")

st.markdown("""
SHAP quantifies how each feature contributes to model predictions.

### Key Insights
- Month-to-month contracts increase churn risk.
- Short customer tenure is a strong churn indicator.
- High monthly charges elevate churn probability.
- Lack of online security and technical support contributes to churn.
- Electronic check payment method is associated with higher churn.

Higher mean absolute SHAP values indicate greater overall influence.
""")

# -----------------------------------------------------
# Why SHAP?
# -----------------------------------------------------
st.markdown("---")
st.subheader("🧠 Why SHAP?")

st.markdown("""
SHAP provides consistent, model-agnostic explanations based on game
theory. It enables both global feature importance analysis and
individual customer-level explanations.
""")