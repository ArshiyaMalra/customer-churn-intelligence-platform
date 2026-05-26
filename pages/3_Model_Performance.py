from utils.load_css import load_css

load_css()

import streamlit as st
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD METRICS
# =====================================================
@st.cache_resource
def load_metrics():

    return joblib.load(
        "models/model_performance.pkl"
    )

metrics = load_metrics()

# =====================================================
# EXTRACT METRICS
# =====================================================
accuracy = float(metrics["Accuracy"])

precision = float(metrics["Precision"])

recall = float(metrics["Recall"])

f1_score = float(metrics["F1-Score"])

roc_auc = float(metrics["ROC-AUC"])

# =====================================================
# PAGE HEADER
# =====================================================
st.title("Model Performance Dashboard")

st.markdown("""
This dashboard summarizes the evaluation performance of the
final Logistic Regression model developed for customer churn prediction.
""")

# =====================================================
# KPI METRICS
# =====================================================
st.markdown("---")

st.subheader("📈 Evaluation Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )

with col3:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{f1_score:.4f}"
    )

with col5:

    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )

# =====================================================
# BUSINESS INTERPRETATION
# =====================================================
st.markdown("---")

st.subheader("📈 Business Interpretation")

st.markdown("""
### Key Insights

- The model demonstrates reliable overall predictive performance for customer churn analysis.
- The ROC-AUC score indicates good discrimination capability between churn and non-churn customers.
- The model can support proactive retention campaigns and targeted customer engagement strategies.

### Business Relevance

The churn prediction system can help organizations:
- identify customers with elevated churn risk,
- support targeted retention strategies,
- improve customer engagement efforts,
- and assist decision-making using explainable AI techniques.
""")

# =====================================================
# WHY LOGISTIC REGRESSION
# =====================================================
st.markdown("---")

st.subheader("Why Logistic Regression?")

st.markdown("""
Logistic Regression was selected as the final model because it:

- Provides strong predictive performance with high interpretability.
- Produces reliable churn probability estimates.
- Integrates effectively with SHAP explainability.
- Is computationally efficient and lightweight.
- Is easier to deploy and maintain in production systems.

The model achieved an effective balance between
performance, interpretability, and deployment simplicity.
""")

# =====================================================
# FINAL CONCLUSION
# =====================================================
st.markdown("---")

st.subheader("✅ Conclusion")

st.success(f"""
The final Logistic Regression model achieved strong predictive performance
with an ROC-AUC score of {roc_auc:.2%}.

The model is suitable for identifying customers at high churn risk
and supporting data-driven customer retention strategies.
""")