from utils.load_css import load_css
load_css()

import streamlit as st

# =====================================================

# PAGE CONFIG

# =====================================================

st.set_page_config(
page_title="Model Performance",
page_icon="📊",
layout="wide"
)

# =====================================================

# MANUAL METRICS

# =====================================================

accuracy = 0.7970
precision = 0.5017
recall = 0.7807
f1_score = 0.6109
roc_auc = 0.8374

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

col1.metric("Accuracy", f"{accuracy:.4f}")

col2.metric("Precision", f"{precision:.4f}")

col3.metric("Recall", f"{recall:.4f}")

col4.metric("F1 Score", f"{f1_score:.4f}")

col5.metric("ROC-AUC", f"{roc_auc:.4f}")

# =====================================================

# BUSINESS INTERPRETATION

# =====================================================

st.markdown("---")

st.subheader("📈 Business Interpretation")

st.markdown("""

### Key Insights

* The model demonstrates reliable overall predictive performance for customer churn analysis.
* The ROC-AUC score indicates good discrimination capability between churn and non-churn customers.
* The model can support proactive retention campaigns and targeted customer engagement strategies.

### Business Relevance

The churn prediction system can help organizations:

* identify customers with elevated churn risk,
* support targeted retention strategies,
* improve customer engagement efforts,
* and assist decision-making using explainable AI techniques.
  """)

# =====================================================

# WHY LOGISTIC REGRESSION

# =====================================================

st.markdown("---")

st.subheader("Why Logistic Regression?")

st.markdown("""
Logistic Regression was selected as the final model because it:

* Provides strong predictive performance with high interpretability.
* Produces reliable churn probability estimates.
* Integrates effectively with SHAP explainability.
* Is computationally efficient and lightweight.
* Is easier to deploy and maintain in production systems.

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
