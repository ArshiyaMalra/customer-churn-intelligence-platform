import streamlit as st
import joblib
import pandas as pd

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Load Custom CSS
# =========================
from utils.load_css import load_css

load_css()

# =========================
# Load Saved Files
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_logistic_model.pkl")
    scaler = joblib.load("models/final_scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    metrics = joblib.load("models/model_performance.pkl")
    return model, scaler, feature_names, metrics


@st.cache_data
def load_data():
    return pd.read_csv("data/telco_churn_encoded_full.csv")


# Load artifacts
model, scaler, feature_names, metrics = load_artifacts()
df = load_data()

# =========================
# Safely Read Metrics
# =========================
roc_auc = float(metrics.get("ROC-AUC", metrics.get("roc_auc", 0)))
accuracy = float(metrics.get("Accuracy", metrics.get("accuracy", 0)))

# =========================
# Hero Section
# =========================
st.title("📊 Customer Churn Intelligence Platform")
st.subheader(
    "Predict Customer Attrition and Generate Actionable Retention Strategies"
)

st.markdown(
    """
    Welcome to an end-to-end analytics platform designed to identify
    customers at risk of churn, explain the key drivers behind each
    prediction, and recommend targeted retention strategies.
    """
)

# =========================
# KPI Cards
# =========================
st.markdown("---")
st.header("📈 Model Snapshot")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "Logistic Regression")

with col2:
    st.metric("ROC-AUC", f"{roc_auc:.4f}")

with col3:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col4:
    st.metric("Features", len(feature_names))

# =========================
# Dataset Summary
# =========================
st.markdown("---")
st.header("📂 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    st.metric("Input Features", df.shape[1] - 1)

with col3:
    st.metric("Churn Rate", f"{df['Churn'].mean() * 100:.2f}%")

# =========================
# Project Overview
# =========================
st.markdown("---")
st.header("📌 Project Overview")

st.markdown(
    """
    This platform predicts customer churn using a trained
    Logistic Regression model and provides actionable retention
    recommendations supported by SHAP-based explainable AI.

    ### Key Capabilities
    - 🔮 Individual customer churn prediction
    - 📊 Model performance dashboard
    - 🔍 SHAP explainability analysis
    - 🎯 Retention strategy recommendations
    - 💰 Business impact simulation

    ### Technology Stack
    - Python
    - Streamlit
    - scikit-learn
    - SHAP
    - Pandas
    - Matplotlib
    """
)

# =========================
# Navigation Guide
# =========================
st.markdown("---")
st.header("🧭 Navigation Guide")

st.info(
    """
    Use the sidebar to explore the following dashboards:

    • Predict Churn  
    • Model Performance  
    • SHAP Analysis  
    • Retention Strategies  
    • Business Impact Simulator
    """
)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Built as a final-year B.Tech CSE project for customer churn prediction and retention analytics."
)