from utils.load_css import load_css

load_css()

import streamlit as st
import pandas as pd

from utils.strategy_mapping import (
    strategy_map
)

st.set_page_config(
    page_title="Retention Strategies",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Retention Strategy Dashboard")
st.markdown("""
This dashboard presents the standardized mapping between major churn
drivers and the recommended customer retention actions.

The same strategy mapping is used throughout the application,
including the Predict Churn page, ensuring full consistency.
""")

# -----------------------------------------------------
# Convert dictionary to DataFrame
# -----------------------------------------------------
strategy_df = pd.DataFrame({
    "Primary Churn Driver": list(strategy_map.keys()),
    "Recommended Retention Strategy": list(strategy_map.values())
})

# -----------------------------------------------------
# Display strategy table
# -----------------------------------------------------
st.subheader("📋 Churn Driver to Strategy Mapping")

st.dataframe(
    strategy_df,
    use_container_width=True,
    hide_index=True
)

# -----------------------------------------------------
# Interactive strategy explorer
# -----------------------------------------------------
st.markdown("---")
st.subheader("🔍 Explore a Specific Churn Driver")

selected_driver = st.selectbox(
    "Select a churn driver",
    strategy_df["Primary Churn Driver"]
)

selected_strategy = strategy_map[selected_driver]

st.info(f"**Recommended Strategy:** {selected_strategy}")

# -----------------------------------------------------
# Business Insights
# -----------------------------------------------------
st.markdown("---")
st.subheader("💡 Business Insights")

st.markdown("""
The retention strategies shown here are derived from the most
important churn drivers identified using SHAP explainability.

By connecting model insights to actionable interventions,
the organization can take targeted steps to reduce churn and
maximize customer lifetime value.
""")

# -----------------------------------------------------
# Strategic Summary
# -----------------------------------------------------
st.markdown("---")
st.subheader("📈 Strategic Summary")

st.success("""
This dashboard serves as a decision-support tool that translates
machine learning insights into practical business actions for
customer retention.
""")