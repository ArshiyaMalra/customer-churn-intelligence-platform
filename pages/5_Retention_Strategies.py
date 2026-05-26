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

# =====================================================
# PAGE HEADER
# =====================================================
st.title("Retention Strategy Dashboard")

st.markdown("""
This dashboard connects major churn drivers identified by the
machine learning model with actionable customer retention strategies.
""")

# =====================================================
# STRATEGY DATAFRAME
# =====================================================
strategy_df = pd.DataFrame({

    "Primary Churn Driver":
        list(strategy_map.keys()),

    "Recommended Retention Strategy":
        list(strategy_map.values())
})

# =====================================================
# OVERVIEW METRICS
# =====================================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Churn Drivers",
        len(strategy_df)
    )

with col2:

    st.metric(
        "Strategy Categories",
        "Customer Retention"
    )

with col3:

    st.metric(
        "Decision Support",
        "Enabled"
    )

# =====================================================
# STRATEGY TABLE
# =====================================================
st.markdown("---")

st.subheader(
    "Churn Driver to Strategy Mapping"
)

st.dataframe(
    strategy_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# INTERACTIVE EXPLORER
# =====================================================
st.markdown("---")

st.subheader(
    "Explore Retention Recommendations"
)

selected_driver = st.selectbox(
    "Select a Primary Churn Driver",
    strategy_df["Primary Churn Driver"]
)

selected_strategy = strategy_map[
    selected_driver
]

# =====================================================
# DISPLAY SELECTED STRATEGY
# =====================================================
st.markdown("### Recommended Business Action")

st.success(
    selected_strategy
)


