import streamlit as st
from utils.load_css import load_css

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Business Impact Simulator",
    page_icon="💰",
    layout="wide"
)

# Load custom CSS
load_css()

# =========================================================
# PAGE HEADER
# =========================================================
st.title("💰 Business Impact Simulator")
st.markdown("""
Estimate the financial impact of customer retention campaigns
based on churn predictions.
""")

# =========================================================
# INPUT SECTION (ON MAIN PAGE, NOT SIDEBAR)
# =========================================================
st.markdown("---")
st.subheader("📝 Simulation Inputs")

col1, col2 = st.columns(2)

with col1:
    at_risk_customers = st.number_input(
        "Number of At-Risk Customers",
        min_value=1,
        value=1000,
        step=100
    )

    average_clv = st.number_input(
        "Average Customer Lifetime Value ($)",
        min_value=100.0,
        value=4500.0,
        step=100.0
    )

with col2:
    intervention_cost = st.number_input(
        "Retention Cost per Customer ($)",
        min_value=1.0,
        value=150.0,
        step=10.0
    )

    success_rate = st.slider(
        "Retention Success Rate (%)",
        min_value=1,
        max_value=100,
        value=25
    )

# =========================================================
# CALCULATIONS
# =========================================================
customers_retained = at_risk_customers * (success_rate / 100)
revenue_saved = customers_retained * average_clv
campaign_cost = customers_retained * intervention_cost
net_profit = revenue_saved - campaign_cost

if campaign_cost > 0:
    roi = (net_profit / campaign_cost) * 100
else:
    roi = 0

# =========================================================
# METRICS DASHBOARD
# =========================================================
st.markdown("---")
st.subheader("📊 Financial Impact Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Customers Retained",
        f"{customers_retained:,.0f}"
    )

with col2:
    st.metric(
        "Revenue Saved",
        f"${revenue_saved:,.0f}"
    )

with col3:
    st.metric(
        "Campaign Cost",
        f"${campaign_cost:,.0f}"
    )

col4, col5 = st.columns(2)

with col4:
    st.metric(
        "Net Profit",
        f"${net_profit:,.0f}"
    )

with col5:
    st.metric(
        "ROI",
        f"{roi:.2f}%"
    )

# =========================================================
# EXECUTIVE INTERPRETATION
# =========================================================
st.markdown("---")
st.subheader("📈 Executive Interpretation")

if roi > 300:
    st.success(
        "Excellent ROI. The retention campaign is highly profitable "
        "and financially attractive."
    )
elif roi > 100:
    st.info(
        "Strong ROI. The campaign is expected to generate substantial "
        "business value."
    )
elif roi > 0:
    st.warning(
        "Positive ROI. The campaign is profitable, but assumptions may "
        "be optimized further."
    )
else:
    st.error(
        "Negative ROI. The retention campaign is not financially viable "
        "under current assumptions."
    )

# =========================================================
# BUSINESS INSIGHTS
# =========================================================
st.markdown("---")
st.subheader("💡 Business Insights")

st.markdown(f"""
- **Expected Customers Retained:** {customers_retained:,.0f}
- **Estimated Revenue Preserved:** ${revenue_saved:,.0f}
- **Total Retention Investment:** ${campaign_cost:,.0f}
- **Projected Net Profit:** ${net_profit:,.0f}
- **Return on Investment:** {roi:.2f}%

This simulator helps decision-makers estimate the financial impact
of retention campaigns before implementation.
""")

# =========================================================
# CALCULATION LOGIC
# =========================================================
st.markdown("---")
st.subheader("🧮 Calculation Logic")

st.markdown("""
- **Customers Retained** = At-Risk Customers × Retention Success Rate
- **Revenue Saved** = Customers Retained × Average Customer Lifetime Value
- **Campaign Cost** = Customers Retained × Retention Cost per Customer
- **Net Profit** = Revenue Saved − Campaign Cost
- **ROI (%)** = (Net Profit ÷ Campaign Cost) × 100
""")