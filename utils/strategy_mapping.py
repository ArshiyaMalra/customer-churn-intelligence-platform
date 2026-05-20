# =========================================================
# Churn Driver to Retention Strategy Mapping
# =========================================================

strategy_map = {
    "High Monthly Charges":
        "Provide a personalized discount or bundle plan to reduce cost concerns.",

    "High Total Charges":
        "Offer bill credits or flexible payment options.",

    "Low Customer Tenure":
        "Implement onboarding support and early engagement campaigns.",

    "Month-to-Month Contract":
        "Offer discounted annual or two-year contracts to improve retention.",

    "Electronic Check Payment":
        "Encourage enrollment in automatic payment methods.",

    "No Technical Support":
        "Provide complimentary technical support for a limited period.",

    "No Online Security":
        "Offer free online security add-ons to increase perceived value.",

    "Fiber Optic Service Issues":
        "Improve service quality and provide loyalty incentives."
}


# =========================================================
# Convert Raw SHAP Feature Name to Business-Friendly Driver
# =========================================================
def categorize_churn_reason(feature_name):
    feature = str(feature_name).lower()

    if "monthlycharges" in feature:
        return "High Monthly Charges"

    elif "totalcharges" in feature:
        return "High Total Charges"

    elif "tenure" in feature:
        return "Low Customer Tenure"

    elif "contract_month-to-month" in feature:
        return "Month-to-Month Contract"

    elif "paymentmethod_electronic check" in feature:
        return "Electronic Check Payment"

    elif "techsupport_no" in feature:
        return "No Technical Support"

    elif "onlinesecurity_no" in feature:
        return "No Online Security"

    elif "internetservice_fiber optic" in feature:
        return "Fiber Optic Service Issues"

    else:
        return "General Churn Risk"


# =========================================================
# Get Recommended Retention Strategy
# =========================================================
def get_reason_based_strategy(reason):
    return strategy_map.get(
        reason,
        "Maintain proactive engagement and monitor customer satisfaction."
    )