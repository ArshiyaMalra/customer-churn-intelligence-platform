# =========================================================
# CENTRALIZED STRATEGY MAPPING
# =========================================================

strategy_map = {

    "High Monthly Charges":
        "Provide personalized discounts, bundle offers, or lower-cost plan recommendations to reduce pricing concerns.",

    "Low Customer Tenure":
        "Implement onboarding support, engagement campaigns, and early customer success follow-ups.",

    "Month-to-Month Contract":
        "Encourage customers to switch to long-term contracts using loyalty incentives and discounted pricing.",

    "Fiber Optic Service Issues":
        "Prioritize service quality improvements and proactive technical support for fiber customers.",

    "Payment Friction":
        "Encourage automatic payment enrollment through incentives and simplified billing support.",

    "Lack of Support Services":
        "Offer complimentary technical support and online security add-ons to improve customer satisfaction.",

    "General Churn Risk":
        "Maintain proactive customer engagement and monitor satisfaction through loyalty outreach programs."
}


# =========================================================
# CONVERT FEATURE NAME TO BUSINESS DRIVER
# =========================================================

def categorize_churn_reason(feature_name):

    feature = str(feature_name).lower()

    # ---------------------------------------------
    # Pricing-related issues
    # ---------------------------------------------
    if (
        "monthlycharges" in feature
        or "chargelevel" in feature
        or "totalcharges" in feature
    ):
        return "High Monthly Charges"

    # ---------------------------------------------
    # Customer tenure / loyalty
    # ---------------------------------------------
    elif (
        "tenure" in feature
        or "tenurebucket" in feature
    ):
        return "Low Customer Tenure"

    # ---------------------------------------------
    # Contract-related risk
    # ---------------------------------------------
    elif "contract" in feature:
        return "Month-to-Month Contract"

    # ---------------------------------------------
    # Fiber internet dissatisfaction
    # ---------------------------------------------
    elif (
        "fiber optic" in feature
        or "internetservice_fiber optic" in feature
    ):
        return "Fiber Optic Service Issues"

    # ---------------------------------------------
    # Technical support / security gaps
    # ---------------------------------------------
    elif (
        "techsupport" in feature
        or "onlinesecurity" in feature
    ):
        return "Lack of Support Services"

    # ---------------------------------------------
    # Payment-related friction
    # ---------------------------------------------
    elif "paymentmethod" in feature:
        return "Payment Friction"

    # ---------------------------------------------
    # Default
    # ---------------------------------------------
    else:
        return "General Churn Risk"


# =========================================================
# GET RETENTION STRATEGY
# =========================================================

def get_reason_based_strategy(reason):

    return strategy_map.get(
        reason,
        strategy_map["General Churn Risk"]
    )