
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "reports" / "prediction_history.csv"
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Prediction System")
st.write("Predict loan default risk and review prediction history.")

# Applicant input
st.subheader("Applicant Details")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income", min_value=1.0, value=50000.0
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=1.0, value=15000.0
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=700
        )

    with col2:
        existing_debt = st.number_input(
            "Existing Debt", min_value=0.0, value=5000.0
        )
        previous_defaults = st.selectbox(
            "Previous Defaults", [0, 1, 2]
        )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    applicant = {
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "previous_defaults": previous_defaults
    }

    try:
        response = requests.post(
            API_URL, json=applicant, timeout=30
        )
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Default Probability",
            f"{result['default_probability'] * 100:.2f}%"
        )
        col2.metric("Risk Level", result["risk_level"])
        col3.metric(
            "Predicted Default",
            str(result["predicted_default"])
        )

        if result["risk_level"] == "Low":
            st.success(result["prediction_message"])
        elif result["risk_level"] == "Medium":
            st.warning(result["prediction_message"])
        else:
            st.error(result["prediction_message"])

    except requests.RequestException as error:
        st.error(f"Could not contact prediction API: {error}")

# Prediction history
st.divider()
st.subheader("📊 Prediction History")

if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH)

    if not history.empty:
        st.metric("Total Predictions", len(history))

        st.write("### Risk Level Distribution")
        risk_counts = (
            history["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )
        st.bar_chart(risk_counts)

        st.write("### Recent Predictions")
        st.dataframe(history.tail(10), use_container_width=True)
    else:
        st.info("No predictions recorded yet.")
else:
    st.info("Submit a prediction to create prediction history.")
