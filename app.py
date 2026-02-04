import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction System",
    layout="wide"
)

st.title("💳 Credit Risk Prediction – Early Warning System")
st.markdown("Predict **default risk** for single or multiple loan records.")

# ----------------------------------
# Load model & schema
# ----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/rf_model.pkl")
    feature_cols = joblib.load("model/feature_columns.pkl")
    return model, feature_cols

model, feature_columns = load_model()

# ----------------------------------
# Helper functions
# ----------------------------------
def preprocess_input(df):
    # Drop forbidden / leakage columns
    forbidden_cols = [c for c in df.columns if c.startswith("loan_status")]
    df = df.drop(columns=forbidden_cols, errors="ignore")

    # One-hot encode
    df = pd.get_dummies(df)

    # Align with training schema
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

def classify_risk(prob):
    if prob >= 0.5:
        return "High Risk (Likely Default)"
    elif prob >= 0.3:
        return "Watchlist (Early Warning)"
    else:
        return "Low Risk"

# ----------------------------------
# Sidebar – mode selection
# ----------------------------------
st.sidebar.header("⚙️ Prediction Mode")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Single Record", "Batch Upload"]
)

# ==================================
# SINGLE RECORD MODE
# ==================================
if mode == "Single Record":

    st.subheader("🔍 Single Record Risk Assessment")

    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amnt = st.number_input("Loan Amount", 1000, 100000, 15000)
            int_rate = st.number_input("Interest Rate (%)", 5.0, 35.0, 13.5)
            term = st.selectbox("Term", ["36 months", "60 months"])

        with col2:
            annual_inc = st.number_input("Annual Income", 10000, 300000, 60000)
            dti = st.number_input("Debt-to-Income Ratio", 0.0, 50.0, 18.0)
            delinq_2yrs = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)

        with col3:
            revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 35.0)
            installment = st.number_input("Monthly Installment", 50, 5000, 450)
            home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "MORTGAGE", "OWN", "OTHER"]
            )

        submit = st.form_submit_button("Predict Risk")

    if submit:
        input_df = pd.DataFrame([{
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "term": term,
            "annual_inc": annual_inc,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "revol_util": revol_util,
            "installment": installment,
            "home_ownership": home_ownership
        }])

        X = preprocess_input(input_df)
        prob = model.predict_proba(X)[0][1]

        st.metric("Default Probability", f"{prob:.2%}")
        st.success(f"Risk Category: **{classify_risk(prob)}**")

# ==================================
# BATCH MODE
# ==================================
else:

    st.subheader("📂 Batch Risk Assessment (CSV Upload)")

    uploaded_file = st.file_uploader(
        "Upload CSV file (multiple records)",
        type=["csv"]
    )

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(raw_df.head())

        X = preprocess_input(raw_df)
        probs = model.predict_proba(X)[:, 1]

        result_df = raw_df.copy()
        result_df["default_probability"] = probs
        result_df["risk_category"] = result_df["default_probability"].apply(classify_risk)

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df.head(20))

        # Download option
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            csv,
            "credit_risk_predictions.csv",
            "text/csv"
        )

        # ===============================
        # Portfolio Risk Distribution Dashboard
        # ===============================
        st.subheader("📊 Portfolio Risk Distribution Dashboard")

        # KPI Metrics
        col1, col2, col3 = st.columns(3)

        high_risk_pct = (result_df["risk_category"] == "High Risk (Likely Default)").mean() * 100
        watchlist_pct = (result_df["risk_category"] == "Watchlist (Early Warning)").mean() * 100
        avg_prob = result_df["default_probability"].mean()

        col1.metric("High Risk (%)", f"{high_risk_pct:.2f}%")
        col2.metric("Watchlist (%)", f"{watchlist_pct:.2f}%")
        col3.metric("Average Default Probability", f"{avg_prob:.2%}")

        # Risk Category Distribution
        st.markdown("### 🔴🟠🟢 Risk Category Distribution")

        risk_counts = result_df["risk_category"].value_counts()

        fig1, ax1 = plt.subplots()
        risk_counts.plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Number of Accounts")
        ax1.set_xlabel("Risk Category")
        ax1.set_title("Distribution of Credit Risk Categories")
        st.pyplot(fig1)

        # Default Probability Distribution
        st.markdown("### 📈 Default Probability Distribution")

        fig2, ax2 = plt.subplots()
        ax2.hist(result_df["default_probability"], bins=30)
        ax2.axvline(0.3, linestyle="--", label="Watchlist Threshold (0.3)")
        ax2.axvline(0.5, linestyle="--", label="High Risk Threshold (0.5)")
        ax2.set_xlabel("Predicted Default Probability")
        ax2.set_ylabel("Number of Accounts")
        ax2.set_title("Distribution of Default Probabilities")
        ax2.legend()
        st.pyplot(fig2)

        # Threshold Sensitivity
        st.markdown("### 🎚️ Threshold Sensitivity (Portfolio Impact)")

        thresholds = [0.2, 0.3, 0.4, 0.5]
        summary = []

        for t in thresholds:
            summary.append({
                "Threshold": t,
                "Flagged Accounts (%)": (result_df["default_probability"] >= t).mean() * 100
            })

        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Early Warning Credit Risk System | ML + Explainable AI")