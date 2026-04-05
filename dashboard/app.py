#streamlit run dashboard/app.py
import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Load Logistic Regression model and scaler
lr_model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/features.pkl")
feature_index = {name: idx for idx, name in enumerate(feature_names)}

# Load RNN model if available
rnn_model = None
if TORCH_AVAILABLE:
    try:
        from utils.evaluation import TabularRNN

        rnn_model = TabularRNN(n_features=len(feature_names))
        rnn_model.load_state_dict(torch.load("models/rnn_model.pt", map_location="cpu"))
        rnn_model.eval()
        print("RNN model loaded successfully")
    except Exception as e:
        print(f"Could not load RNN model: {e}")
        rnn_model = None

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("📊 Credit Risk Dashboard")

df = pd.read_csv("data/lending_club.csv", nrows=100000, low_memory=False)

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Choose prediction model:",
    ["Logistic Regression", "RNN"] if rnn_model else ["Logistic Regression"],
)

# Dashboard section
st.header("📈 Data Overview")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="loan_amnt", title="Loan Amount Distribution")
    st.plotly_chart(fig1, width="stretch")

with col2:
    fig2 = px.pie(df, names="loan_status", title="Default vs Paid")
    st.plotly_chart(fig2, width="stretch")

# Prediction section
st.header("🔮 Predict Loan Default Risk")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)

with col2:
    loan = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0)

with col3:
    credit = st.number_input(
        "FICO Score", min_value=300.0, max_value=850.0, value=700.0
    )

if st.button("🚀 Predict Risk"):
    # Prepare input features
    values = [0.0] * len(feature_names)

    if income:
        values[feature_index.get("annual_inc")] = income

    if loan:
        for name in ["loan_amnt", "funded_amnt", "funded_amnt_inv", "out_prncp"]:
            if name in feature_index:
                values[feature_index[name]] = loan

    if credit:
        values[feature_index.get("fico_range_low")] = credit
        values[feature_index.get("fico_range_high")] = credit

    X = pd.DataFrame([values], columns=feature_names)
    X_scaled = scaler.transform(X)

    # Get predictions from selected model
    st.subheader(f"Prediction Results ({model_option})")

    if model_option == "Logistic Regression":
        prob = lr_model.predict_proba(X_scaled)[0][1]
        risk_class = "High Risk ⚠️" if prob > 0.5 else "Low Risk ✅"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Default Probability", f"{prob:.2%}")
        with col2:
            st.metric("Risk Level", risk_class)
        with col3:
            st.metric("Model", "Logistic Regression")

        if prob > 0.5:
            st.error(f"⚠️ **High Risk** - Default probability: {prob:.2%}")
        else:
            st.success(f"✅ **Low Risk** - Default probability: {prob:.2%}")

    elif model_option == "RNN" and rnn_model is not None:
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            rnn_logits = rnn_model(X_tensor)
            rnn_prob = torch.sigmoid(rnn_logits).cpu().numpy().item()

        risk_class = "High Risk ⚠️" if rnn_prob > 0.5 else "Low Risk ✅"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Default Probability", f"{rnn_prob:.2%}")
        with col2:
            st.metric("Risk Level", risk_class)
        with col3:
            st.metric("Model", "RNN")

        if rnn_prob > 0.5:
            st.error(f"⚠️ **High Risk** - Default probability: {rnn_prob:.2%}")
        else:
            st.success(f"✅ **Low Risk** - Default probability: {rnn_prob:.2%}")

    # Comparison section
    if rnn_model is not None:
        st.divider()
        st.subheader("📊 Model Comparison")

        lr_prob = lr_model.predict_proba(X_scaled)[0][1]
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            rnn_logits = rnn_model(X_tensor)
            rnn_prob = torch.sigmoid(rnn_logits).cpu().numpy().item()

        comparison_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "RNN"],
                "Default Probability": [f"{lr_prob:.2%}", f"{rnn_prob:.2%}"],
                "Risk Level": [
                    "High Risk ⚠️" if lr_prob > 0.5 else "Low Risk ✅",
                    "High Risk ⚠️" if rnn_prob > 0.5 else "Low Risk ✅",
                ],
            }
        )

        st.dataframe(comparison_df, use_container_width=True)

        st.info(f"Probability Difference: {abs(lr_prob - rnn_prob):.2%}")
