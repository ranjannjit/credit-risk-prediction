#uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
import os
import sys
from fastapi import FastAPI, Response, HTTPException, Query
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

app = FastAPI(
    title="Credit Risk API",
    description="Predict loan default risk using Logistic Regression and RNN models",
    version="1.0.0",
)

# Load Logistic Regression model and preprocessing artifacts
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

INPUT_FEATURE_MAP = {
    "income": "annual_inc",
    "loan_amount": "loan_amnt",
    "credit_score": "fico_range_low",
}


@app.get("/")
def home():
    models_available = ["logistic"]
    if rnn_model is not None:
        models_available.append("rnn")

    return {
        "message": "Credit Risk Prediction API",
        "available_models": models_available,
        "endpoints": {
            "GET /": "API information",
            "POST /predict": "Predict default risk (use 'model' query parameter to select model)",
            "POST /predict_both": "Get predictions from both models (if RNN available)",
        },
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/predict")
def predict(
    data: dict,
    model: str = Query(
        "logistic", description="Model to use for prediction: 'logistic' or 'rnn'"
    ),
):
    # Validate model selection
    if model not in ["logistic", "rnn"]:
        raise HTTPException(
            status_code=400, detail="Invalid model. Choose 'logistic' or 'rnn'."
        )

    if model == "rnn" and rnn_model is None:
        raise HTTPException(
            status_code=400,
            detail="RNN model not available. PyTorch may not be installed. Use 'logistic' model instead.",
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object."
        )

    unknown_fields = set(data) - set(INPUT_FEATURE_MAP)
    if unknown_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields: {sorted(unknown_fields)}. Supported fields are: {sorted(INPUT_FEATURE_MAP)}.",
        )

    values = np.zeros(len(feature_names), dtype=float)

    if "income" in data:
        values[feature_index[INPUT_FEATURE_MAP["income"]]] = data["income"]

    if "loan_amount" in data:
        loan_amount = data["loan_amount"]
        for name in ["loan_amnt", "funded_amnt", "funded_amnt_inv", "out_prncp"]:
            if name in feature_index:
                values[feature_index[name]] = loan_amount

    if "credit_score" in data:
        score = data["credit_score"]
        for name in ["fico_range_low", "fico_range_high"]:
            if name in feature_index:
                values[feature_index[name]] = score

    features = pd.DataFrame([values], columns=feature_names)
    features_scaled = scaler.transform(features)

    # Get prediction from selected model
    if model == "logistic":
        prob = lr_model.predict_proba(features_scaled)[0][1]
    else:  # model == "rnn"
        with torch.no_grad():
            X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            rnn_logits = rnn_model(X_tensor)
            prob = float(torch.sigmoid(rnn_logits).cpu().numpy()[0][0])

    print(f"[{model.upper()}] payload:", data, "probability:", prob)

    return {
        "model": model,
        "default_probability": float(prob),
        "risk": "HIGH" if prob > 0.5 else "LOW",
    }


@app.post("/predict_both")
def predict_both(data: dict):
    """Get predictions from both Logistic Regression and RNN models."""
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object."
        )

    unknown_fields = set(data) - set(INPUT_FEATURE_MAP)
    if unknown_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields: {sorted(unknown_fields)}. Supported fields are: {sorted(INPUT_FEATURE_MAP)}.",
        )

    values = np.zeros(len(feature_names), dtype=float)

    if "income" in data:
        values[feature_index[INPUT_FEATURE_MAP["income"]]] = data["income"]

    if "loan_amount" in data:
        loan_amount = data["loan_amount"]
        for name in ["loan_amnt", "funded_amnt", "funded_amnt_inv", "out_prncp"]:
            if name in feature_index:
                values[feature_index[name]] = loan_amount

    if "credit_score" in data:
        score = data["credit_score"]
        for name in ["fico_range_low", "fico_range_high"]:
            if name in feature_index:
                values[feature_index[name]] = score

    features = pd.DataFrame([values], columns=feature_names)
    features_scaled = scaler.transform(features)

    # Logistic Regression prediction
    lr_prob = lr_model.predict_proba(features_scaled)[0][1]

    # RNN prediction (if available)
    rnn_prob = None
    if rnn_model is not None:
        with torch.no_grad():
            X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            rnn_logits = rnn_model(X_tensor)
            rnn_prob = float(torch.sigmoid(rnn_logits).cpu().numpy()[0][0])

    print("[BOTH] payload:", data)

    result = {
        "logistic": {
            "default_probability": float(lr_prob),
            "risk": "HIGH" if lr_prob > 0.5 else "LOW",
        }
    }

    if rnn_prob is not None:
        result["rnn"] = {
            "default_probability": rnn_prob,
            "risk": "HIGH" if rnn_prob > 0.5 else "LOW",
        }
        result["probability_difference"] = abs(lr_prob - rnn_prob)

    return result
