# ==========================================================
# UPI-Guard++ Production-Safe API
# ==========================================================

from fastapi import FastAPI
import pickle
import pandas as pd
import os
import sys

# Add project root to path so we can import preprocess
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocess import build_features

app = FastAPI()

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "threshold.txt")

model = pickle.load(open(MODEL_PATH, "rb"))
features = pickle.load(open(FEATURE_PATH, "rb"))

if os.path.exists(THRESHOLD_PATH):
    threshold = float(open(THRESHOLD_PATH).read())
else:
    threshold = 0.5


@app.post("/predict")
def predict(data: dict):

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Apply same feature engineering as training
    df = build_features(df)

    # Drop non-model columns if present
    df = df.drop(columns=[
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "fraud_flag"
    ], errors="ignore")

    # Align columns
    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    decision = "Fraud" if prob > threshold else "Safe"

    return {
        "fraud_probability": float(prob),
        "decision": decision
    }
