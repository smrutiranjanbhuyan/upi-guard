
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import os
import sys
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = FastAPI(
    title="UPI-Guard++ Fraud Detection API",
    description="Graph-aware, cost-sensitive UPI fraud detection using XGBoost",
    version="2.0.0"
)

# ── CORS — allow Streamlit UI (localhost:8501) ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ──────────────────────────────────
MODEL_PATH     = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
FEATURE_PATH   = os.path.join(BASE_DIR, "model", "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "threshold.pkl")

model     = pickle.load(open(MODEL_PATH, "rb"))
features  = list(pickle.load(open(FEATURE_PATH, "rb")))

if os.path.exists(THRESHOLD_PATH):
    threshold = float(pickle.load(open(THRESHOLD_PATH, "rb")))
else:
    threshold = 0.18

print(f"Model loaded | Features: {len(features)} | Threshold: {threshold:.4f}")


# ── Request Schema ────────────────────────────────────────
class TransactionRequest(BaseModel):
    transaction_id:    str   = Field(example="TXN_001")
    timestamp:         str   = Field(example="2024-03-15T14:30:00")
    sender_id:         str   = Field(example="USER_042")
    receiver_id:       str   = Field(example="USER_899")
    amount:            float = Field(gt=0, example=48500.0)
    transaction_type:  str   = Field(example="P2P")
    merchant_category: str   = Field(example="Food")
    sender_state:      str   = Field(example="Odisha")
    receiver_state:    str   = Field(example="Maharashtra")
    sender_bank:       str   = Field(example="SBI")
    receiver_bank:     str   = Field(example="HDFC")
    device_type:       str   = Field(example="Android")
    network_type:      str   = Field(example="4G")
    account_age_days:  int   = Field(ge=0, example=45)
    txn_velocity_1h:   Optional[float] = 1.0
    sender_mean_amt:   Optional[float] = None
    sender_std_amt:    Optional[float] = None
    sender_txn_count:  Optional[int]   = 1
    unique_receivers:  Optional[int]   = 1
    sender_pagerank:   Optional[float] = 0.0
    sender_degree:     Optional[float] = 0.0


# ── Single-row safe feature engineering ──────────────────
def engineer_single_row(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    ts = pd.to_datetime(data["timestamp"])
    df["hour"]           = ts.hour
    df["day_of_week"]    = ts.dayofweek
    df["is_weekend"]     = int(ts.dayofweek >= 5)
    df["month"]          = ts.month
    df["is_night"]       = int(0 <= ts.hour <= 5)
    df["is_salary_week"] = int(ts.day <= 7)

    df["cross_state"] = int(data["sender_state"] != data["receiver_state"])

    amt      = data["amount"]
    mean_amt = data.get("sender_mean_amt") or amt
    std_amt  = data.get("sender_std_amt")  or 1.0
    df["sender_mean_amt"]  = mean_amt
    df["sender_std_amt"]   = std_amt
    df["sender_txn_count"] = data.get("sender_txn_count") or 1
    df["unique_receivers"] = data.get("unique_receivers") or 1
    df["amount_zscore"]    = (amt - mean_amt) / (std_amt + 1e-5)
    df["txn_velocity_1h"]  = data.get("txn_velocity_1h") or 1.0
    df["sender_pagerank"]  = data.get("sender_pagerank") or 0.0
    df["sender_degree"]    = data.get("sender_degree")   or 0.0

    cat_cols = [
        "transaction_type", "merchant_category",
        "sender_state", "receiver_state",
        "sender_bank", "receiver_bank",
        "device_type", "network_type"
    ]
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)

    df = df.drop(columns=[
        "transaction_id", "timestamp", "sender_id",
        "receiver_id", "fraud_flag"
    ], errors="ignore")

    df = df.reindex(columns=features, fill_value=0)
    return df


# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "UPI-Guard++ Fraud Detection API",
        "version": "2.0.0",
        "status": "running",
        "threshold": round(threshold, 4),
        "features": len(features)
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(features),
        "threshold": round(threshold, 4)
    }

@app.post("/predict")
def predict(request: TransactionRequest):
    try:
        data = request.dict()
        df   = engineer_single_row(data)
        prob     = float(model.predict_proba(df)[0][1])
        decision = "Fraud" if prob > threshold else "Safe"
        risk     = "High" if prob > 0.70 else ("Medium" if prob > threshold else "Low")
        return {
            "transaction_id":    data["transaction_id"],
            "fraud_probability": round(prob, 6),
            "decision":          decision,
            "risk_level":        risk,
            "threshold_used":    round(threshold, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))