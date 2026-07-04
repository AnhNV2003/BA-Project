"""Real-time scoring API (Module 6, draft) — FastAPI.

Loads the trained model and scores a single transaction, returning a fraud
probability and an allow / review / block decision.

Run locally:
    uvicorn api.main:app --reload
    # then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import pathlib
import sys

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from features import build_features            # noqa: E402
from config import MODELS                      # noqa: E402

BUNDLE = joblib.load(MODELS / "fraud_model.joblib")
MODEL, FEATURES, THRESHOLD = BUNDLE["model"], BUNDLE["features"], float(BUNDLE["threshold"])
BLOCK_THRESHOLD = float(max(0.9, THRESHOLD))   # high-confidence auto-block

app = FastAPI(title="E-Commerce Fraud Scoring API",
              description=f"Model: {BUNDLE['model_name']} | review threshold: {THRESHOLD:.3f}")


class Transaction(BaseModel):
    # --- base (PaySim) --- defaults form one complete fraud-like example
    type: str = "TRANSFER"
    amount: float = 21279.19
    oldbalanceOrg: float = 21279.19
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    # --- synthetic risk context (enriched before scoring in production) ---
    account_age_days: int = 52
    is_new_device: int = 1
    shipping_billing_mismatch: int = 1
    num_failed_payment_attempts: int = 3
    ip_billing_distance_km: float = 1299.7
    is_disposable_email: int = 0
    high_risk_country: int = 1
    hour_of_day: int = 6
    is_night: int = 1
    txn_count_last_24h: int = 0
    time_since_last_hours: int = -1
    account_txn_total: int = 1


def decide(score: float) -> str:
    if score >= BLOCK_THRESHOLD:
        return "block"
    if score >= THRESHOLD:
        return "review"
    return "allow"


@app.get("/")
def health():
    return {"status": "ok", "model": BUNDLE["model_name"], "threshold": THRESHOLD}


@app.post("/score")
def score(txn: Transaction):
    row = pd.DataFrame([txn.model_dump()])
    X = build_features(row)[FEATURES].astype(float)
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    return {
        "fraud_probability": round(prob, 4),
        "decision": decide(prob),
        "review_threshold": THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    }
