"""Real-time scoring API (Module 6) — FastAPI.

Scores a single transaction with ALL deployed models (Logistic Regression,
Random Forest, XGBoost) independently and returns each model's fraud probability
and allow / review / block decision, plus a combined `max-risk` aggregate verdict.

Run locally:
    uvicorn api.main:app --reload
    # then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from infer import enrich, _ensure_required_columns   # noqa: E402
from ensemble import load_ensemble, score_record      # noqa: E402
from config import MODELS                              # noqa: E402

ENSEMBLE = load_ensemble(MODELS / "fraud_ensemble.joblib")
MODEL_NAMES = {k: v["model_name"] for k, v in ENSEMBLE["models"].items()}

app = FastAPI(
    title="E-Commerce Fraud Scoring API",
    description="Scores each transaction with " + ", ".join(MODEL_NAMES.values())
    + " independently, plus a max-risk aggregate verdict.",
)


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


@app.get("/")
def health():
    return {
        "status": "ok",
        "models": MODEL_NAMES,
        "aggregate_rule": ENSEMBLE.get("rule", "max-risk"),
    }


@app.post("/score")
def score(txn: Transaction):
    # Real-time single record: no transaction history is available, so
    # dest-history features are disabled (zero-filled), exactly as infer.py does
    # for streaming records. Enrichment happens once; each model then transforms
    # + scores independently inside score_record, which also computes the
    # max-risk aggregate and isolates any single model failure.
    row = _ensure_required_columns(pd.DataFrame([txn.model_dump()]))
    enriched = enrich(row, use_dest_history=False)
    return score_record(enriched, ENSEMBLE)
