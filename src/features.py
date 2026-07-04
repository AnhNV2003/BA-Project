"""Feature engineering (Module 4, draft).

Turns the raw base + synthetic columns into a numeric model matrix, and exposes
which features are BASE (PaySim) vs SYNTHETIC so we can run the leakage /
contribution experiment in train_validate.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature groups -------------------------------------------------------------
BASE_NUMERIC = [
    "amount", "log_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "orig_drained", "dest_was_empty",
    "is_transfer", "is_cash_out",
]

# "Realistic e-commerce" base: what a platform actually knows at authorization
# time. Drops PaySim's near-deterministic post-hoc balance-reconciliation
# features (errorBalance*, orig_drained, dest_was_empty, newbalance*) which make
# the toy problem trivially separable. Use this to get a genuine trade-off where
# the synthetic risk context matters.
BASE_REALISTIC = [
    "amount", "log_amount",
    "oldbalanceOrg", "oldbalanceDest",
    "is_transfer", "is_cash_out",
]
SYNTH_NUMERIC = [
    "account_age_days",
    "is_new_device", "shipping_billing_mismatch", "num_failed_payment_attempts",
    "ip_billing_distance_km", "log_ip_distance",
    "is_disposable_email", "high_risk_country",
    "hour_of_day", "is_night",
    "txn_count_last_24h", "time_since_last_hours", "account_txn_total",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered model-ready columns added."""
    x = df.copy()

    # ---- base / balance-signature features (the strong PaySim signal) ----
    x["log_amount"] = np.log1p(x["amount"])
    # PaySim's classic fraud tells: balances don't reconcile with the amount.
    x["errorBalanceOrig"] = x["oldbalanceOrg"] - x["amount"] - x["newbalanceOrig"]
    x["errorBalanceDest"] = x["oldbalanceDest"] + x["amount"] - x["newbalanceDest"]
    x["orig_drained"] = ((x["newbalanceOrig"] == 0) & (x["oldbalanceOrg"] > 0)).astype(int)
    x["dest_was_empty"] = (x["oldbalanceDest"] == 0).astype(int)
    x["is_transfer"] = (x["type"] == "TRANSFER").astype(int)
    x["is_cash_out"] = (x["type"] == "CASH_OUT").astype(int)

    # ---- synthetic ----
    x["log_ip_distance"] = np.log1p(x["ip_billing_distance_km"])

    return x


def feature_matrix(df: pd.DataFrame, groups: str = "all"):
    """Return (X, feature_names) for groups in {'base','synth','all'}."""
    x = build_features(df)
    if groups == "base":
        cols = BASE_NUMERIC
    elif groups == "synth":
        cols = SYNTH_NUMERIC
    elif groups == "realistic":
        cols = BASE_REALISTIC + SYNTH_NUMERIC
    else:
        cols = BASE_NUMERIC + SYNTH_NUMERIC
    return x[cols].astype(float), cols
