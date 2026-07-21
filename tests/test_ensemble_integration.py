"""Integration tests over the trained ensemble bundle + drift.

Skipped if fraud_ensemble.joblib hasn't been built yet.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "monitoring"))

BUNDLE_PATH = ROOT / "models" / "fraud_ensemble.joblib"
if not BUNDLE_PATH.exists():
    pytest.skip("fraud_ensemble.joblib not built; run train_validate.py", allow_module_level=True)

from ensemble import (  # noqa: E402
    REQUIRED_MODEL_KEYS, load_ensemble, score_batch, score_record,
)
from infer import enrich, _ensure_required_columns  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return load_ensemble(BUNDLE_PATH)


def test_bundle_has_valid_model_entries(bundle):
    assert bundle["models"], "ensemble has no models"
    for key, entry in bundle["models"].items():
        assert REQUIRED_MODEL_KEYS <= set(entry), f"{key} missing keys"
        assert 0.0 <= float(entry["threshold"]) <= 1.0
        assert entry["matrix"] in {"tree", "linear"}
        assert len(entry["features"]) > 0


def _record(**overrides):
    base = {
        "type": "TRANSFER", "amount": 21279.19,
        "oldbalanceOrg": 21279.19, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "account_age_days": 52, "is_new_device": 1,
        "shipping_billing_mismatch": 1, "num_failed_payment_attempts": 3,
        "ip_billing_distance_km": 1299.7, "is_disposable_email": 0,
        "high_risk_country": 1, "hour_of_day": 6, "is_night": 1,
        "txn_count_last_24h": 0, "time_since_last_hours": -1, "account_txn_total": 1,
    }
    base.update(overrides)
    return base


def test_record_and_batch_paths_agree(bundle):
    # Same record scored as a single record (API path) and inside a batch
    # (Streamlit path) must yield identical per-model + aggregate decisions.
    rows = [_record(), _record(type="PAYMENT", amount=42.5, oldbalanceOrg=5000,
                               newbalanceOrig=4957.5, is_new_device=0,
                               shipping_billing_mismatch=0, num_failed_payment_attempts=0,
                               ip_billing_distance_km=5.0, high_risk_country=0,
                               is_night=0, account_age_days=900, account_txn_total=300)]
    df = _ensure_required_columns(pd.DataFrame(rows))
    enriched = enrich(df, use_dest_history=False)
    batch = score_batch(enriched, bundle)

    for i, row in enumerate(rows):
        rec_enriched = enrich(_ensure_required_columns(pd.DataFrame([row])), use_dest_history=False)
        rec = score_record(rec_enriched, bundle)
        assert rec["aggregate"]["decision"] == batch.iloc[i]["agg_decision"]
        for key, entry in rec["models"].items():
            assert entry["decision"] == batch.iloc[i][f"{key}_decision"]


def test_per_model_drift_rows_and_trigger():
    from drift import load, split_reference_current, inject_drift, psi_map, RETRAIN_PSI

    df = load()
    ensemble = load_ensemble(BUNDLE_PATH)
    ref, cur, _ = split_reference_current(df)
    rng = np.random.default_rng(0)

    natural = psi_map(ref, cur, ensemble)
    drifted = psi_map(ref, inject_drift(cur, rng), ensemble)

    # one prediction row per model
    for key in ensemble["models"]:
        assert f"PREDICTION_SCORE_{key}" in natural

    # natural split is stable; injected campaign trips at least one row
    assert all(v < RETRAIN_PSI for v in natural.values())
    assert any(v >= RETRAIN_PSI for v in drifted.values())
