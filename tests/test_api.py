"""Integration tests for the 3-model scoring API."""
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("fastapi")

if not (ROOT / "models" / "fraud_ensemble.joblib").exists():
    pytest.skip("fraud_ensemble.joblib not built; run train_validate.py", allow_module_level=True)

from api.main import app, MODEL_NAMES  # noqa: E402

client = TestClient(app)
EXPECTED_MODELS = {"logreg", "rf", "xgb"}


def test_health_lists_models():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models"]) == set(MODEL_NAMES)
    assert body["aggregate_rule"] == "max-risk"


def test_score_returns_all_models_plus_aggregate():
    r = client.post("/score", json={})   # defaults = fraud-like example
    assert r.status_code == 200
    body = r.json()
    assert set(body["models"]) == EXPECTED_MODELS
    for entry in body["models"].values():
        assert "fraud_probability" in entry
        assert entry["decision"] in {"allow", "review", "block"}
    assert body["aggregate"]["decision"] in {"allow", "review", "block"}
    assert body["aggregate"]["rule"] == "max-risk"


def test_fraud_like_defaults_escalate():
    # The default payload is a drained-origin TRANSFER with high-risk context.
    body = client.post("/score", json={}).json()
    assert body["aggregate"]["decision"] in {"review", "block"}


def test_benign_payment_allows():
    benign = {
        "type": "PAYMENT", "amount": 42.5,
        "oldbalanceOrg": 5000, "newbalanceOrig": 4957.5,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "is_new_device": 0, "shipping_billing_mismatch": 0,
        "num_failed_payment_attempts": 0, "ip_billing_distance_km": 5.0,
        "high_risk_country": 0, "is_night": 0,
        "account_age_days": 900, "account_txn_total": 300,
    }
    body = client.post("/score", json=benign).json()
    assert body["aggregate"]["decision"] == "allow"


def test_aggregate_is_max_severity_of_models():
    body = client.post("/score", json={}).json()
    sev = {"allow": 0, "review": 1, "block": 2}
    model_decisions = [m["decision"] for m in body["models"].values() if "decision" in m]
    assert sev[body["aggregate"]["decision"]] == max(sev[d] for d in model_decisions)
