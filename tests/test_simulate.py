"""Tests for the live-feed transaction simulator (src/simulate.py)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from simulate import decision_timeline, generate_pool, score_stream  # noqa: E402


def test_decision_timeline_buckets_and_colors_columns():
    df = pd.DataFrame({
        "arrival": list(range(1, 21)),
        "agg_decision": (["allow"] * 8 + ["review"] * 2)     # bucket 1..10
                        + (["allow"] * 7 + ["block"] * 2 + ["review"]),  # bucket 11..20
    })
    tl = decision_timeline(df, bin_size=10)
    assert list(tl.columns) == ["review", "block"]   # order matters for chart colors
    assert tl.loc[1, "review"] == 2 and tl.loc[1, "block"] == 0
    assert tl.loc[11, "block"] == 2 and tl.loc[11, "review"] == 1


def test_decision_timeline_empty():
    tl = decision_timeline(pd.DataFrame(columns=["arrival", "agg_decision"]))
    assert list(tl.columns) == ["review", "block"]
    assert len(tl) == 0

BUNDLE_PATH = ROOT / "models" / "fraud_ensemble.joblib"


def test_generate_pool_shape_and_fraud_rate():
    df = generate_pool(n=2000, seed=1)
    assert len(df) == 2000
    for col in ("step", "isFraud", "type", "amount", "account_age_days", "ip_billing_distance_km"):
        assert col in df.columns
    # realistic, low fraud prevalence (not a fraud in every tiny batch)
    assert 0 < df["isFraud"].mean() < 0.05


def test_generate_pool_seed_varies_output():
    a = generate_pool(n=500, seed=1)
    b = generate_pool(n=500, seed=2)
    # different seeds -> different transactions (amounts won't be identical)
    assert not a["amount"].reset_index(drop=True).equals(b["amount"].reset_index(drop=True))


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="fraud_ensemble.joblib not built")
def test_score_stream_adds_model_and_agg_columns():
    from ensemble import load_ensemble
    bundle = load_ensemble(BUNDLE_PATH)
    pool = generate_pool(n=300, seed=3)
    scored = score_stream(pool, bundle)
    assert len(scored) == len(pool)
    assert "agg_decision" in scored.columns
    for key in bundle["models"]:
        assert f"{key}_score" in scored.columns
        assert f"{key}_decision" in scored.columns
    assert set(scored["agg_decision"].unique()) <= {"allow", "review", "block"}


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="fraud_ensemble.joblib not built")
def test_streaming_accumulates_over_ticks():
    # Mirrors the page's per-tick accumulation: stream K rows at a time from a
    # pool and concatenate — the queue grows monotonically.
    import pandas as pd
    from ensemble import load_ensemble
    bundle = load_ensemble(BUNDLE_PATH)
    pool = generate_pool(n=60, seed=7)
    acc, total = None, 0
    for start in range(0, 50, 10):
        scored = score_stream(pool.iloc[start:start + 10], bundle)
        acc = scored if acc is None else pd.concat([acc, scored], ignore_index=True)
        total += len(scored)
        assert len(acc) == total     # grows every tick
    assert total == 50
    assert "agg_decision" in acc.columns
