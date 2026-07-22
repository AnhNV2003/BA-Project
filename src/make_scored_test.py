"""Precompute the scored held-out test split for the Streamlit app.

The Overview and Cost & ROI pages call `app_common.get_scored_context()`, which
otherwise enriches (~8 min) and scores (~1.5 min) the full ~6.36M-row frame on
first load of every session — a ~10-minute white-screen. This script does that
work ONCE at build time and writes a small `scored_test.parquet` (the test
split only, ready to display), so the app reads it in ~1s.

It replicates the exact logic of `get_scored_context`:
  load full context -> enrich (dest-history on the FULL frame) -> score all
  three models -> max-risk `risk` -> keep only the temporal test split.

Heavy per-row identity strings (names/email/city/device/account ids) are dropped
- the app's KPIs, cost math, and rule-based reason codes never read them - which
keeps the artifact small enough to commit.

Regenerate whenever the model bundle or full frame changes:
    PYTHONPATH=src python src/make_scored_test.py
"""
from __future__ import annotations

import time

import pandas as pd

from config import DATA_PROCESSED, MODELS
from ensemble import load_ensemble, score_batch
from infer import enrich

CONTEXT_PARQUET = DATA_PROCESSED / "transactions_context.parquet"
OUT = DATA_PROCESSED / "scored_test.parquet"

# Columns the Overview/Cost pages and reason codes never read — dropped to keep
# the committed artifact small. High-cardinality identity strings first, then
# fields only the (removed) Segment page used.
DROP_COLS = [
    "customer_id", "customer_name", "email", "billing_city",
    "device_id", "nameOrig", "nameDest",
    "billing_country", "browser", "device_os", "day_index",
    "account_txn_total", "account_txn_index", "time_since_last_hours",
    "txn_count_last_24h", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud",
]


def build() -> None:
    if not CONTEXT_PARQUET.exists():
        raise FileNotFoundError(
            f"{CONTEXT_PARQUET} not found — build the full dataset first "
            "(python src/build_dataset.py --full)."
        )
    started = time.perf_counter()
    bundle = load_ensemble(MODELS / "fraud_ensemble.joblib")
    keys = list(bundle["models"].keys())

    ctx = pd.read_parquet(CONTEXT_PARQUET).reset_index(drop=True)
    enriched = enrich(ctx, use_dest_history=True)          # full frame -> correct dest history
    scores = score_batch(enriched, bundle).reset_index(drop=True)
    out = pd.concat([ctx, scores], axis=1)
    out["risk"] = out[[f"{k}_score" for k in keys]].max(axis=1)

    test_min = bundle.get("split_info", {}).get("test_step_min")
    if test_min is not None and "step" in out.columns:
        out = out[out["step"] >= test_min].reset_index(drop=True)

    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns])
    out.to_parquet(OUT, index=False)

    size_mb = OUT.stat().st_size / 1e6
    elapsed = time.perf_counter() - started
    print("=== SCORED TEST SPLIT ===")
    print(f"rows       : {len(out):,}")
    print(f"fraud      : {int(out['isFraud'].sum()):,}  ({out['isFraud'].mean():.4%})")
    print(f"columns    : {out.shape[1]}  ({', '.join(out.columns)})")
    print(f"saved      : {OUT.relative_to(OUT.parents[2])}  ({size_mb:.1f} MB)")
    print(f"elapsed    : {elapsed:.1f}s")


if __name__ == "__main__":
    build()
