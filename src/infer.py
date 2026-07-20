"""Minimal inference entrypoint for the saved fraud-detection bundle.

Purpose: show a maintainer exactly how to load `models/fraud_model.joblib`
and score a transaction, so this logic can be ported into a Hugging Face
Space / Inference Endpoint for cloud serving (e.g. a `handler.py` for a
custom HF Inference Endpoint, or a Gradio/FastAPI app on HF Spaces).

Usage:
    python src/infer.py                 # scores the built-in example transaction
    python src/infer.py --json txn.json # scores a transaction from a JSON file

What the bundle contains (see src/train_validate.py):
    model       -> trained XGBoost classifier (feature_group="realistic", no
                   post-transaction balance columns -> safe for authorization time)
    features    -> exact ordered list of column names the model expects
    threshold   -> operating point chosen on validation (min expected cost)
    matrix      -> "tree" -> NaNs are fine, no scaling needed for this bundle

IMPORTANT — destination-history features (`dest_*`, see docs/feature_groups.md):
these are past-only aggregates over `nameDest` (e.g. "how many prior
transactions has this destination account received"). During training they
are computed once over the full sorted history (see
`features.prepare_feature_frame`). A single incoming transaction, scored in
isolation, has no such history available inline — `build_features()` will
default every `dest_*` column to "first time seen" (0 / -1) if they are not
already present on the input row.

For correct serving, a production deployment must maintain destination
history as external state (e.g. a feature store keyed by `nameDest`, updated
after every transaction) and attach the resulting `dest_*` columns to the row
before calling `score_df` — otherwise every transaction is scored as if its
destination account had never been seen before, which understates risk for
known mule accounts. The example transaction below has no history attached
(the common case for a fresh integration); it still scores, just in
degraded/conservative mode.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import joblib
import pandas as pd

SRC = pathlib.Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import MODELS          # noqa: E402
from features import build_features  # noqa: E402

BUNDLE_PATH = MODELS / "fraud_model.joblib"

# One raw transaction, no destination history known (typical for a brand-new
# nameDest, or a caller that has no feature-store integration yet).
EXAMPLE_TRANSACTION = {
    # step/nameOrig/nameDest are required by the destination-history
    # computation (sort + group-by), even though nameOrig/nameDest are
    # dropped before the matrix reaches the model. See score_df docstring.
    "step": 1,
    "nameOrig": "C0000000000",
    "nameDest": "C1234567890",
    "type": "TRANSFER",
    "amount": 181000.00,
    "oldbalanceOrg": 181000.00,
    "oldbalanceDest": 0.0,
    # synthetic risk context — in production this comes from the request
    # (device/session data) or a real-time enrichment service, not PaySim.
    "account_age_days": 12,
    "is_new_device": 1,
    "shipping_billing_mismatch": 1,
    "num_failed_payment_attempts": 2,
    "ip_billing_distance_km": 950.0,
    "is_disposable_email": 0,
    "high_risk_country": 1,
    "hour_of_day": 3,
    "is_night": 1,
    "txn_count_last_24h": 0,
    "time_since_last_hours": -1,
    "account_txn_total": 1,
    "browser": "chrome",
    "device_os": "windows",
    "billing_country": "US",
}


def load_bundle(path: pathlib.Path = BUNDLE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/train_validate.py` first, or "
            "download a pre-built bundle (see docs/data_setup.md)."
        )
    return joblib.load(path)


def score_df(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Score one or more raw transactions with the bundled model.

    `rows` is raw transaction data (same shape as one PaySim row + synthetic
    context columns). Missing feature columns are filled by `build_features`;
    columns the model was not trained on are ignored (`nameOrig`/`nameDest`
    are used internally to compute destination history, then dropped before
    scoring). `step` is required for the internal sort; for single-row/no-
    history serving any constant is fine since ordering is moot with one row.
    """
    rows = rows.copy()
    required_ids = {"step": 1, "nameOrig": "UNKNOWN_ORIG", "nameDest": "UNKNOWN_DEST"}
    for col, default in required_ids.items():
        if col not in rows.columns:
            rows[col] = default
    # newbalance* is post-transaction state — unknown at authorization time,
    # and excluded from the "realistic" feature group this bundle uses. But
    # `add_base_and_context_features` computes errorBalance*/orig_drained
    # from it for EVERY group before the group filter is applied, so the
    # raw column must exist. We approximate "transaction succeeds as
    # requested": orig balance drops by `amount`, dest balance rises by it.
    # This approximation only affects leaky columns already excluded from
    # `realistic`; it does not leak into the model's actual inputs.
    if "newbalanceOrig" not in rows.columns:
        rows["newbalanceOrig"] = (rows["oldbalanceOrg"] - rows["amount"]).clip(lower=0)
    if "newbalanceDest" not in rows.columns:
        rows["newbalanceDest"] = rows["oldbalanceDest"] + rows["amount"]
    features = bundle["features"]
    x = build_features(rows).reindex(columns=features)
    if bundle["matrix"] == "tree":
        x = x.astype("float32")
    scores = bundle["model"].predict_proba(x)[:, 1]

    threshold = float(bundle["threshold"])
    block_threshold = max(0.9, threshold)
    decisions = [
        "block" if s >= block_threshold else "review" if s >= threshold else "allow"
        for s in scores
    ]
    return pd.DataFrame({"fraud_probability": scores, "decision": decisions})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=pathlib.Path, default=None,
                         help="Path to a JSON file: one object or a list of objects.")
    args = parser.parse_args()

    if args.json:
        payload = json.loads(args.json.read_text())
        rows = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    else:
        rows = pd.DataFrame([EXAMPLE_TRANSACTION])

    bundle = load_bundle()
    result = score_df(bundle, rows)

    print(f"[infer] model={bundle['metrics'].get('model_name')} "
          f"feature_group={bundle['feature_group']} threshold={bundle['threshold']:.4f}")
    for i, row in result.iterrows():
        print(f"[infer] row {i}: fraud_probability={row['fraud_probability']:.4f} "
              f"decision={row['decision']}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Porting to Hugging Face for cloud serving
# ---------------------------------------------------------------------------
# The two functions above are the entire contract a HF deployment needs:
#
#   load_bundle()          -> load once at process/worker startup
#   score_df(bundle, rows) -> call per request
#
# HF Inference Endpoints (custom handler.py):
#   class EndpointHandler:
#       def __init__(self, path=""):
#           import sys; sys.path.insert(0, f"{path}/src")  # ship src/ alongside the model
#           from infer import load_bundle
#           self.bundle = load_bundle(f"{path}/models/fraud_model.joblib")
#       def __call__(self, data):
#           import pandas as pd
#           from infer import score_df
#           rows = pd.DataFrame(data["inputs"] if isinstance(data["inputs"], list) else [data["inputs"]])
#           return score_df(self.bundle, rows).to_dict(orient="records")
#
# HF Spaces (Gradio/FastAPI app): same idea — call `load_bundle()` once at
# module import time, then `score_df(bundle, pd.DataFrame([request_json]))`
# inside the request handler.
#
# Whichever route you pick, upload alongside the model weights: this file,
# `src/features.py`, and `src/config.py` — `score_df` imports from all of them.
