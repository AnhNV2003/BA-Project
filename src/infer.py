"""Inference script for PaySim fraud detection.

Loads the saved bundle (fraud_model.joblib) and scores new transactions.

RECOMMENDED MODE
----------------
  Batch file  →  use WITH dest-history (default, omit --no-dest-history).
                 The full transaction history is available, so causal
                 nameDest aggregations are computed exactly as in training.
  Single record →  --no-dest-history is applied automatically.
                   Dest-history features are set to 0 (first-seen defaults).

Usage examples
--------------
# Test on the held-out step range stored inside the bundle:
    python src/infer.py --input data/processed/transactions_clean.parquet --test-only

# Quick sanity check — score only N rows:
    python src/infer.py --input data/processed/transactions_clean.parquet --nrows 50000

# Random sample fraction:
    python src/infer.py --input data/processed/transactions_clean.parquet --sample-frac 0.05

# Single JSON record (dest-history auto-disabled):
    python src/infer.py --record '{"step":1,"type":"TRANSFER","amount":9000,...}'

# Write scored output:
    python src/infer.py --input file.parquet --output scored.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make src/ importable when called from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS
from features import (
    DEST_HISTORY_NUMERIC,
    LINEAR_GROUP,
    TREE_GROUP,
    TYPE_ONEHOT_COLS,
    add_base_and_context_features,
    add_destination_history,
    add_fixed_onehot,
    prepare_feature_frame,
)


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def load_bundle(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    required = {"model", "transformer", "feature_group", "matrix", "features", "threshold"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Bundle is missing keys: {missing}")
    return bundle


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

PAYSIM_DTYPES = {
    "step": "int32",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
}

SYNTH_DEFAULTS: dict[str, object] = {
    "account_age_days": 365,
    "is_new_device": 0,
    "shipping_billing_mismatch": 0,
    "num_failed_payment_attempts": 0,
    "ip_billing_distance_km": 0.0,
    "is_disposable_email": 0,
    "high_risk_country": 0,
    "hour_of_day": 12,
    "is_night": 0,
    "txn_count_last_24h": 1,
    "time_since_last_hours": -1.0,
    "account_txn_total": 1,
    "browser": "unknown",
    "device_os": "unknown",
    "billing_country": "unknown",
}


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast core PaySim columns and fill synthetic feature defaults."""
    out = df.copy()
    for col, dtype in PAYSIM_DTYPES.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    for col, default in SYNTH_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    return out


def read_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in {".csv", ".tsv"}:
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}. Use .parquet or .csv")
    return _ensure_required_columns(df)


def apply_slice(df: pd.DataFrame, args: argparse.Namespace, bundle: dict) -> pd.DataFrame:
    """Optionally restrict rows via --test-only / --nrows / --sample-frac."""
    if args.test_only:
        split = bundle.get("split_info", {})
        lo = split.get("test_step_min")
        hi = split.get("test_step_max")
        if lo is None or hi is None:
            raise ValueError("Bundle has no split_info.test_step_min/max — cannot use --test-only.")
        mask = (df["step"] >= lo) & (df["step"] <= hi)
        df = df[mask].reset_index(drop=True)
        print(f"[infer] --test-only: step [{lo}, {hi}] → {len(df):,} rows", file=sys.stderr)
    if args.nrows is not None:
        df = df.iloc[: args.nrows].reset_index(drop=True)
        print(f"[infer] --nrows: keeping first {len(df):,} rows", file=sys.stderr)
    if args.sample_frac is not None:
        df = df.sample(frac=args.sample_frac, random_state=42).sort_values("step").reset_index(drop=True)
        print(f"[infer] --sample-frac {args.sample_frac}: {len(df):,} rows", file=sys.stderr)
    return df


def record_from_json(raw: str) -> pd.DataFrame:
    obj = json.loads(raw)
    if isinstance(obj, dict):
        obj = [obj]
    df = pd.DataFrame(obj)
    return _ensure_required_columns(df)


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, use_dest_history: bool = True) -> pd.DataFrame:
    """Apply the same feature engineering as training."""
    if use_dest_history:
        x = prepare_feature_frame(df)
    else:
        x = add_base_and_context_features(add_fixed_onehot(df.copy()))
        for col in DEST_HISTORY_NUMERIC:
            x[col] = 0
        x["log_ip_distance"] = np.log1p(x.get("ip_billing_distance_km", 0))
    return x


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(df: pd.DataFrame, bundle: dict, use_dest_history: bool = True) -> pd.DataFrame:
    """Return df with columns: fraud_score, is_fraud_predicted."""
    t0 = time.perf_counter()
    print(f"[infer] enriching {len(df):,} rows (dest_history={use_dest_history}) …", file=sys.stderr)
    enriched = enrich(df, use_dest_history=use_dest_history)
    t1 = time.perf_counter()
    print(f"[infer] enrichment done  ({t1 - t0:.1f}s)", file=sys.stderr)

    transformer = bundle["transformer"]
    matrix_group = bundle["matrix"]
    # The deployed model was fit on the selected feature group's subset
    # (bundle["features"]), but transform() returns the full tree/linear matrix.
    # Select the trained columns so serving matches fit exactly.
    X = transformer.transform(enriched, matrix_group)[bundle["features"]]
    t2 = time.perf_counter()
    print(f"[infer] transform done   ({t2 - t1:.1f}s)  shape={X.shape}", file=sys.stderr)

    model = bundle["model"]
    threshold = float(bundle["threshold"])
    proba = model.predict_proba(X)[:, 1]
    predicted = (proba >= threshold).astype(int)
    t3 = time.perf_counter()
    print(f"[infer] inference done   ({t3 - t2:.1f}s)  total={t3 - t0:.1f}s  "
          f"throughput={len(df) / (t3 - t0):,.0f} rows/s", file=sys.stderr)

    out = df.copy().reset_index(drop=True)
    out["fraud_score"] = proba.round(6)
    out["is_fraud_predicted"] = predicted
    return out


def print_eval_metrics(scored: pd.DataFrame, bundle: dict) -> None:
    """Print classification metrics when ground-truth labels are present."""
    if "isFraud" not in scored.columns:
        return
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, COST_MANUAL_REVIEW

    y_true = scored["isFraud"].to_numpy()
    y_pred = scored["is_fraud_predicted"].to_numpy()
    scores = scored["fraud_score"].to_numpy()
    amount = scored["amount"].to_numpy(dtype=float) if "amount" in scored.columns else np.ones(len(scored))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc_pr = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    flagged = y_pred == 1
    cost = (
        COST_FALSE_NEGATIVE * amount[fn_mask].sum()
        + COST_FALSE_POSITIVE * fp_mask.sum()
        + COST_MANUAL_REVIEW * flagged.sum()
    )
    fraud_amount = amount[y_true == 1].sum()
    caught_amount = amount[(y_true == 1) & (y_pred == 1)].sum()
    loss_avoided_pct = caught_amount / fraud_amount * 100 if fraud_amount else 0.0

    n_fraud = int(y_true.sum())
    n_flagged = int(flagged.sum())
    sep = "-" * 50
    print(sep, file=sys.stderr)
    print(f"[eval] rows          : {len(scored):,}", file=sys.stderr)
    print(f"[eval] actual fraud  : {n_fraud:,}  ({n_fraud / len(scored):.4%})", file=sys.stderr)
    print(f"[eval] flagged       : {n_flagged:,}  ({n_flagged / len(scored):.4%})", file=sys.stderr)
    print(f"[eval] threshold     : {bundle['threshold']:.4f}", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(f"[eval] precision     : {precision:.4f}", file=sys.stderr)
    print(f"[eval] recall        : {recall:.4f}", file=sys.stderr)
    print(f"[eval] F1            : {f1:.4f}", file=sys.stderr)
    print(f"[eval] AUC-PR        : {auc_pr:.4f}", file=sys.stderr)
    print(f"[eval] ROC-AUC       : {roc_auc:.4f}", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(f"[eval] expected cost : {cost:,.2f}", file=sys.stderr)
    print(f"[eval] loss avoided  : {loss_avoided_pct:.2f}%", file=sys.stderr)
    print(sep, file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score transactions with the trained fraud bundle.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="FILE", help="Parquet or CSV file of transactions.")
    src.add_argument("--record", metavar="JSON", help="Single transaction as a JSON object.")
    parser.add_argument("--model", default=str(MODELS / "fraud_model.joblib"), metavar="PATH",
                        help="Path to the .joblib bundle (default: models/fraud_model.joblib).")
    parser.add_argument("--output", metavar="FILE", help="Write scored CSV to this path (default: stdout).")
    parser.add_argument("--no-dest-history", action="store_true",
                        help="Skip causal dest-history enrichment (use zeros). "
                             "Suitable for single-record scoring without prior context.")
    # Slicing flags
    parser.add_argument("--test-only", action="store_true",
                        help="Filter to the held-out test step range stored in the bundle.")
    parser.add_argument("--nrows", type=int, metavar="N",
                        help="Keep only the first N rows (after any --test-only filter).")
    parser.add_argument("--sample-frac", type=float, metavar="F",
                        help="Random sample fraction, e.g. 0.05 for 5%% of rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(Path(args.model))

    split = bundle.get("split_info", {})
    print(
        f"[infer] bundle: model={bundle.get('model_key', type(bundle['model']).__name__)} "
        f"group={bundle['feature_group']} threshold={bundle['threshold']:.4f} "
        f"features={len(bundle['features'])}",
        file=sys.stderr,
    )
    if split:
        print(
            f"[infer] train steps [{split.get('train_step_min')}, {split.get('train_step_max')}]  "
            f"val [{split.get('val_step_min')}, {split.get('val_step_max')}]  "
            f"test [{split.get('test_step_min')}, {split.get('test_step_max')}]",
            file=sys.stderr,
        )

    if args.input:
        df = read_input(args.input)
        print(f"[infer] loaded {len(df):,} rows from {args.input}", file=sys.stderr)
        df = apply_slice(df, args, bundle)
    else:
        df = record_from_json(args.record)

    use_dest_history = not args.no_dest_history
    if use_dest_history and args.record:
        # Single-record JSON: dest history is meaningless; default to zeros.
        use_dest_history = False
        print("[infer] single record — dest-history disabled automatically", file=sys.stderr)

    scored = score(df, bundle, use_dest_history=use_dest_history)
    print_eval_metrics(scored, bundle)

    output_cols = [c for c in scored.columns if c not in {"isFlaggedFraud"}]
    if args.output:
        scored[output_cols].to_csv(args.output, index=False)
        print(f"[infer] scored output -> {args.output}", file=sys.stderr)
    else:
        scored[output_cols].to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
