"""Shared multi-model scoring — one source of truth for API and Streamlit.

The deployable ensemble bundle (`models/fraud_ensemble.joblib`) holds all three
models trained on the `realistic` feature group, a shared FeatureTransformer, and
each model's own cost-tuned decision threshold and feature matrix. This module
turns that bundle into per-model probabilities and decisions, and combines them
with the `max-risk` rule.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Decision severity ordering — higher means more restrictive.
SEVERITY = {"allow": 0, "review": 1, "block": 2}
_BY_SEVERITY = {v: k for k, v in SEVERITY.items()}
AGGREGATE_RULE = "max-risk"
DEFAULT_BLOCK_FLOOR = 0.9

REQUIRED_MODEL_KEYS = {"model", "model_name", "matrix", "features", "threshold"}


def decide(prob: float, review_thr: float, block_floor: float = DEFAULT_BLOCK_FLOOR) -> str:
    """Map a fraud probability to allow / review / block.

    block if prob >= max(block_floor, review_thr); review if prob >= review_thr;
    otherwise allow. The block gate never sits below the review threshold.
    """
    block_thr = max(block_floor, review_thr)
    if prob >= block_thr:
        return "block"
    if prob >= review_thr:
        return "review"
    return "allow"


def block_threshold(review_thr: float, block_floor: float = DEFAULT_BLOCK_FLOOR) -> float:
    return max(block_floor, review_thr)


def aggregate_maxrisk(decisions) -> str:
    """Return the most severe decision. Ignores None/unknown entries (a model
    that errored contributes nothing). Empty / all-invalid -> 'allow'."""
    severities = [SEVERITY[d] for d in decisions if d in SEVERITY]
    if not severities:
        return "allow"
    return _BY_SEVERITY[max(severities)]


def load_ensemble(path: str | Path) -> dict:
    """Load and validate the ensemble bundle."""
    bundle = joblib.load(path)
    if "models" not in bundle or not isinstance(bundle["models"], dict) or not bundle["models"]:
        raise ValueError(f"{path} is not a valid ensemble bundle (missing non-empty 'models').")
    if "transformer" not in bundle:
        raise ValueError(f"{path} ensemble bundle missing 'transformer'.")
    for key, entry in bundle["models"].items():
        missing = REQUIRED_MODEL_KEYS - set(entry)
        if missing:
            raise ValueError(f"Ensemble model {key!r} missing keys: {sorted(missing)}")
    return bundle


def score_all(enriched_df: pd.DataFrame, bundle: dict) -> dict[str, np.ndarray]:
    """Score an already-enriched frame with every model in the bundle.

    The caller performs enrichment once (API: single record with dest-history
    disabled; Streamlit: batch with dest-history enabled). Each model applies its
    own matrix transform and trained feature subset, so serving matches training.
    Returns {model_key: probability array}.
    """
    transformer = bundle["transformer"]
    out: dict[str, np.ndarray] = {}
    for key, entry in bundle["models"].items():
        X = transformer.transform(enriched_df, entry["matrix"])[entry["features"]].astype(float)
        out[key] = entry["model"].predict_proba(X)[:, 1]
    return out


def score_batch(enriched_df: pd.DataFrame, bundle: dict, block_floor: float = DEFAULT_BLOCK_FLOOR) -> pd.DataFrame:
    """Score a batch with every model; return per-model score + decision columns
    and a row-wise max-risk `agg_decision`. Used by the Streamlit review queue.
    """
    probs = score_all(enriched_df, bundle)
    out = pd.DataFrame(index=enriched_df.index)
    max_sev = np.zeros(len(enriched_df), dtype=int)
    for key, entry in bundle["models"].items():
        review_thr = float(entry["threshold"])
        p = probs[key]
        out[f"{key}_score"] = p
        decisions = np.array([decide(float(v), review_thr, block_floor) for v in p])
        out[f"{key}_decision"] = decisions
        sev = np.array([SEVERITY[d] for d in decisions])
        max_sev = np.maximum(max_sev, sev)
    out["agg_decision"] = [_BY_SEVERITY[v] for v in max_sev]
    return out


def score_record(enriched_df: pd.DataFrame, bundle: dict, block_floor: float = DEFAULT_BLOCK_FLOOR) -> dict:
    """Score a single enriched record and return the API payload.

    Per-model failures are isolated: a model that raises contributes an
    {"error": ...} entry and is excluded from the aggregate (degraded mode).
    """
    transformer = bundle["transformer"]
    models: dict[str, dict] = {}
    decisions: list[str] = []
    degraded = False
    for key, entry in bundle["models"].items():
        review_thr = float(entry["threshold"])
        try:
            X = transformer.transform(enriched_df, entry["matrix"])[entry["features"]].astype(float)
            prob = float(entry["model"].predict_proba(X)[:, 1][0])
            d = decide(prob, review_thr, block_floor)
            models[key] = {
                "model_name": entry["model_name"],
                "fraud_probability": round(prob, 4),
                "decision": d,
                "review_threshold": round(review_thr, 4),
                "block_threshold": round(block_threshold(review_thr, block_floor), 4),
            }
            decisions.append(d)
        except Exception as exc:  # isolate one bad model from the rest
            degraded = True
            models[key] = {"model_name": entry.get("model_name", key), "error": str(exc)}

    payload = {
        "models": models,
        "aggregate": {"decision": aggregate_maxrisk(decisions), "rule": bundle.get("rule", AGGREGATE_RULE)},
    }
    if degraded:
        payload["aggregate"]["degraded"] = True
    return payload
