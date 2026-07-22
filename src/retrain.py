"""In-app retraining for the live monitor.

Refits the three ensemble models on a freshly-generated, current-distribution
labelled sample — reusing the deployed transformer and each model's
hyperparameters — and returns a new in-memory ensemble bundle. This is a fast
adaptation for the demo (a few seconds), not the full M1-M5 pipeline; the
on-disk fraud_ensemble.joblib is not modified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from infer import enrich
from train_validate import choose_threshold


def retrain_ensemble(base_bundle: dict, sample_df: pd.DataFrame, seed: int = 0,
                     grid: int = 101) -> dict:
    """Return a new ensemble bundle with the models refit on `sample_df`.

    `sample_df` is a labelled (isFraud) transaction sample drawn from the
    current distribution. Uses the base bundle's fitted transformer, clones each
    model to preserve hyperparameters, and re-picks each threshold by validation
    expected cost.
    """
    if "isFraud" not in sample_df:
        raise ValueError("retrain sample must include an 'isFraud' label column")

    transformer = base_bundle["transformer"]
    enriched = enrich(sample_df, use_dest_history=False)
    y = sample_df["isFraud"].to_numpy().astype(int)
    amount = sample_df["amount"].to_numpy(dtype=float)

    n = len(sample_df)
    order = np.random.default_rng(seed).permutation(n)
    cut = max(1, int(n * 0.7))
    tr, va = order[:cut], order[cut:]
    if len(va) == 0:                      # tiny sample: validate on train
        va = tr

    models: dict[str, dict] = {}
    for key, entry in base_bundle["models"].items():
        X = transformer.transform(enriched, entry["matrix"])[entry["features"]].astype(float)
        est = clone(entry["model"])
        est.fit(X.iloc[tr], y[tr])
        val_scores = est.predict_proba(X.iloc[va])[:, 1]
        thr = choose_threshold(y[va], val_scores, amount[va], grid)["threshold"]
        models[key] = {
            "model": est,
            "model_name": entry["model_name"],
            "matrix": entry["matrix"],
            "features": list(entry["features"]),
            "threshold": float(thr),
        }

    return {
        "transformer": transformer,
        "feature_group": base_bundle.get("feature_group", "realistic"),
        "rule": base_bundle.get("rule", "max-risk"),
        "models": models,
        "retrained": True,
        "retrain_rows": int(n),
        "retrain_fraud": int(y.sum()),
    }
