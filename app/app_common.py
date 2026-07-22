"""Shared setup for the Streamlit app: path wiring + cached resource loaders."""
from __future__ import annotations

import pathlib
import sys

import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "monitoring"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from ensemble import load_ensemble          # noqa: E402
from config import MODELS, DATA_PROCESSED   # noqa: E402
import drift                                 # noqa: E402

REPORTS = ROOT / "monitoring" / "reports"
ENSEMBLE_PATH = MODELS / "fraud_ensemble.joblib"


@st.cache_resource
def get_ensemble() -> dict:
    return load_ensemble(ENSEMBLE_PATH)


@st.cache_data
def get_context_data():
    """Full context frame (for the monitoring temporal split)."""
    return drift.load()


def model_keys(bundle: dict) -> list[str]:
    return list(bundle["models"].keys())


# --------------------------------------------------------------------------- #
# Shared display helpers — one currency formatter and one dataset badge so
# every page speaks the same language (consistency across the demo).
# --------------------------------------------------------------------------- #
CURRENCY = "€"


def fmt_money(x: float) -> str:
    """Compact currency: 1234 -> €1.2k, 1_500_000 -> €1.50M."""
    ax = abs(x)
    if ax >= 1e6:
        return f"{CURRENCY}{x/1e6:,.2f}M"
    if ax >= 1e3:
        return f"{CURRENCY}{x/1e3:,.1f}k"
    return f"{CURRENCY}{x:,.0f}"


def dataset_badge(kind: str) -> None:
    """Render a uniform caption naming the data a page is computed on, so figures
    that differ across pages are never mistaken for inconsistencies.

    kind: 'test'   -> held-out temporal test split (unseen at training)
          'full'   -> full labelled population (descriptive analytics)
          'sample' -> 2k preview batch (interactive triage)
          'live'   -> freshly simulated stream
    """
    import streamlit as st
    notes = {
        "test": "📁 **Data:** held-out temporal test split — transactions unseen at training time.",
        "full": "📁 **Data:** full labelled population — descriptive analytics across all history.",
        "sample": "📁 **Data:** 2,000-transaction preview batch — for interactive triage.",
        "live": "📁 **Data:** freshly simulated real-time stream.",
    }
    st.caption(notes.get(kind, ""))


CONTEXT_PARQUET = DATA_PROCESSED / "transactions_context.parquet"


@st.cache_data
def get_scored_context():
    """Score the labelled **held-out test set** once (cached) for the cost,
    evaluation, and executive-summary pages.

    Honest evaluation: the models were trained on the earlier steps of this
    frame, so we report on the temporal test split only (steps ≥
    `split_info.test_step_min`, unseen at fit time). Enrichment runs on the FULL
    frame first so the causal dest-history features for test rows correctly see
    the prior history, then we filter — matching how the model was trained.

    Returns (df, keys): per-model `*_score` columns, `agg_decision`, a `risk` =
    max-across-models score, plus the original `isFraud` and `amount`.
    """
    import pandas as pd
    from ensemble import score_batch
    from infer import enrich

    bundle = get_ensemble()
    keys = model_keys(bundle)
    ctx = pd.read_parquet(CONTEXT_PARQUET).reset_index(drop=True)
    enriched = enrich(ctx, use_dest_history=True)
    scores = score_batch(enriched, bundle).reset_index(drop=True)
    out = pd.concat([ctx, scores], axis=1)
    out["risk"] = out[[f"{k}_score" for k in keys]].max(axis=1)

    test_min = bundle.get("split_info", {}).get("test_step_min")
    if test_min is not None and "step" in out.columns:
        out = out[out["step"] >= test_min].reset_index(drop=True)
    return out, keys
