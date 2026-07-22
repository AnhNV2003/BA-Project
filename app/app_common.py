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


def sample_mode_note() -> None:
    """Warn (once per page) when analytics are computed on the small committed
    sample because the full context frame is absent — so approximate figures on
    a fresh clone are never mistaken for production numbers."""
    import streamlit as st
    if context_is_sample():
        st.caption(
            "⚠️ **Sample mode** — computed on the committed ~300k-row stratified "
            "sample (the full frame is absent). Figures are approximate; rebuild "
            "`transactions_context.parquet` (see docs/data_setup.md) for full numbers."
        )


CONTEXT_PARQUET = DATA_PROCESSED / "transactions_context.parquet"
# Small, committed fallback (~28MB) so a fresh clone runs the whole app without
# rebuilding the ~526MB full frame. Stratified -> preserves the real fraud rate.
CONTEXT_SAMPLE = DATA_PROCESSED / "context_sample.parquet"


def resolve_context_path():
    """Full context frame if present, else the committed sample. Returns
    (path, is_sample). Raises if neither exists."""
    if CONTEXT_PARQUET.exists():
        return CONTEXT_PARQUET, False
    if CONTEXT_SAMPLE.exists():
        return CONTEXT_SAMPLE, True
    raise FileNotFoundError(
        "Neither transactions_context.parquet nor context_sample.parquet found "
        "in data/processed/. Rebuild the dataset (see docs/data_setup.md)."
    )


def context_is_sample() -> bool:
    """True when the app is falling back to the small committed sample (the full
    frame is absent) — pages use this to badge themselves as 'sample mode'."""
    return not CONTEXT_PARQUET.exists() and CONTEXT_SAMPLE.exists()


@st.cache_data
def get_scored_context():
    """Score the labelled **held-out test set** once (cached) for the cost,
    evaluation, and executive-summary pages.

    Honest evaluation: the models were trained on the earlier steps of this
    frame, so we report on the temporal test split only (steps ≥
    `split_info.test_step_min`, unseen at fit time). Enrichment runs on the FULL
    frame first so the causal dest-history features for test rows correctly see
    the prior history, then we filter — matching how the model was trained.

    Falls back to the small committed sample when the full frame is absent (see
    `context_is_sample()`); numbers are then approximate but structurally real.

    Returns (df, keys): per-model `*_score` columns, `agg_decision`, a `risk` =
    max-across-models score, plus the original `isFraud` and `amount`.
    """
    import pandas as pd
    from ensemble import score_batch
    from infer import enrich

    bundle = get_ensemble()
    keys = model_keys(bundle)
    path, _ = resolve_context_path()
    ctx = pd.read_parquet(path).reset_index(drop=True)
    enriched = enrich(ctx, use_dest_history=True)
    scores = score_batch(enriched, bundle).reset_index(drop=True)
    out = pd.concat([ctx, scores], axis=1)
    out["risk"] = out[[f"{k}_score" for k in keys]].max(axis=1)

    test_min = bundle.get("split_info", {}).get("test_step_min")
    if test_min is not None and "step" in out.columns:
        out = out[out["step"] >= test_min].reset_index(drop=True)
    return out, keys
