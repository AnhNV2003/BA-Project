"""Live-feed transaction simulator.

Generates fresh synthetic transactions (PaySim base + synthetic risk context)
and scores them with the deployed ensemble. Used by the Streamlit Live Feed page
to emulate an incoming transaction stream. Pure functions only — no Streamlit — so
the generation and scoring logic is unit-testable on its own.
"""
from __future__ import annotations

import pandas as pd

from data_base import make_standin
from synth_context import add_synthetic_context
from infer import enrich
from ensemble import score_batch

# Fresh transactions are generated a pool at a time (realistic ~0.15% fraud
# rate); the page streams K rows per tick and regenerates a new pool — with a
# new seed — when the current one is exhausted. Per-tick generation is avoided
# because make_standin forces >=1 fraud per call, which would inflate the rate.
DEFAULT_POOL_SIZE = 2000


def generate_pool(n: int = DEFAULT_POOL_SIZE, seed: int = 0, verbose: bool = False) -> pd.DataFrame:
    """A fresh pool of n synthetic transactions with full risk context."""
    base = make_standin(n=n, seed=seed)
    return add_synthetic_context(base, seed=seed, verbose=verbose)


def decision_timeline(scored_df: pd.DataFrame, bin_size: int = 25) -> pd.DataFrame:
    """Count review/block transactions per arrival bucket for the timeline chart.

    Expects columns `arrival` (1-based sequence) and `agg_decision`. Returns a
    DataFrame indexed by bucket-start arrival with columns ['review', 'block']
    (allow is excluded — the chart shows only flagged transactions).
    """
    cols = ["review", "block"]
    if scored_df is None or len(scored_df) == 0:
        return pd.DataFrame(columns=cols)
    d = scored_df[["arrival", "agg_decision"]].copy()
    d["bucket"] = ((d["arrival"] - 1) // bin_size) * bin_size + 1
    flagged = d[d["agg_decision"].isin(cols)]
    counts = flagged.groupby(["bucket", "agg_decision"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=cols, fill_value=0)
    # keep every bucket on the axis, even all-allow ones, so time spacing is real
    full_index = range(1, int(d["bucket"].max()) + bin_size, bin_size)
    return counts.reindex(full_index, fill_value=0)


def score_stream(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score raw transactions as streaming records (dest-history disabled) and
    return the raw columns joined with per-model scores + max-risk agg_decision.
    """
    enriched = enrich(raw_df, use_dest_history=False)
    scores = score_batch(enriched, bundle).reset_index(drop=True)
    return pd.concat([raw_df.reset_index(drop=True), scores], axis=1)
