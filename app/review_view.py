"""Review Queue page — score the batch with all 3 models, triage by max-risk."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app_common import DATA_PROCESSED, dataset_badge, get_ensemble, model_keys
from ensemble import score_batch
from infer import enrich
from reasons import reason_series

_DECISION_COLORS = {"allow": "#e8f5e9", "review": "#fff8e1", "block": "#ffebee"}


@st.cache_data
def _load_and_score():
    bundle = get_ensemble()
    keys = model_keys(bundle)
    df = pd.read_csv(DATA_PROCESSED / "sample_preview.csv")
    # Batch with full history → dest-history enabled.
    enriched = enrich(df, use_dest_history=True)
    scores = score_batch(enriched, bundle).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    df["max_score"] = df[[f"{k}_score" for k in keys]].max(axis=1)
    # Transparent "why flagged" signals — computed on the enriched frame so
    # engineered features (ip distance, night, account age) are available.
    df["why_flagged"] = reason_series(enriched.reset_index(drop=True))
    return df, keys


def _color_decision(val):
    return f"background-color: {_DECISION_COLORS.get(val, '')}"


def render():
    bundle = get_ensemble()
    df, keys = _load_and_score()

    st.title("🛡️ Fraud Review Queue")
    st.caption(
        "Models: **" + "**, **".join(bundle["models"][k]["model_name"] for k in keys)
        + "** · decision = **max-risk** aggregate (block > review > allow)"
    )
    dataset_badge("sample")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("To review", int((df.agg_decision == "review").sum()))
    c3.metric("Auto-block", int((df.agg_decision == "block").sum()))
    c4.metric("Actual fraud", int(df.isFraud.sum()))

    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Min. model risk score", 0.0, 1.0, 0.10, 0.01)
    types = st.sidebar.multiselect(
        "Transaction type", sorted(df["type"].unique()), default=["TRANSFER", "CASH_OUT"]
    )
    only_flagged = st.sidebar.checkbox("Only flagged (review/block)", value=True)

    view = df[df["max_score"] >= min_score]
    if types:
        view = view[view["type"].isin(types)]
    if only_flagged:
        view = view[view["agg_decision"].isin(["review", "block"])]
    # sort by aggregate severity, then peak model score
    sev = {"allow": 0, "review": 1, "block": 2}
    view = view.assign(_sev=view["agg_decision"].map(sev)).sort_values(
        ["_sev", "max_score"], ascending=False
    )

    score_cols = [f"{k}_score" for k in keys]
    cols = ["agg_decision", "why_flagged", *score_cols, "type", "amount", "account_age_days",
            "is_new_device", "shipping_billing_mismatch", "num_failed_payment_attempts",
            "ip_billing_distance_km", "high_risk_country", "hour_of_day", "isFraud"]
    cols = [c for c in cols if c in view.columns]

    st.subheader(f"Queue — {len(view):,} transactions")
    styler = (
        view[cols].style
        .format({**{c: "{:.3f}" for c in score_cols},
                 "amount": "{:,.0f}", "ip_billing_distance_km": "{:,.0f}"})
        .map(_color_decision, subset=["agg_decision"])
        .background_gradient(subset=score_cols, cmap="Reds", vmin=0, vmax=1)
    )
    st.dataframe(
        styler, use_container_width=True, height=520,
        column_config={"why_flagged": st.column_config.TextColumn(
            "why flagged", width="large",
            help="Transparent risk signals present on the transaction (rule-based reason codes).")},
    )
    st.caption("`why flagged` lists the concrete risk signals an analyst would act on; per-model "
               "scores are shown side by side and `agg_decision` is the max-risk verdict. "
               "`isFraud` is for demo evaluation only — unavailable at scoring time.")
