"""Executive summary — the 5-second value story.

A KPI header a professor (or a CFO) can read at a glance: volume, fraud
exposure, money saved by the model, detection performance, and live model
health (drift). Everything ties back to the detail pages.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import drift
from app_common import dataset_badge, fmt_money as _money, get_ensemble, get_scored_context, model_keys, sample_mode_note
import costs
from reasons import reason_series

_REPORT_MD = drift.REPORTS / "drift_report.md"


def render():
    st.title("E-Commerce Real-Time Payment Fraud Detection")
    st.caption("A multi-model decision system that scores every payment in real time, prices the "
               "fraud/friction trade-off in €, explains each flag, and monitors itself for drift.")
    dataset_badge("test")
    sample_mode_note()

    bundle = get_ensemble()
    df, keys = get_scored_context()
    y = df["isFraud"].to_numpy()
    amount = df["amount"].to_numpy(dtype=float)
    flagged = df["agg_decision"].to_numpy() != "allow"
    res = costs.cost_breakdown(y, flagged, amount)

    # ---- Headline KPIs ----------------------------------------------------- #
    k = st.columns(4)
    k[0].metric("Transactions scored", f"{res.n:,}")
    k[1].metric("Fraud exposure", _money(res.fraud_exposure),
                delta=f"{res.n_fraud} cases", delta_color="off")
    k[2].metric("💶 Money saved", _money(res.net_savings),
                help="Net € saved vs. running no model, at the deployed thresholds.")
    k[3].metric("🛡️ Fraud loss avoided", f"{res.loss_avoided_pct:.0f}%")

    k2 = st.columns(4)
    k2[0].metric("Recall (fraud caught)", f"{res.recall:.0%}")
    k2[1].metric("Precision", f"{res.precision:.0%}")
    k2[2].metric("Flag rate", f"{res.n_flagged/res.n:.1%}",
                 help="Share of transactions sent to review or blocked.")

    # model health from the latest drift report, if present
    health, note = _model_health()
    k2[3].metric("Model health", health, delta=note, delta_color="off")

    st.divider()

    # ---- Decision mix + top fraud signals ---------------------------------- #
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Decision mix**")
        mix = (df["agg_decision"].value_counts()
               .reindex(["allow", "review", "block"]).fillna(0).astype(int)
               .rename_axis("decision").reset_index(name="count"))
        chart = alt.Chart(mix).mark_bar().encode(
            x=alt.X("count:Q", title=None, axis=alt.Axis(format="~s")),
            y=alt.Y("decision:N", sort=["block", "review", "allow"], title=None),
            color=alt.Color("decision:N", scale=alt.Scale(
                domain=["allow", "review", "block"],
                range=["#2E7D32", "#F6A445", "#C62828"]), legend=None),
            tooltip=["decision:N", "count:Q"],
        ).properties(height=140)
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.markdown("**Top risk signals on flagged transactions**")
        flagged_df = df[flagged]
        reasons = reason_series(flagged_df, top=1)  # single top reason per txn
        top = (reasons[reasons != "—"].value_counts().head(6)
               .rename_axis("signal").reset_index(name="count"))
        if len(top):
            chart = alt.Chart(top).mark_bar(color="#1565C0").encode(
                x=alt.X("count:Q", title=None),
                y=alt.Y("signal:N", sort="-x", title=None),
                tooltip=["signal:N", "count:Q"],
            ).properties(height=180)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No flagged transactions in this sample.")

    st.divider()
    st.markdown("**The story, in order** (follow the sidebar top-to-bottom):")
    st.markdown(
        "1. 🔎 **Segment Analytics** — where fraud concentrates (the problem)\n"
        "2. 🛡️ **Review Queue** — triage transactions with reason codes (the product)\n"
        "3. 📊 **Model Evaluation** — how well the models detect, on unseen data (the rigor)\n"
        "4. 💰 **Cost & ROI** — what detection is worth in € vs. naive baselines (the value)\n"
        "5. 📡 **Live Feed** — real-time scoring stream\n"
        "6. 📈 **Monitoring** — drift detection & automated retraining (operations)\n"
        "7. 🧪 **API Tester** — score a transaction through the live API (serving)"
    )


def _model_health():
    """Read the latest drift report to summarise health, if available."""
    if not _REPORT_MD.exists():
        return "✅ Healthy", "no drift report yet"
    text = _REPORT_MD.read_text(encoding="utf-8")
    if "SIGNIFICANT" in text:
        return "🔴 Drift", "retrain triggered"
    if "moderate" in text.lower():
        return "🟡 Watch", "moderate drift"
    return "✅ Healthy", "within thresholds"
