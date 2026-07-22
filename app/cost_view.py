"""Cost & ROI page — reframe fraud detection as a money decision.

Fraud detection is a cost-optimization problem: every decision trades the cost
of a missed fraud (≈ the transaction amount) against the cost of a false alarm
(customer friction + review labour). This page puts € on the board:

  * headline ROI of the deployed model vs. two naive baselines, and
  * an editable cost matrix for live sensitivity analysis.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_common import (CURRENCY as CUR, dataset_badge, fmt_money as _money,
                        get_ensemble, get_scored_context)
import costs

ENSEMBLE_LABEL = "Ensemble (max-risk)"


def _confusion_df(r: costs.CostResult) -> pd.DataFrame:
    return pd.DataFrame(
        [[r.tp, r.fn], [r.fp, r.tn]],
        index=["Actual fraud", "Actual legit"],
        columns=["Flagged", "Allowed"],
    )


def render():
    st.title("💰 Cost & ROI")
    st.caption("Fraud detection is a **cost-optimization** problem. Every threshold trades "
               "missed-fraud loss against false-alarm friction — this page prices that trade in €.")
    dataset_badge("test")

    df, keys = get_scored_context()
    y = df["isFraud"].to_numpy()
    amount = df["amount"].to_numpy(dtype=float)

    # ---- Editable cost matrix (defaults come from config.py) ---------------- #
    with st.expander("⚙️ Cost assumptions (edit for sensitivity analysis)", expanded=False):
        st.caption("Defaults are the team's assumptions in `config.py`. Change them to see how "
                   "the business case shifts — this is the sensitivity analysis your report needs.")
        cc = st.columns(3)
        c_fn = cc[0].number_input(
            f"Missed-fraud cost  (× amount)", value=costs.DEFAULT_FN, step=0.1, min_value=0.0,
            help="A missed fraud loses ≈ the transaction amount. Weight per € of amount lost.")
        c_fp = cc[1].number_input(
            f"False-alarm cost  ({CUR} each)", value=costs.DEFAULT_FP, step=1.0, min_value=0.0,
            help="Blocking a good customer: lost sale + friction + possible churn.")
        c_review = cc[2].number_input(
            f"Manual-review cost  ({CUR} each)", value=costs.DEFAULT_REVIEW, step=0.5, min_value=0.0,
            help="Analyst labour to work one flagged transaction.")

    kw = dict(c_fn=c_fn, c_fp=c_fp, c_review=c_review)

    # ---- Model chooser: ensemble or any single model ----------------------- #
    bundle = get_ensemble()
    name_by_key = {k: bundle["models"][k]["model_name"] for k in keys}
    options = [ENSEMBLE_LABEL] + [name_by_key[k] for k in keys]

    def _flagged_for(choice: str):
        """Flagged mask (review or block) for the ensemble or one named model,
        each using its own trained decision threshold."""
        if choice == ENSEMBLE_LABEL:
            return df["agg_decision"].to_numpy() != "allow"
        key = next(k for k, n in name_by_key.items() if n == choice)
        return df[f"{key}_decision"].to_numpy() != "allow"

    choice = st.selectbox("Score the business impact of:", options, index=0,
                          help="Compare each model's economics against the deployed max-risk ensemble.")
    sel = costs.cost_breakdown(y, _flagged_for(choice), amount, **kw)

    label = "ensemble" if choice == ENSEMBLE_LABEL else choice
    st.subheader(f"{choice} — business impact")
    st.caption(f"On {sel.n:,} transactions carrying {_money(sel.fraud_exposure)} of fraud exposure "
               f"({sel.n_fraud} fraudulent).")
    k = st.columns(4)
    k[0].metric("💶 Net savings vs. no model", _money(sel.net_savings),
                help="Money saved versus letting every transaction through (the do-nothing baseline).")
    k[1].metric("🛡️ Fraud loss avoided", f"{sel.loss_avoided_pct:.0f}%",
                delta=_money(sel.caught_amount), help="Share of € fraud exposure the model prevents.")
    k[2].metric("💸 Fraud still missed", _money(sel.missed_amount),
                delta=f"-{sel.fn} cases", delta_color="inverse")
    k[3].metric("📈 ROI", f"{sel.roi:.1f}×",
                help="Net savings per € spent on false alarms + manual review.")

    # ---- Baseline comparison bar (selected model) -------------------------- #
    st.markdown(f"**Total cost vs. naive strategies** ({label}) — lower is better")
    comp = pd.DataFrame({
        "Strategy": ["Do nothing\n(allow all)", "Review everything\n(no ML)", f"{label}"],
        "Total cost": [sel.do_nothing_cost, sel.review_all_cost, sel.model_cost],
        "kind": ["baseline", "baseline", "model"],
    })
    bar = (
        alt.Chart(comp).mark_bar().encode(
            x=alt.X("Total cost:Q", title="Total cost (€)", axis=alt.Axis(format="~s")),
            y=alt.Y("Strategy:N", sort=None, title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["baseline", "model"],
                            range=["#B0BEC5", "#2E7D32"]), legend=None),
            tooltip=[alt.Tooltip("Total cost:Q", format=",.0f")],
        ).properties(height=140)
    )
    st.altair_chart(bar, use_container_width=True)

    with st.expander(f"Confusion matrix & full cost breakdown ({label})"):
        cm = _confusion_df(sel)
        st.dataframe(cm.style.format("{:,}"), use_container_width=True)
        breakdown = pd.DataFrame({
            "Component": ["Missed-fraud loss (FN)", "False-alarm cost (FP)", "Manual-review labour",
                          "— Total model cost", "Do-nothing baseline", "Net savings"],
            "Amount": [-sel.fn_loss, -sel.fp_cost, -sel.review_cost,
                       -sel.model_cost, -sel.do_nothing_cost, sel.net_savings],
        })
        st.dataframe(
            breakdown.style.format({"Amount": lambda v: _money(v)}),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ---- All models vs. ensemble side by side ------------------------------ #
    st.subheader("All models vs. ensemble")
    st.caption("Every model priced on the same cost assumptions. The max-risk **ensemble maximizes "
               "recall** (it flags if *any* model flags) and is robust when one model degrades — but "
               "that union also inherits every model's false alarms, so its precision (and sometimes "
               "net savings) can trail the best single model on clean data. Read the trade-off, don't "
               "assume a winner.")
    rows = []
    for opt in options:
        r = costs.cost_breakdown(y, _flagged_for(opt), amount, **kw)
        rows.append({
            "Model": opt, "Net savings": r.net_savings, "ROI": r.roi,
            "Loss avoided %": r.loss_avoided_pct, "Recall": r.recall, "Precision": r.precision,
            "Flagged": r.n_flagged, "is_ensemble": opt == ENSEMBLE_LABEL,
        })
    cmp_df = pd.DataFrame(rows)

    chart = (
        alt.Chart(cmp_df).mark_bar().encode(
            x=alt.X("Net savings:Q", title="Net savings (€)", axis=alt.Axis(format="~s")),
            y=alt.Y("Model:N", sort="-x", title=None),
            color=alt.Color("is_ensemble:N", scale=alt.Scale(domain=[True, False],
                            range=["#2E7D32", "#90A4AE"]), legend=None),
            tooltip=["Model:N", alt.Tooltip("Net savings:Q", format=",.0f"),
                     alt.Tooltip("ROI:Q", format=".1f"), alt.Tooltip("Recall:Q", format=".0%"),
                     alt.Tooltip("Precision:Q", format=".0%")],
        ).properties(height=40 * len(cmp_df) + 30)
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        cmp_df.drop(columns="is_ensemble").style
        .format({"Net savings": lambda v: _money(v), "ROI": "{:.1f}×",
                 "Loss avoided %": "{:.0f}%", "Recall": "{:.0%}", "Precision": "{:.0%}",
                 "Flagged": "{:,}"})
        .background_gradient(subset=["Net savings"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
