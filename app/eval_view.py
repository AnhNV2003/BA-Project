"""Model evaluation page — how well do the models actually discriminate fraud?

Standard analytics evaluation on the labelled context: a per-model scoreboard
(AUC-PR, ROC-AUC, precision/recall/F1 at each model's deployed threshold), ROC
and precision–recall curves, the ensemble confusion matrix, and a cumulative
gains / lift chart that speaks to the review-capacity question directly:
"if we can only review the top X% of transactions, how much fraud do we catch?"
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from app_common import dataset_badge, get_ensemble, get_scored_context, sample_mode_note

_PALETTE = ["#1565C0", "#2E7D32", "#8E24AA", "#EF6C00", "#C62828"]


def _scoreboard(df, keys, bundle):
    y = df["isFraud"].to_numpy().astype(int)
    rows = []
    series = [(bundle["models"][k]["model_name"], df[f"{k}_score"].to_numpy(),
               float(bundle["models"][k]["threshold"])) for k in keys]
    series.append(("Ensemble (max-risk)", df["risk"].to_numpy(), None))
    for name, score, thr in series:
        if thr is None:
            pred = (df["agg_decision"].to_numpy() != "allow").astype(int)
        else:
            pred = (score >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        rows.append({
            "Model": name,
            "AUC-PR": round(average_precision_score(y, score), 3),
            "ROC-AUC": round(roc_auc_score(y, score), 3),
            "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
            "Flagged": int(pred.sum()),
        })
    return pd.DataFrame(rows)


def _curve_frames(df, keys, bundle):
    y = df["isFraud"].to_numpy().astype(int)
    roc_parts, pr_parts = [], []
    named = [(bundle["models"][k]["model_name"], df[f"{k}_score"].to_numpy()) for k in keys]
    named.append(("Ensemble", df["risk"].to_numpy()))
    for name, score in named:
        fpr, tpr, _ = roc_curve(y, score)
        roc_parts.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "Model": name}))
        prec, rec, _ = precision_recall_curve(y, score)
        pr_parts.append(pd.DataFrame({"recall": rec, "precision": prec, "Model": name}))
    return pd.concat(roc_parts, ignore_index=True), pd.concat(pr_parts, ignore_index=True)


def _gains_frame(df):
    """Cumulative gains: sort by ensemble risk desc, fraction of fraud captured
    within the top-x fraction of reviewed volume."""
    y = df["isFraud"].to_numpy().astype(int)
    order = np.argsort(-df["risk"].to_numpy())
    y_sorted = y[order]
    cum_fraud = np.cumsum(y_sorted) / max(y.sum(), 1)
    frac_reviewed = np.arange(1, len(y) + 1) / len(y)
    # thin to ~200 points for a light chart
    idx = np.linspace(0, len(y) - 1, min(200, len(y))).astype(int)
    model = pd.DataFrame({"reviewed": frac_reviewed[idx], "captured": cum_fraud[idx], "kind": "Model"})
    base = pd.DataFrame({"reviewed": [0, 1], "captured": [0, 1], "kind": "Random"})
    return pd.concat([model, base], ignore_index=True)


def render():
    st.title("📊 Model Evaluation")
    st.caption("How well the models separate fraud from legit, independent of the operating threshold.")
    dataset_badge("test")
    sample_mode_note()

    bundle = get_ensemble()
    df, keys = get_scored_context()
    n_fraud = int(df["isFraud"].sum())
    st.caption(f"Evaluated on the **held-out temporal test split** — {len(df):,} transactions unseen "
               f"at training time · {n_fraud} fraudulent ({n_fraud/len(df):.3%} prevalence, a hard, "
               f"highly-imbalanced problem).")
    st.info("Reported on unseen data (later time steps than training) — not the training set. On the "
            "current **synthetic** stand-in dataset the injected fraud carries strong, cleanly "
            "separable signals, so the tree models score near-perfectly; the linear baseline's lower "
            "AUC-PR shows the problem is non-trivial. Drop the real Kaggle PaySim CSV in `data/raw/` "
            "for production-grade numbers.", icon="ℹ️")

    board = _scoreboard(df, keys, bundle)
    st.subheader("Scoreboard")
    st.dataframe(
        board.style
        .format({"AUC-PR": "{:.3f}", "ROC-AUC": "{:.3f}", "Precision": "{:.3f}",
                 "Recall": "{:.3f}", "F1": "{:.3f}", "Flagged": "{:,}"})
        .background_gradient(subset=["AUC-PR", "ROC-AUC", "F1"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
    st.caption("AUC-PR is the key metric under this much class imbalance; ROC-AUC flatters "
               "imbalanced problems. Per-model precision/recall are at each model's deployed threshold.")

    roc_df, pr_df = _curve_frames(df, keys, bundle)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC curves**")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#B0BEC5").encode(x="x:Q", y="y:Q")
        roc = alt.Chart(roc_df).mark_line().encode(
            x=alt.X("fpr:Q", title="False positive rate"),
            y=alt.Y("tpr:Q", title="True positive rate"),
            color=alt.Color("Model:N", scale=alt.Scale(range=_PALETTE)),
            tooltip=["Model:N", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
        )
        st.altair_chart((diag + roc).properties(height=300), use_container_width=True)
    with c2:
        st.markdown("**Precision–Recall curves**")
        pr = alt.Chart(pr_df).mark_line().encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            color=alt.Color("Model:N", scale=alt.Scale(range=_PALETTE)),
            tooltip=["Model:N", alt.Tooltip("recall:Q", format=".3f"),
                     alt.Tooltip("precision:Q", format=".3f")],
        )
        st.altair_chart(pr.properties(height=300), use_container_width=True)

    st.divider()
    c3, c4 = st.columns([1, 1.3])
    with c3:
        st.markdown("**Ensemble confusion matrix**")
        y = df["isFraud"].to_numpy().astype(int)
        pred = (df["agg_decision"].to_numpy() != "allow").astype(int)
        cm = pd.DataFrame(
            [[int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())],
             [int(((y == 0) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum())]],
            index=["Actual fraud", "Actual legit"], columns=["Flagged", "Allowed"])
        st.dataframe(cm.style.format("{:,}").background_gradient(cmap="Blues"),
                     use_container_width=True)
        tp, fn = cm.loc["Actual fraud"]
        st.caption(f"Catches **{tp}/{tp+fn}** frauds ({tp/(tp+fn):.0%} recall) at the deployed thresholds.")
    with c4:
        st.markdown("**Cumulative gains — fraud captured vs. volume reviewed**")
        gains = _gains_frame(df)
        chart = alt.Chart(gains).mark_line().encode(
            x=alt.X("reviewed:Q", title="Fraction of transactions reviewed", axis=alt.Axis(format="%")),
            y=alt.Y("captured:Q", title="Fraction of fraud caught", axis=alt.Axis(format="%")),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Model", "Random"],
                            range=["#2E7D32", "#B0BEC5"]), title=None),
            strokeDash=alt.StrokeDash("kind:N", legend=None),
        )
        st.altair_chart(chart.properties(height=300), use_container_width=True)
        st.caption("Steeper is better: review the highest-risk transactions first and catch most "
                   "fraud within a small review budget.")
