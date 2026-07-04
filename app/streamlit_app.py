"""Fraud-analyst review queue (Module 6, draft) — Streamlit demo.

Loads the trained model, scores a batch of transactions, and presents them as a
prioritised review queue — mirroring what a risk analyst would triage.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import pathlib
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build_features            # noqa: E402
from config import MODELS, DATA_PROCESSED      # noqa: E402

st.set_page_config(page_title="Fraud Review Queue", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "fraud_model.joblib")


@st.cache_data
def load_and_score():
    bundle = load_model()
    p = DATA_PROCESSED / "sample_preview.csv"
    df = pd.read_csv(p)
    X = build_features(df)[bundle["features"]].astype(float)
    df["risk_score"] = bundle["model"].predict_proba(X)[:, 1]
    thr = bundle["threshold"]
    block = max(0.9, thr)
    df["decision"] = pd.cut(df["risk_score"], [-1, thr, block, 2],
                            labels=["allow", "review", "block"])
    return df, thr, block


bundle = load_model()
df, thr, block = load_and_score()

st.title("🛡️ Fraud Review Queue")
st.caption(f"Model: **{bundle['model_name']}** · review ≥ {thr:.3f} · block ≥ {block:.3f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("To review", int((df.decision == "review").sum()))
c3.metric("Auto-block", int((df.decision == "block").sum()))
c4.metric("Actual fraud", int(df.isFraud.sum()))

st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum risk score", 0.0, 1.0, float(thr), 0.01)
types = st.sidebar.multiselect("Transaction type", sorted(df["type"].unique()),
                               default=["TRANSFER", "CASH_OUT"])
only_flagged = st.sidebar.checkbox("Only flagged (review/block)", value=True)

view = df[df["risk_score"] >= min_score]
if types:
    view = view[view["type"].isin(types)]
if only_flagged:
    view = view[view["decision"].isin(["review", "block"])]
view = view.sort_values("risk_score", ascending=False)

cols = ["risk_score", "decision", "type", "amount", "account_age_days",
        "is_new_device", "shipping_billing_mismatch", "num_failed_payment_attempts",
        "ip_billing_distance_km", "high_risk_country", "hour_of_day", "isFraud"]
st.subheader(f"Queue — {len(view):,} transactions")
st.dataframe(
    view[cols].style.format({"risk_score": "{:.3f}", "amount": "{:,.0f}",
                             "ip_billing_distance_km": "{:,.0f}"})
    .background_gradient(subset=["risk_score"], cmap="Reds"),
    use_container_width=True, height=520,
)
st.caption("`isFraud` shown here only for demo evaluation; unavailable at real "
           "scoring time. Sort/triage by `risk_score`.")
