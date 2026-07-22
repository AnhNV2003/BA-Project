"""Segment analytics — where does fraud concentrate?

Descriptive business intelligence on the full labelled population: fraud rate
and € exposure by transaction type, hour of day, and amount band, plus a
risk-factor "lift" table quantifying how much each signal (new device,
high-risk country, disposable email, shipping≠billing) multiplies fraud odds.
This is the "know your enemy" view that motivates the model's features.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import context_is_sample, dataset_badge, fmt_money as _money, resolve_context_path


@st.cache_data
def _load():
    path, _ = resolve_context_path()
    return pd.read_parquet(path)


def _rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col).agg(
        transactions=("isFraud", "size"),
        fraud=("isFraud", "sum"),
        fraud_amount=("amount", lambda s: s[df.loc[s.index, "isFraud"] == 1].sum()),
    ).reset_index()
    g["fraud_rate"] = g["fraud"] / g["transactions"]
    return g


def _lift_table(df: pd.DataFrame, flags: dict[str, str]) -> pd.DataFrame:
    base = df["isFraud"].mean()
    rows = []
    for col, label in flags.items():
        if col not in df.columns:
            continue
        on = df[df[col] == 1]
        off = df[df[col] == 0]
        if len(on) == 0:
            continue
        rate_on = on["isFraud"].mean()
        rate_off = off["isFraud"].mean() if len(off) else 0.0
        rows.append({
            "Risk factor": label,
            "Fraud rate (present)": rate_on,
            "Fraud rate (absent)": rate_off,
            "Lift vs. baseline": rate_on / base if base else np.nan,
            "% of transactions": len(on) / len(df),
        })
    out = pd.DataFrame(rows).sort_values("Lift vs. baseline", ascending=False)
    return out


def render():
    st.title("🔎 Segment Analytics")
    st.caption("Where fraud concentrates across the business — the descriptive picture behind the "
               "model's features.")
    dataset_badge("sample" if context_is_sample() else "full")

    df = _load()
    base_rate = df["isFraud"].mean()
    total_fraud_amount = df.loc[df["isFraud"] == 1, "amount"].sum()

    k = st.columns(4)
    k[0].metric("Transactions", f"{len(df):,}")
    k[1].metric("Fraud cases", f"{int(df['isFraud'].sum()):,}")
    k[2].metric("Baseline fraud rate", f"{base_rate:.3%}")
    k[3].metric("Fraud € exposure", _money(total_fraud_amount))

    st.divider()

    # ---- By transaction type ---------------------------------------------- #
    st.subheader("By transaction type")
    st.caption("In PaySim, fraud occurs only in TRANSFER and CASH_OUT — the model exploits this.")
    by_type = _rate_by(df, "type")
    c1, c2 = st.columns(2)
    with c1:
        chart = alt.Chart(by_type).mark_bar().encode(
            x=alt.X("fraud_rate:Q", title="Fraud rate", axis=alt.Axis(format="%")),
            y=alt.Y("type:N", sort="-x", title=None),
            color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["type:N", alt.Tooltip("fraud_rate:Q", format=".3%"),
                     "fraud:Q", alt.Tooltip("fraud_amount:Q", format=",.0f")],
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)
    with c2:
        chart = alt.Chart(by_type).mark_bar(color="#C62828").encode(
            x=alt.X("fraud_amount:Q", title="Fraud € exposure", axis=alt.Axis(format="~s")),
            y=alt.Y("type:N", sort="-x", title=None),
            tooltip=["type:N", alt.Tooltip("fraud_amount:Q", format=",.0f")],
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)

    st.divider()

    # ---- Risk-factor lift -------------------------------------------------- #
    st.subheader("Risk-factor lift")
    st.caption("How much more likely is fraud when each signal is present, vs. the baseline rate? "
               "Lift > 1 means the factor concentrates fraud.")
    lift = _lift_table(df, {
        "is_new_device": "New device",
        "high_risk_country": "High-risk country",
        "is_disposable_email": "Disposable email",
        "shipping_billing_mismatch": "Shipping ≠ billing",
        "is_night": "Night-time",
    })
    lc1, lc2 = st.columns([1.2, 1])
    with lc1:
        st.dataframe(
            lift.style.format({
                "Fraud rate (present)": "{:.3%}", "Fraud rate (absent)": "{:.3%}",
                "Lift vs. baseline": "{:.1f}×", "% of transactions": "{:.1%}",
            }).background_gradient(subset=["Lift vs. baseline"], cmap="Oranges"),
            use_container_width=True, hide_index=True,
        )
    with lc2:
        chart = alt.Chart(lift).mark_bar(color="#EF6C00").encode(
            x=alt.X("Lift vs. baseline:Q", title="Lift (×baseline)"),
            y=alt.Y("Risk factor:N", sort="-x", title=None),
            tooltip=["Risk factor:N", alt.Tooltip("Lift vs. baseline:Q", format=".1f")],
        ).properties(height=220)
        st.altair_chart(chart, use_container_width=True)

    st.divider()

    # ---- By hour of day + amount band ------------------------------------- #
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("By hour of day")
        if "hour_of_day" in df.columns:
            by_hour = _rate_by(df, "hour_of_day")
            base_line = alt.Chart(pd.DataFrame({"y": [base_rate]})).mark_rule(
                color="#B0BEC5", strokeDash=[4, 4]).encode(y="y:Q")
            bars = alt.Chart(by_hour).mark_bar(color="#1565C0").encode(
                x=alt.X("hour_of_day:O", title="Hour"),
                y=alt.Y("fraud_rate:Q", title="Fraud rate", axis=alt.Axis(format="%")),
                tooltip=["hour_of_day:O", alt.Tooltip("fraud_rate:Q", format=".3%"), "fraud:Q"],
            )
            st.altair_chart((bars + base_line).properties(height=240), use_container_width=True)
            st.caption("Dashed line = baseline fraud rate.")
        else:
            st.info("No `hour_of_day` column in the dataset.")
    with c4:
        st.subheader("By amount band")
        bands = [0, 200, 1000, 5000, 20000, 100000, np.inf]
        labels = ["<200", "200–1k", "1k–5k", "5k–20k", "20k–100k", "100k+"]
        tmp = df.assign(band=pd.cut(df["amount"], bins=bands, labels=labels, right=False))
        by_band = _rate_by(tmp, "band")
        chart = alt.Chart(by_band).mark_bar().encode(
            x=alt.X("band:N", sort=labels, title="Amount (€)"),
            y=alt.Y("fraud_rate:Q", title="Fraud rate", axis=alt.Axis(format="%")),
            color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["band:N", alt.Tooltip("fraud_rate:Q", format=".3%"), "fraud:Q",
                     alt.Tooltip("fraud_amount:Q", format=",.0f")],
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Fraud skews toward higher-value transactions — where the € loss is largest.")
