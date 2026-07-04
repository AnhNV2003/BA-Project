"""Exploratory Data Analysis (Module 2, draft).

Loads the processed dataset (base + synthetic), writes figures to docs/figures/,
and a human-readable summary to docs/eda_summary.md.

Run:  python eda.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                       # headless — save PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_PROCESSED, FIGURES, DOCS

sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C9F70", 1: "#D1495B"}       # legit / fraud


def load() -> pd.DataFrame:
    p = DATA_PROCESSED / "transactions_context.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def savefig(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    df = load()
    lines = ["# EDA Summary — E-Commerce Fraud Detection\n"]
    n, nf = len(df), int(df["isFraud"].sum())
    lines.append(f"- Rows: **{n:,}** | Fraud: **{nf:,}** (**{nf/n:.4%}**) — severe imbalance.\n")

    # 1) Class imbalance ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["isFraud"].value_counts().sort_index()
    ax.bar(["legit (0)", "fraud (1)"], vc.values, color=[PALETTE[0], PALETTE[1]])
    ax.set_yscale("log"); ax.set_ylabel("count (log)"); ax.set_title("Class imbalance")
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    savefig(fig, "01_class_imbalance.png")

    # 2) Fraud rate by transaction type --------------------------------------
    fr_type = df.groupby("type")["isFraud"].agg(["mean", "sum", "count"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(fr_type.index, fr_type["mean"] * 100, color="#3E7CB1")
    ax.set_ylabel("fraud rate (%)"); ax.set_title("Fraud rate by transaction type")
    plt.xticks(rotation=30)
    savefig(fig, "02_fraud_by_type.png")
    lines.append("## Fraud by transaction type\n")
    lines.append("Fraud occurs **only in TRANSFER & CASH_OUT** (a structural PaySim fact).\n")
    lines.append(fr_type.assign(mean=lambda d: (d["mean"] * 100).round(3)).to_markdown() + "\n")

    # 3) Amount distribution by class ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in (0, 1):
        sub = df.loc[df["isFraud"] == cls, "amount"]
        ax.hist(np.log1p(sub), bins=60, alpha=0.55, label=f"class {cls}",
                color=PALETTE[cls], density=True)
    ax.set_xlabel("log(1 + amount)"); ax.set_ylabel("density")
    ax.set_title("Transaction amount by class"); ax.legend()
    savefig(fig, "03_amount_by_class.png")

    # 4) Fraud rate by hour of day -------------------------------------------
    hr = df.groupby("hour_of_day")["isFraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hr.index, hr.values, marker="o", color="#D1495B")
    ax.set_xlabel("hour of day"); ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate by hour of day (synthetic time pattern)")
    savefig(fig, "04_fraud_by_hour.png")

    # 5) Binary risk signals: fraud rate when flag=1 vs 0 ---------------------
    flags = ["is_new_device", "shipping_billing_mismatch", "is_disposable_email",
             "high_risk_country", "is_night"]
    rows = []
    for f in flags:
        g = df.groupby(f)["isFraud"].mean() * 100
        rows.append({"signal": f, "rate_if_0(%)": round(g.get(0, np.nan), 3),
                     "rate_if_1(%)": round(g.get(1, np.nan), 3)})
    sig = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(flags)); w = 0.38
    ax.bar(x - w/2, sig["rate_if_0(%)"], w, label="flag = 0", color="#A9C6D6")
    ax.bar(x + w/2, sig["rate_if_1(%)"], w, label="flag = 1", color="#D1495B")
    ax.set_xticks(x); ax.set_xticklabels(flags, rotation=25, ha="right")
    ax.set_ylabel("fraud rate (%)"); ax.set_title("Fraud rate by synthetic risk flag"); ax.legend()
    savefig(fig, "05_risk_flags.png")
    lines.append("## Synthetic risk flags (fraud rate when flag on vs off)\n")
    lines.append(sig.to_markdown(index=False) + "\n")

    # 6) Numeric signals by class (box) --------------------------------------
    nums = ["account_age_days", "ip_billing_distance_km", "num_failed_payment_attempts"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes, nums):
        data = [df.loc[df.isFraud == 0, c], df.loc[df.isFraud == 1, c]]
        ax.boxplot(data, tick_labels=["legit", "fraud"], showfliers=False)
        ax.set_title(c)
    savefig(fig, "06_numeric_by_class.png")

    # 7) Correlation with target ---------------------------------------------
    num_cols = df.select_dtypes("number").columns
    corr = df[num_cols].corr(numeric_only=True)["isFraud"].drop("isFraud").sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    top = corr.head(15)[::-1]
    ax.barh(top.index, top.values, color=np.where(top.values >= 0, "#D1495B", "#3E7CB1"))
    ax.set_title("Correlation with isFraud (top 15 |r|)")
    savefig(fig, "07_corr_with_target.png")
    lines.append("## Top correlations with isFraud\n")
    lines.append(corr.head(10).round(3).to_frame("pearson_r").to_markdown() + "\n")

    (DOCS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[eda] wrote 7 figures -> {FIGURES}")
    print(f"[eda] wrote summary   -> {DOCS / 'eda_summary.md'}")
    print("\n--- key facts ---")
    print(f"rows={n:,} fraud={nf:,} ({nf/n:.4%})")
    print(fr_type.assign(mean=lambda d: (d['mean']*100).round(3)))


if __name__ == "__main__":
    main()
