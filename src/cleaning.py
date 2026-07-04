"""Data cleaning (Module 3, draft).

Runs conservative, DOCUMENTED cleaning on the processed dataset and writes a
before/after report to docs/cleaning_report.md plus a cleaned parquet.

Philosophy: PaySim is already tidy, so cleaning here is mostly *validation* +
documenting known quirks. We do NOT "fix" the balance-error signature (that is
fraud signal, not dirt).

Run:  python cleaning.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DOCS, PAYSIM_COLUMNS


def load() -> pd.DataFrame:
    p = DATA_PROCESSED / "transactions_context.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def main():
    df = load()
    before = len(df)
    log = ["# Cleaning Report — E-Commerce Fraud Detection\n",
           f"Input rows: **{before:,}**, columns: **{df.shape[1]}**\n"]
    decisions = []

    # 1) Missing values -------------------------------------------------------
    miss = df.isna().sum()
    miss = miss[miss > 0]
    log.append("## 1. Missing values\n")
    if miss.empty:
        log.append("No missing values in any column. ✅\n")
    else:
        log.append(miss.to_frame("n_missing").to_markdown() + "\n")

    # 2) Exact duplicate transactions (on the base PaySim fields) -------------
    dup_mask = df.duplicated(subset=PAYSIM_COLUMNS, keep="first")
    n_dup = int(dup_mask.sum())
    log.append("## 2. Duplicate base transactions\n")
    log.append(f"Exact duplicates on PaySim fields: **{n_dup:,}**. "
               f"{'Dropped.' if n_dup else 'None.'}\n")
    if n_dup:
        df = df.loc[~dup_mask].copy()
        decisions.append(f"Dropped {n_dup:,} duplicate base transactions.")

    # 3) Invalid amounts ------------------------------------------------------
    n_neg = int((df["amount"] < 0).sum())
    n_zero = int((df["amount"] == 0).sum())
    log.append("## 3. Invalid / zero amounts\n")
    log.append(f"- Negative amounts: **{n_neg:,}** (removed)\n"
               f"- Zero amounts: **{n_zero:,}** (kept — legal in PaySim, but flagged)\n")
    if n_neg:
        df = df.loc[df["amount"] >= 0].copy()
        decisions.append(f"Removed {n_neg:,} rows with negative amount.")
    df["flag_zero_amount"] = (df["amount"] == 0).astype(int)

    # 4) Amount outliers (document, do NOT clip — large txns are meaningful) --
    q999 = df["amount"].quantile(0.999)
    n_out = int((df["amount"] > q999).sum())
    log.append("## 4. Amount outliers\n")
    log.append(f"99.9th percentile = {q999:,.0f}. Rows above it: **{n_out:,}** "
               f"({n_out/len(df):.3%}). **Kept** — extreme amounts are informative "
               f"for fraud; we add a capped feature instead of dropping.\n")
    df["amount_capped"] = df["amount"].clip(upper=q999)

    # 5) Known PaySim balance quirks (validate, don't fix) -------------------
    dest_zero = int(((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)).sum())
    err_orig = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    n_err = int((err_orig.abs() > 1e-6).sum())
    log.append("## 5. Known PaySim balance quirks (documented, NOT modified)\n")
    log.append(f"- Destination balances 0 before & after despite amount>0: "
               f"**{dest_zero:,}** (merchant/mule accounts — expected).\n")
    log.append(f"- Rows where oldbalanceOrg − amount ≠ newbalanceOrig: "
               f"**{n_err:,}** — this **balance-error is fraud signal**, kept as a feature.\n")

    # 6) Range validation on synthetic fields --------------------------------
    checks = {
        "hour_of_day ∈ 0..23": df["hour_of_day"].between(0, 23).all(),
        "account_age_days ≥ 1": (df["account_age_days"] >= 1).all(),
        "num_failed_payment_attempts ≥ 0": (df["num_failed_payment_attempts"] >= 0).all(),
        "ip_billing_distance_km ≥ 0": (df["ip_billing_distance_km"] >= 0).all(),
    }
    log.append("## 6. Range validation (synthetic fields)\n")
    for k, v in checks.items():
        log.append(f"- {k}: {'OK ✅' if v else 'FAIL ❌'}\n")

    # ---- save + summary ----
    after = len(df)
    log.append("## Summary\n")
    log.append(f"- Rows before: **{before:,}** → after: **{after:,}** "
               f"(removed {before-after:,}, {(before-after)/before:.3%}).\n")
    log.append("- Decisions:\n" + ("".join(f"  - {d}\n" for d in decisions) if decisions
               else "  - No rows removed; dataset already clean. Added flags: "
                    "`flag_zero_amount`, `amount_capped`.\n"))

    out = DATA_PROCESSED / "transactions_clean.parquet"
    try:
        df.to_parquet(out, index=False)
    except Exception:
        out = out.with_suffix(".csv"); df.to_csv(out, index=False)
    (DOCS / "cleaning_report.md").write_text("\n".join(log), encoding="utf-8")

    print(f"[clean] {before:,} -> {after:,} rows")
    print(f"[clean] duplicates={n_dup:,} neg_amount={n_neg:,} zero_amount={n_zero:,} outliers>{q999:,.0f}={n_out:,}")
    print(f"[clean] balance-error rows (kept as signal): {n_err:,}")
    print(f"[clean] wrote cleaned data -> {out.name} ; report -> docs/cleaning_report.md")


if __name__ == "__main__":
    main()
