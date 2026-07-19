"""Exploratory Data Analysis for PaySim (Module 2)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, DOCS, FIGURES, PAYSIM_CSV

sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C9F70", 1: "#D1495B"}
OUT_DIR = FIGURES / "paysim"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_processed() -> pd.DataFrame:
    path = DATA_PROCESSED / "transactions_context.parquet"
    if not path.exists():
        path = path.with_suffix(".csv")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_raw_full() -> pd.DataFrame:
    cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
    ]
    if PAYSIM_CSV.exists():
        return pd.read_csv(PAYSIM_CSV, usecols=cols)
    processed = load_processed()
    return processed[cols].copy()


def savefig(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def rate_table(df: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    data = df
    if top is not None:
        keep = df[col].value_counts(dropna=False).head(top).index
        data = df[df[col].isin(keep)]
    out = data.groupby(col, dropna=False)["isFraud"].agg(["sum", "count", "mean"])
    out = out.rename(columns={"sum": "fraud", "mean": "fraud_rate"})
    out["fraud_rate_pct"] = (out["fraud_rate"] * 100).round(4)
    return out.sort_values("fraud_rate", ascending=False)


def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "hour_of_day" not in x:
        x["hour_of_day"] = (x["step"] % 24).astype("int8")
    if "day_index" not in x:
        x["day_index"] = (x["step"] // 24).astype("int16")
    x["log_amount"] = np.log1p(x["amount"])
    x["errorBalanceOrig"] = x["oldbalanceOrg"] - x["amount"] - x["newbalanceOrig"]
    x["errorBalanceDest"] = x["oldbalanceDest"] + x["amount"] - x["newbalanceDest"]
    x["abs_errorBalanceOrig"] = x["errorBalanceOrig"].abs()
    x["abs_errorBalanceDest"] = x["errorBalanceDest"].abs()
    x["orig_drained"] = ((x["newbalanceOrig"] == 0) & (x["oldbalanceOrg"] > 0)).astype("int8")
    x["dest_was_empty"] = (x["oldbalanceDest"] == 0).astype("int8")
    x["dest_is_customer"] = x["nameDest"].astype("string").str.startswith("C").astype("int8")
    return x


def safe_auc(y: pd.Series, values: pd.Series) -> float:
    try:
        auc = roc_auc_score(y, values)
        return float(max(auc, 1 - auc))
    except ValueError:
        return float("nan")


def main() -> None:
    df = add_eda_features(load_processed())
    raw = add_eda_features(load_raw_full())
    n = len(raw)
    fraud = int(raw["isFraud"].sum())
    fraud_rate = fraud / n
    fraud_amount = float(raw.loc[raw["isFraud"] == 1, "amount"].sum())
    sample_n = len(df)
    sample_fraud = int(df["isFraud"].sum())

    lines = [
        "# EDA Summary — PaySim E-Commerce Fraud Detection\n",
        "## Dataset Snapshot\n",
        f"- Rows: **{n:,}**.\n",
        f"- Fraud rows: **{fraud:,}** (**{fraud_rate:.4%}**).\n",
        f"- Total fraudulent amount: **{fraud_amount:,.2f}**.\n",
        f"- Step range: **{int(raw['step'].min())}..{int(raw['step'].max())}** hours.\n",
        f"- Synthetic/context sections use processed sample: **{sample_n:,}** rows, **{sample_fraud:,}** fraud rows.\n",
        "- Raw-sensitive sections (`type`, `amount`, balance, `nameDest` reuse) use the full PaySim CSV to avoid sampling-compressed reuse counts.\n",
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    vc = raw["isFraud"].value_counts().sort_index()
    ax.bar(["legit (0)", "fraud (1)"], vc.values, color=[PALETTE[0], PALETTE[1]])
    ax.set_yscale("log")
    ax.set_ylabel("count (log)")
    ax.set_title("Class imbalance")
    for i, value in enumerate(vc.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    savefig(fig, "01_class_imbalance.png")

    fr_type = rate_table(raw, "type")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fr_type.index.astype(str), fr_type["fraud_rate_pct"], color="#3E7CB1")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate by transaction type")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, "02_fraud_by_type.png")
    lines.extend([
        "\n## Fraud by Transaction Type\n",
        "Fraud appears only in `TRANSFER` and `CASH_OUT` in PaySim.\n",
        fr_type[["fraud", "count", "fraud_rate_pct"]].to_markdown() + "\n",
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in (0, 1):
        sub = raw.loc[raw["isFraud"] == cls, "log_amount"]
        ax.hist(sub, bins=70, alpha=0.55, label=f"class {cls}", color=PALETTE[cls], density=True)
    ax.set_xlabel("log1p(amount)")
    ax.set_ylabel("density")
    ax.set_title("Transaction amount by class")
    ax.legend()
    savefig(fig, "03_amount_by_class.png")

    raw["amount_decile"] = pd.qcut(raw["amount"], q=10, duplicates="drop")
    amount_bins = raw.groupby("amount_decile", observed=True)["isFraud"].agg(["sum", "count", "mean"])
    amount_bins["fraud_rate_pct"] = (amount_bins["mean"] * 100).round(4)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(amount_bins)), amount_bins["fraud_rate_pct"], marker="o", color="#D1495B")
    ax.set_xticks(range(len(amount_bins)))
    ax.set_xticklabels([str(i) for i in amount_bins.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud likelihood by amount decile")
    savefig(fig, "04_amount_decile_fraud_rate.png")
    amount_q = raw["amount"].quantile([0.5, 0.9, 0.99, 0.999]).round(2)
    lines.extend([
        "\n## Amount Distribution / Outliers\n",
        amount_q.to_frame("amount").to_markdown() + "\n",
        f"- Max amount: **{raw['amount'].max():,.2f}**.\n",
        f"- Zero-amount rows in full raw data: **{int((raw['amount'] == 0).sum()):,}**.\n",
        amount_bins[["sum", "count", "fraud_rate_pct"]].to_markdown() + "\n",
    ])

    by_hour = rate_table(raw, "hour_of_day").sort_index()
    by_day = rate_table(raw, "day_index").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(by_hour.index, by_hour["fraud_rate_pct"], marker="o", color="#D1495B")
    axes[0].set_title("Fraud rate by hour of day")
    axes[0].set_xlabel("hour")
    axes[0].set_ylabel("fraud rate (%)")
    axes[1].plot(by_day.index, by_day["fraud_rate_pct"], marker="o", color="#3E7CB1")
    axes[1].set_title("Fraud rate by simulation day")
    axes[1].set_xlabel("day_index")
    savefig(fig, "05_time_windows.png")
    last_day = by_day.tail(1)
    lines.extend([
        "\n## Time Windows\n",
        "- `hour_of_day` and `day_index` are derived from PaySim `step` using the same M1 formula.\n",
        "- Late simulation days can have tiny volume, so extreme daily fraud rates should be treated as a simulation quirk.\n",
        last_day[["fraud", "count", "fraud_rate_pct"]].to_markdown() + "\n",
    ])

    flags = [
        "is_new_device", "shipping_billing_mismatch", "is_disposable_email",
        "high_risk_country", "is_night",
    ]
    flag_rows = []
    for flag in flags:
        g = df.groupby(flag)["isFraud"].mean() * 100
        flag_rows.append({
            "signal": flag,
            "rate_if_0_pct": round(float(g.get(0, np.nan)), 4),
            "rate_if_1_pct": round(float(g.get(1, np.nan)), 4),
        })
    flag_df = pd.DataFrame(flag_rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(flag_df))
    width = 0.38
    ax.bar(x - width / 2, flag_df["rate_if_0_pct"], width, label="flag = 0", color="#A9C6D6")
    ax.bar(x + width / 2, flag_df["rate_if_1_pct"], width, label="flag = 1", color="#D1495B")
    ax.set_xticks(x)
    ax.set_xticklabels(flag_df["signal"], rotation=25, ha="right")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate by synthetic risk flag")
    ax.legend()
    savefig(fig, "06_synthetic_risk_flags.png")
    lines.extend(["\n## Synthetic Risk Flags\n", flag_df.to_markdown(index=False) + "\n"])

    nums = ["account_age_days", "ip_billing_distance_km", "num_failed_payment_attempts"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, nums):
        ax.boxplot(
            [df.loc[df.isFraud == 0, col], df.loc[df.isFraud == 1, col]],
            tick_labels=["legit", "fraud"],
            showfliers=False,
        )
        ax.set_title(col)
    savefig(fig, "07_numeric_context_by_class.png")

    balance_cols = ["abs_errorBalanceOrig", "abs_errorBalanceDest", "orig_drained", "dest_was_empty"]
    balance_auc = pd.DataFrame({
        "feature": balance_cols,
        "directionless_auc": [round(safe_auc(raw["isFraud"], raw[col]), 4) for col in balance_cols],
    }).sort_values("directionless_auc", ascending=False)
    balance_rate = pd.DataFrame({
        "signal": ["orig_drained", "dest_was_empty"],
        "legit_rate_pct": [
            round(float(raw.loc[raw.isFraud == 0, "orig_drained"].mean() * 100), 4),
            round(float(raw.loc[raw.isFraud == 0, "dest_was_empty"].mean() * 100), 4),
        ],
        "fraud_rate_pct": [
            round(float(raw.loc[raw.isFraud == 1, "orig_drained"].mean() * 100), 4),
            round(float(raw.loc[raw.isFraud == 1, "dest_was_empty"].mean() * 100), 4),
        ],
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cls in (0, 1):
        sample = raw.loc[raw.isFraud == cls, "abs_errorBalanceOrig"].clip(upper=raw["abs_errorBalanceOrig"].quantile(0.99))
        axes[0].hist(np.log1p(sample), bins=60, alpha=0.55, label=f"class {cls}", color=PALETTE[cls], density=True)
    axes[0].set_title("Origin balance error by class")
    axes[0].set_xlabel("log1p(abs error)")
    axes[0].legend()
    axes[1].bar(balance_rate["signal"], balance_rate["legit_rate_pct"], width=0.4, label="legit", color=PALETTE[0])
    axes[1].bar(
        np.arange(len(balance_rate)) + 0.4,
        balance_rate["fraud_rate_pct"],
        width=0.4,
        label="fraud",
        color=PALETTE[1],
    )
    axes[1].set_xticks(np.arange(len(balance_rate)) + 0.2)
    axes[1].set_xticklabels(balance_rate["signal"])
    axes[1].set_ylabel("rate (%)")
    axes[1].set_title("Balance-derived binary signals")
    axes[1].legend()
    savefig(fig, "08_balance_signature.png")
    lines.extend([
        "\n## Balance Signature\n",
        "PaySim balance reconciliation is very predictive, so later modelling should also test a realistic feature set without post-transaction balance leakage-like signals.\n",
        balance_auc.to_markdown(index=False) + "\n",
        balance_rate.to_markdown(index=False) + "\n",
    ])

    dest = raw.groupby("nameDest").agg(
        txn_count=("isFraud", "size"),
        fraud_count=("isFraud", "sum"),
        sender_count=("nameOrig", "nunique"),
        amount_sum=("amount", "sum"),
        dest_is_customer=("dest_is_customer", "max"),
    )
    dest["has_fraud"] = dest["fraud_count"] > 0
    dest["reuse_bucket"] = pd.cut(
        dest["txn_count"],
        bins=[0, 1, 3, 10, 1000],
        labels=["1", "2-3", "4-10", "11+"],
        include_lowest=True,
    )
    mule = dest.groupby("reuse_bucket", observed=True).agg(
        destinations=("txn_count", "size"),
        total_txns=("txn_count", "sum"),
        fraud_destinations=("has_fraud", "sum"),
        fraud_txns=("fraud_count", "sum"),
        avg_senders=("sender_count", "mean"),
    )
    mule["fraud_dest_rate_pct"] = (mule["fraud_destinations"] / mule["destinations"] * 100).round(4)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.log1p(dest["txn_count"]), bins=60, color="#3E7CB1")
    axes[0].set_title("Destination account reuse")
    axes[0].set_xlabel("log1p(txn_count per nameDest)")
    axes[1].bar(mule.index.astype(str), mule["fraud_dest_rate_pct"], color="#D1495B")
    axes[1].set_title("Fraud-destination rate by reuse bucket")
    axes[1].set_xlabel("transactions per destination")
    axes[1].set_ylabel("fraud destinations (%)")
    savefig(fig, "09_destination_reuse_pattern.png")
    dest_type = raw.groupby("dest_is_customer")["isFraud"].agg(["sum", "count", "mean"])
    dest_type["fraud_rate_pct"] = (dest_type["mean"] * 100).round(4)
    lines.extend([
        "\n## Destination Reuse / Mule Pattern\n",
        "- This section is computed on the full raw PaySim file. A 15% row sample compresses cross-row reuse and undercounts max/repeated `nameDest` values.\n",
        f"- Unique destination accounts: **{len(dest):,}**.\n",
        f"- Destination accounts with at least 2 transactions: **{int((dest['txn_count'] >= 2).sum()):,}**.\n",
        f"- Destination accounts with at least 4 transactions: **{int((dest['txn_count'] >= 4).sum()):,}**.\n",
        f"- Max transactions for one destination: **{int(dest['txn_count'].max()):,}**.\n",
        "- This supports M4 historical aggregation on `nameDest` using past transactions only.\n",
        mule.to_markdown() + "\n",
        dest_type.to_markdown() + "\n",
    ])

    context_tables = {
        "browser": rate_table(df, "browser", top=10),
        "device_os": rate_table(df, "device_os"),
        "billing_country": rate_table(df, "billing_country"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, table) in zip(axes, context_tables.items()):
        top = table.head(8)
        ax.bar(top.index.astype(str), top["fraud_rate_pct"], color="#3E7CB1")
        ax.set_title(col)
        ax.set_ylabel("fraud rate (%)")
        ax.tick_params(axis="x", rotation=35)
    savefig(fig, "10_context_channels.png")
    lines.append("\n## Channels / Context\n")
    for name, table in context_tables.items():
        lines.append(f"### {name}\n")
        lines.append(table[["fraud", "count", "fraud_rate_pct"]].head(12).to_markdown() + "\n")

    numeric = df.select_dtypes(include=["number"]).copy()
    corr = numeric.corr(numeric_only=True)["isFraud"].drop("isFraud").sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    top = corr.head(18)[::-1]
    ax.barh(top.index, top.values, color=np.where(top.values >= 0, "#D1495B", "#3E7CB1"))
    ax.set_title("Correlation with isFraud")
    savefig(fig, "11_corr_with_target.png")
    lines.extend(["\n## Top Numeric Correlations with Target\n", corr.head(15).round(4).to_frame("pearson_r").to_markdown() + "\n"])

    (DOCS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[eda] wrote figures -> {OUT_DIR}")
    print(f"[eda] wrote summary -> {DOCS / 'eda_summary.md'}")
    print(f"[eda] rows={n:,} fraud={fraud:,} ({fraud_rate:.4%})")
    print(fr_type[["fraud", "count", "fraud_rate_pct"]])


if __name__ == "__main__":
    main()
