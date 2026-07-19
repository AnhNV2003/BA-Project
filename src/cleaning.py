"""Conservative data cleaning and validation for PaySim (Module 3)."""
from __future__ import annotations

import pandas as pd

from config import DATA_PROCESSED, DOCS, PAYSIM_COLUMNS
from synth_context import _BROWSERS, _COUNTRIES, _OS


VALID_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}


def load() -> pd.DataFrame:
    path = DATA_PROCESSED / "transactions_context.parquet"
    if not path.exists():
        path = path.with_suffix(".csv")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def missing_table(df: pd.DataFrame, *, only_nonzero: bool = False) -> pd.DataFrame:
    rows = []
    for col, n_missing in df.isna().sum().items():
        if only_nonzero and n_missing == 0:
            continue
        rows.append({
            "column": col,
            "missing": int(n_missing),
            "missing_pct": round(float(n_missing / len(df) * 100), 4),
        })
    return pd.DataFrame(rows)


def category_profile(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df:
            rows.append({"column": col, "nunique": 0, "values": "missing column"})
            continue
        values = sorted(str(v) for v in df[col].dropna().unique())
        preview = ", ".join(values[:20])
        if len(values) > 20:
            preview += f", ... (+{len(values) - 20} more)"
        rows.append({"column": col, "nunique": len(values), "values": preview})
    return pd.DataFrame(rows)


def validate_category(df: pd.DataFrame, col: str, valid: set[str]) -> tuple[pd.Series, pd.DataFrame]:
    invalid = df[col].notna() & ~df[col].isin(valid)
    invalid_values = (
        df.loc[invalid, col]
        .value_counts(dropna=False)
        .rename_axis("invalid_value")
        .reset_index(name="count")
    )
    return invalid, invalid_values


def normalize_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    out = df.copy()
    decisions: list[str] = []

    before_type = out["type"].copy()
    out["type"] = out["type"].astype("string").str.strip().str.upper()
    if (before_type.astype("string").fillna("<NA>") != out["type"].astype("string").fillna("<NA>")).any():
        decisions.append("Normalized `type` with strip + uppercase.")

    before_browser = out["browser"].copy()
    browser_map = {v.casefold(): v for v in _BROWSERS}
    out["browser"] = (
        out["browser"].astype("string").str.strip().map(lambda v: browser_map.get(str(v).casefold(), v))
    )
    if (before_browser.astype("string").fillna("<NA>") != out["browser"].astype("string").fillna("<NA>")).any():
        decisions.append("Normalized `browser` with strip + canonical generator casing.")

    before_os = out["device_os"].copy()
    os_map = {v.casefold(): v for v in _OS}
    out["device_os"] = (
        out["device_os"].astype("string").str.strip().map(lambda v: os_map.get(str(v).casefold(), v))
    )
    if (before_os.astype("string").fillna("<NA>") != out["device_os"].astype("string").fillna("<NA>")).any():
        decisions.append("Normalized `device_os` with strip + canonical generator casing.")

    before_country = out["billing_country"].copy()
    out["billing_country"] = out["billing_country"].astype("string").str.strip().str.upper()
    if (before_country.astype("string").fillna("<NA>") != out["billing_country"].astype("string").fillna("<NA>")).any():
        decisions.append("Normalized `billing_country` with strip + uppercase.")

    invalids: dict[str, pd.DataFrame] = {}
    for col, valid in {
        "type": VALID_TYPES,
        "browser": set(_BROWSERS),
        "device_os": set(_OS),
        "billing_country": set(_COUNTRIES),
    }.items():
        invalid_mask, invalid_values = validate_category(out, col, valid)
        invalids[col] = invalid_values
        if invalid_mask.any():
            out.loc[invalid_mask, col] = pd.NA
            decisions.append(f"Set {int(invalid_mask.sum()):,} invalid `{col}` values to missing.")

    if not decisions:
        decisions.append("No category normalization or invalid-category repair was needed.")
    return out, decisions, invalids


def main() -> None:
    df = load()
    before_rows = len(df)
    before_cols = df.shape[1]
    before_fraud = int(df["isFraud"].sum())
    before_rate = float(df["isFraud"].mean())
    decisions: list[str] = []

    category_cols = ["type", "browser", "device_os", "billing_country"]
    missing_before = missing_table(df)
    category_before = category_profile(df, category_cols)

    dup_mask = df.duplicated(subset=PAYSIM_COLUMNS, keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        df = df.loc[~dup_mask].copy()
        decisions.append(f"Dropped {n_dup:,} duplicate base transactions, keeping first occurrence.")
    else:
        decisions.append("No duplicate base transactions found.")

    df, category_decisions, invalid_category_values = normalize_categories(df)
    decisions.extend(category_decisions)
    category_after = category_profile(df, category_cols)

    n_neg = int((df["amount"] < 0).sum())
    n_zero = int((df["amount"] == 0).sum())
    if n_neg:
        df = df.loc[df["amount"] >= 0].copy()
        decisions.append(f"Removed {n_neg:,} rows with negative amount.")
    else:
        decisions.append("No negative amount rows found.")
    df["flag_zero_amount"] = (df["amount"] == 0).astype("int8")
    decisions.append("Kept zero-amount rows and added `flag_zero_amount`.")

    amount_p99 = float(df["amount"].quantile(0.99))
    amount_p999 = float(df["amount"].quantile(0.999))
    amount_max = float(df["amount"].max())
    amount_out_p99 = int((df["amount"] > amount_p99).sum())
    amount_out_p999 = int((df["amount"] > amount_p999).sum())
    df["amount_capped"] = df["amount"].clip(upper=amount_p999)
    decisions.append("Kept amount outliers and added `amount_capped` at p99.9 for optional robust modelling.")

    step_invalid = int((~df["step"].between(1, 743)).sum())
    if step_invalid:
        decisions.append(f"Found {step_invalid:,} rows outside expected `step` range 1..743; documented only.")
    else:
        decisions.append("All `step` values are inside expected PaySim range 1..743.")

    dest_zero = int(((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)).sum())
    err_orig = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    err_dest = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    n_err_orig = int((err_orig.abs() > 1e-6).sum())
    n_err_dest = int((err_dest.abs() > 1e-6).sum())
    decisions.append("Preserved balance-reconciliation quirks as predictive PaySim signal.")

    synthetic_checks = pd.DataFrame([
        {"check": "hour_of_day in 0..23", "status": bool(df["hour_of_day"].between(0, 23).all())},
        {"check": "account_age_days >= 1", "status": bool((df["account_age_days"] >= 1).all())},
        {"check": "num_failed_payment_attempts >= 0", "status": bool((df["num_failed_payment_attempts"] >= 0).all())},
        {"check": "ip_billing_distance_km >= 0", "status": bool((df["ip_billing_distance_km"] >= 0).all())},
    ])

    missing_after = missing_table(df)
    after_rows = len(df)
    after_cols = df.shape[1]
    after_fraud = int(df["isFraud"].sum())
    after_rate = float(df["isFraud"].mean())

    before_after = pd.DataFrame([
        {"metric": "rows", "before": before_rows, "after": after_rows, "change": after_rows - before_rows},
        {"metric": "columns", "before": before_cols, "after": after_cols, "change": after_cols - before_cols},
        {"metric": "fraud_rows", "before": before_fraud, "after": after_fraud, "change": after_fraud - before_fraud},
        {"metric": "fraud_rate", "before": round(before_rate, 8), "after": round(after_rate, 8), "change": round(after_rate - before_rate, 8)},
        {"metric": "missing_cells", "before": int(missing_before["missing"].sum()), "after": int(missing_after["missing"].sum()), "change": int(missing_after["missing"].sum() - missing_before["missing"].sum())},
        {"metric": "duplicate_base_rows", "before": n_dup, "after": 0, "change": -n_dup},
        {"metric": "negative_amount_rows", "before": n_neg, "after": int((df["amount"] < 0).sum()), "change": -n_neg},
        {"metric": "zero_amount_rows", "before": n_zero, "after": int(df["flag_zero_amount"].sum()), "change": 0},
        {"metric": "step_outside_1_743", "before": step_invalid, "after": step_invalid, "change": 0},
    ])

    amount_report = pd.DataFrame([
        {"metric": "amount_p99", "value": round(amount_p99, 3)},
        {"metric": "amount_p999", "value": round(amount_p999, 3)},
        {"metric": "amount_max", "value": round(amount_max, 3)},
        {"metric": "rows_above_p99", "value": amount_out_p99},
        {"metric": "rows_above_p999", "value": amount_out_p999},
    ])

    balance_report = pd.DataFrame([
        {"check": "destination balances 0 before/after while amount > 0", "rows": dest_zero, "decision": "preserve"},
        {"check": "origin balance reconciliation error", "rows": n_err_orig, "decision": "preserve"},
        {"check": "destination balance reconciliation error", "rows": n_err_dest, "decision": "preserve"},
    ])

    invalid_sections = []
    for col, table in invalid_category_values.items():
        invalid_sections.append(f"### `{col}` invalid values\n")
        if table.empty:
            invalid_sections.append("No inconsistency found.\n")
        else:
            invalid_sections.append(table.to_markdown(index=False) + "\n")

    log = [
        "# Cleaning Report — PaySim E-Commerce Fraud Detection\n",
        "## Scope\n",
        "- Input: `data/processed/transactions_context.parquet`.\n",
        "- Cleaning is conservative: validate and document, remove only exact duplicates or negative amounts if found.\n",
        "- PaySim balance-reconciliation quirks are preserved because they are predictive signal.\n",
        "\n## Before / After Summary\n",
        before_after.to_markdown(index=False) + "\n",
        "\n## Missing Values\n",
        "Before:\n",
        missing_before.to_markdown(index=False) + "\n",
        "After:\n",
        missing_after.to_markdown(index=False) + "\n",
        "\n## Duplicate Base Transactions\n",
        f"- Exact duplicates on PaySim base columns: **{n_dup:,}**.\n",
        "\n## Category Validation\n",
        "Expected sets:\n",
        pd.DataFrame([
            {"column": "type", "valid_values": ", ".join(sorted(VALID_TYPES))},
            {"column": "browser", "valid_values": ", ".join(_BROWSERS)},
            {"column": "device_os", "valid_values": ", ".join(_OS)},
            {"column": "billing_country", "valid_values": ", ".join(_COUNTRIES)},
        ]).to_markdown(index=False) + "\n",
        "Before category value-set:\n",
        category_before.to_markdown(index=False) + "\n",
        "After category value-set:\n",
        category_after.to_markdown(index=False) + "\n",
        *invalid_sections,
        "\n## Amount / Step Validation\n",
        f"- Negative amount rows: **{n_neg:,}**.\n",
        f"- Zero amount rows: **{n_zero:,}**; kept and flagged with `flag_zero_amount`.\n",
        f"- Rows outside `step` range 1..743: **{step_invalid:,}**.\n",
        amount_report.to_markdown(index=False) + "\n",
        "\n## Balance Reconciliation Quirks\n",
        balance_report.to_markdown(index=False) + "\n",
        "\n## Synthetic Field Range Validation\n",
        synthetic_checks.assign(status=synthetic_checks["status"].map({True: "OK", False: "FAIL"})).to_markdown(index=False) + "\n",
        "\n## Decisions\n",
        "".join(f"- {d}\n" for d in decisions),
    ]

    out = DATA_PROCESSED / "transactions_clean.parquet"
    df.to_parquet(out, index=False)
    (DOCS / "cleaning_report.md").write_text("\n".join(log), encoding="utf-8")

    print(f"[clean] {before_rows:,} -> {after_rows:,} rows")
    print(f"[clean] fraud_rate {before_rate:.4%} -> {after_rate:.4%}")
    print(f"[clean] duplicates={n_dup:,} neg_amount={n_neg:,} zero_amount={n_zero:,}")
    print(f"[clean] amount p99={amount_p99:,.0f} p99.9={amount_p999:,.0f} max={amount_max:,.0f}")
    print(f"[clean] balance-error rows kept: origin={n_err_orig:,} dest={n_err_dest:,}")
    print(f"[clean] wrote cleaned data -> {out}")


if __name__ == "__main__":
    main()
