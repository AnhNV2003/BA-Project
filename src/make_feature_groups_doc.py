"""Generate docs/feature_groups.md — the exact column membership of every
training feature scenario (base/dest/synth/realistic/all).

Source of truth is code, not prose: the group -> column mapping is imported from
`train_validate.FEATURE_GROUPS` (what actually reaches the model) and the leaky
set from `train_validate.LEAKY_GROUPS`. Re-run after changing features:

    python src/make_feature_groups_doc.py
"""
from __future__ import annotations

from config import DOCS
from train_validate import FEATURE_GROUPS, LEAKY_GROUPS

# Per-column metadata. `auth` = available at authorization time (before the
# transaction settles); `leaky` = encodes post-transaction state / the label.
# Unknown columns fall back to a safe default so the script never crashes.
Meta = tuple[str, str, bool, bool, str]  # (layer, dtype/unit, auth, leaky, description)

COLUMN_META: dict[str, Meta] = {
    # --- Base PaySim: amount ---
    "amount": ("Base PaySim", "float / currency", True, False, "Transaction amount."),
    "log_amount": ("Base PaySim", "float", True, False, "log1p(amount) — tames heavy tail."),
    "amount_cents": ("Base PaySim", "int 0-99", True, False, "Decimal part of amount (round-number vs odd — card-testing signal)."),
    # --- Base PaySim: balances BEFORE (available) ---
    "oldbalanceOrg": ("Base PaySim", "float / currency", True, False, "Originator balance BEFORE the transaction."),
    "oldbalanceDest": ("Base PaySim", "float / currency", True, False, "Destination balance BEFORE the transaction."),
    # --- Base PaySim: balances AFTER / derived (LEAKY) ---
    "newbalanceOrig": ("Base PaySim (post-txn)", "float / currency", False, True, "Originator balance AFTER — unknown at authorization time."),
    "newbalanceDest": ("Base PaySim (post-txn)", "float / currency", False, True, "Destination balance AFTER — unknown at authorization time."),
    "errorBalanceOrig": ("Balance-derived (post-txn)", "float", False, True, "oldbalanceOrg - amount - newbalanceOrig; uses post-txn balance."),
    "errorBalanceDest": ("Balance-derived (post-txn)", "float", False, True, "oldbalanceDest + amount - newbalanceDest; uses post-txn balance."),
    "orig_drained": ("Balance-derived (post-txn)", "0/1", False, True, "Account emptied (newbalanceOrig==0 & oldbalanceOrg>0); near-deterministic for fraud."),
    "dest_was_empty": ("Base PaySim", "0/1", True, False, "Destination balance was 0 before (uses pre-txn balance)."),
    # --- Base PaySim: type flags ---
    "is_transfer": ("Base PaySim", "0/1", True, False, "type == TRANSFER."),
    "is_cash_out": ("Base PaySim", "0/1", True, False, "type == CASH_OUT."),
    # --- Destination history (past-only) ---
    "dest_seen_before": ("Destination history (past-only)", "0/1", True, False, "nameDest appeared in an earlier transaction."),
    "dest_txn_count_so_far": ("Destination history (past-only)", "int", True, False, "Count of prior transactions to this nameDest."),
    "dest_amount_sum_so_far": ("Destination history (past-only)", "float", True, False, "Sum of prior amounts received by this nameDest."),
    "dest_amount_mean_so_far": ("Destination history (past-only)", "float", True, False, "Mean of prior amounts to this nameDest."),
    "dest_amount_std_so_far": ("Destination history (past-only)", "float", True, False, "Std of prior amounts to this nameDest."),
    "dest_unique_senders_so_far": ("Destination history (past-only)", "int", True, False, "Distinct prior senders to this nameDest (fan-in / mule signal)."),
    "dest_cash_in_count_so_far": ("Destination history (past-only)", "int", True, False, "Prior CASH_IN count to this nameDest."),
    "dest_cash_out_count_so_far": ("Destination history (past-only)", "int", True, False, "Prior CASH_OUT count to this nameDest."),
    "dest_debit_count_so_far": ("Destination history (past-only)", "int", True, False, "Prior DEBIT count to this nameDest."),
    "dest_payment_count_so_far": ("Destination history (past-only)", "int", True, False, "Prior PAYMENT count to this nameDest."),
    "dest_transfer_count_so_far": ("Destination history (past-only)", "int", True, False, "Prior TRANSFER count to this nameDest."),
    "time_since_dest_last_seen": ("Destination history (past-only)", "int / steps(hr)", True, False, "Steps since this nameDest was last seen (-1 if first time)."),
    "amount_to_dest_mean_ratio": ("Destination history (past-only)", "float", True, False, "amount / prior mean amount to this nameDest (anomaly)."),
    "amount_dest_zscore": ("Destination history (past-only)", "float", True, False, "z-score of amount vs this nameDest's prior mean/std."),
    "dest_freq_so_far": ("Destination history (past-only)", "float", True, False, "Frequency-encoding of nameDest from past rows (replaces raw ID)."),
    "orig_freq_so_far": ("Destination history (past-only)", "float", True, False, "Frequency-encoding of nameOrig from past rows (replaces raw ID)."),
    # --- Synthetic / contextual (M1) ---
    "account_age_days": ("Synthetic context (M1)", "int / days", True, False, "Synthetic account tenure."),
    "is_new_device": ("Synthetic context (M1)", "0/1", True, False, "Transaction from a new device."),
    "shipping_billing_mismatch": ("Synthetic context (M1)", "0/1", True, False, "Shipping/billing address mismatch."),
    "num_failed_payment_attempts": ("Synthetic context (M1)", "int", True, False, "Recent failed payment attempts."),
    "ip_billing_distance_km": ("Synthetic context (M1)", "float / km", True, False, "Distance between IP geolocation and billing address."),
    "log_ip_distance": ("Synthetic context (M1)", "float", True, False, "log1p(ip_billing_distance_km)."),
    "is_disposable_email": ("Synthetic context (M1)", "0/1", True, False, "Disposable/temporary email domain."),
    "high_risk_country": ("Synthetic context (M1)", "0/1", True, False, "Billing country flagged high-risk."),
    "hour_of_day": ("Synthetic context (M1)", "int 0-23", True, False, "Hour derived from step."),
    "is_night": ("Synthetic context (M1)", "0/1", True, False, "Night-time transaction flag."),
    "txn_count_last_24h": ("Synthetic context (M1)", "int", True, False, "Account velocity: transactions in the last 24h."),
    "time_since_last_hours": ("Synthetic context (M1)", "float / hr", True, False, "Hours since this account's previous transaction (-1 if first)."),
    "account_txn_total": ("Synthetic context (M1)", "int", True, False, "Total transactions for this synthetic account."),
    # --- Encoded categorical ---
    "type__CASH_IN": ("Encoded categorical", "0/1", True, False, "One-hot: type == CASH_IN."),
    "type__CASH_OUT": ("Encoded categorical", "0/1", True, False, "One-hot: type == CASH_OUT."),
    "type__DEBIT": ("Encoded categorical", "0/1", True, False, "One-hot: type == DEBIT."),
    "type__PAYMENT": ("Encoded categorical", "0/1", True, False, "One-hot: type == PAYMENT."),
    "type__TRANSFER": ("Encoded categorical", "0/1", True, False, "One-hot: type == TRANSFER."),
    "browser_freq_train": ("Encoded categorical", "float", True, False, "Train-frequency encoding of browser."),
    "device_os_freq_train": ("Encoded categorical", "float", True, False, "Train-frequency encoding of device OS."),
    "billing_country_freq_train": ("Encoded categorical", "float", True, False, "Train-frequency encoding of billing country."),
}

DEFAULT_META: Meta = ("(uncategorised)", "—", True, False, "—")
GROUP_ORDER = ["base", "dest", "synth", "realistic", "all"]
GROUP_PURPOSE = {
    "base": "Full PaySim amount/balance signature (includes post-transaction balances).",
    "dest": "Past-only destination-account (nameDest) behaviour.",
    "synth": "M1 synthetic/contextual risk signals + encoded categoricals.",
    "realistic": "Authorization-time features (pre-txn balances) + destination history + context.",
    "all": "Everything: base + destination history + synthetic + encoded categoricals.",
}


def meta(col: str) -> Meta:
    return COLUMN_META.get(col, DEFAULT_META)


def flag_leaky(group: str) -> str:
    return "🚩 leaky" if group in LEAKY_GROUPS else "✅ deployable"


def canonical_columns() -> list[str]:
    order: list[str] = []
    for group in GROUP_ORDER:
        for col in FEATURE_GROUPS[group]:
            if col not in order:
                order.append(col)
    return order


def build() -> str:
    lines: list[str] = [
        "# Feature Groups — PaySim Fraud Detection\n",
        "> Auto-generated by `python src/make_feature_groups_doc.py` from "
        "`src/features.py` + `src/train_validate.py`. **Do not edit by hand** — "
        "re-run after changing features so this stays in sync with the code.\n",
        "\n## Overview\n",
    ]

    overview = ["| group | # features | status | purpose |", "|---|---:|---|---|"]
    for group in GROUP_ORDER:
        cols = FEATURE_GROUPS[group]
        overview.append(f"| `{group}` | {len(cols)} | {flag_leaky(group)} | {GROUP_PURPOSE[group]} |")
    lines.append("\n".join(overview) + "\n")

    lines += [
        "\n## Legend\n",
        "- **🚩 leaky** — contains post-transaction balances (`newbalance*`, `errorBalance*`, `orig_drained`) that near-deterministically encode the label and are **unknown at authorization time**. Kept only as an upper-bound reference; excluded from the deployable bundle.\n",
        "- **✅ deployable** — every column is available at authorization time (past-only or pre-transaction). `realistic` drives the saved model bundle.\n",
        "- **Auth-time** column flag: `Y` = known when scoring the transaction; `N` = only known after it settles.\n",
    ]

    # Per-group column tables.
    for group in GROUP_ORDER:
        cols = FEATURE_GROUPS[group]
        lines.append(f"\n## `{group}` — {len(cols)} features ({flag_leaky(group)})\n")
        lines.append(GROUP_PURPOSE[group] + "\n")
        table = ["| # | column | layer | type / unit | auth-time | leaky | description |",
                 "|---:|---|---|---|:---:|:---:|---|"]
        for i, col in enumerate(cols, 1):
            layer, dtype, auth, leaky, desc = meta(col)
            table.append(
                f"| {i} | `{col}` | {layer} | {dtype} | "
                f"{'Y' if auth else 'N'} | {'🚩' if leaky else ''} | {desc} |"
            )
        lines.append("\n".join(table) + "\n")

    # Column x group membership matrix.
    lines.append("\n## Column × group membership matrix\n")
    header = "| column | layer | auth | leaky | " + " | ".join(GROUP_ORDER) + " |"
    sep = "|---|---|:---:|:---:|" + "|".join([":---:"] * len(GROUP_ORDER)) + "|"
    matrix = [header, sep]
    for col in canonical_columns():
        layer, _dtype, auth, leaky, _desc = meta(col)
        marks = ["✓" if col in FEATURE_GROUPS[g] else "" for g in GROUP_ORDER]
        matrix.append(
            f"| `{col}` | {layer} | {'Y' if auth else 'N'} | {'🚩' if leaky else ''} | "
            + " | ".join(marks) + " |"
        )
    lines.append("\n".join(matrix) + "\n")

    # Coverage check: warn if any column lacks metadata.
    uncategorised = sorted({c for c in canonical_columns() if c not in COLUMN_META})
    if uncategorised:
        lines.append(
            "\n## ⚠️ Columns missing metadata\n"
            "These reached a feature group but have no entry in `COLUMN_META` "
            "(add one in `make_feature_groups_doc.py`):\n"
            + "\n".join(f"- `{c}`" for c in uncategorised) + "\n"
        )

    return "\n".join(lines)


def main() -> None:
    out = DOCS / "feature_groups.md"
    out.write_text(build(), encoding="utf-8")
    n = len(canonical_columns())
    missing = [c for c in canonical_columns() if c not in COLUMN_META]
    print(f"[feature-groups] wrote {out} ({n} distinct columns across {len(FEATURE_GROUPS)} groups)")
    if missing:
        print(f"[feature-groups] WARNING: {len(missing)} columns missing metadata: {missing}")
    else:
        print("[feature-groups] all columns documented.")


if __name__ == "__main__":
    main()
