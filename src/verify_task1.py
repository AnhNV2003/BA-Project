"""Verification checks for Module 1 PaySim artifact."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, PAYSIM_COLUMNS
from synth_context import FIELD_META


def directionless_auc(y, s) -> float:
    if s.nunique(dropna=True) <= 1:
        return 0.5
    try:
        auc = roc_auc_score(y, s)
        return max(float(auc), float(1 - auc))
    except Exception:
        return 0.5


def main() -> None:
    path = DATA_PROCESSED / "transactions_context.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/build_dataset.py` first.")
    df = pd.read_parquet(path)

    missing_base = [c for c in PAYSIM_COLUMNS if c not in df.columns]
    if missing_base:
        raise AssertionError(f"Missing base PaySim columns: {missing_base}")
    missing_synth = [c for c in FIELD_META if c not in df.columns]
    if missing_synth:
        raise AssertionError(f"Missing synthetic columns: {missing_synth}")

    print(f"[verify] rows={len(df):,} cols={df.shape[1]}")
    print(f"[verify] fraud={int(df['isFraud'].sum()):,} ({df['isFraud'].mean():.4%})")
    print(f"[verify] type fraud counts:\n{df.groupby('type')['isFraud'].sum().to_string()}")

    assert df["isFraud"].mean() < 0.01, "PaySim fraud rate should remain naturally imbalanced"
    assert set(df.loc[df["isFraud"] == 1, "type"].unique()).issubset({"TRANSFER", "CASH_OUT"})
    assert df[list(FIELD_META)].isna().sum().sum() == 0, "Synthetic fields should not contain NaN"
    assert df["account_age_days"].between(1, 3650).all()
    assert df["hour_of_day"].between(0, 23).all()
    assert df["day_index"].between(0, 31).all()
    assert (df["ip_billing_distance_km"] >= 0).all()
    assert (df["txn_count_last_24h"] >= 0).all()
    assert (df["time_since_last_hours"] >= -1).all()

    y = df["isFraud"]
    for col in [
        "is_new_device",
        "shipping_billing_mismatch",
        "num_failed_payment_attempts",
        "ip_billing_distance_km",
        "is_disposable_email",
        "high_risk_country",
    ]:
        auc = directionless_auc(y, df[col])
        print(f"[verify] single-feature AUC {col}={auc:.3f}")
        assert auc < 0.95, f"{col} may be too target-leaky; AUC={auc:.3f}"

    print("[verify] M1 PaySim checks passed")


if __name__ == "__main__":
    main()
