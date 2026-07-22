"""Base-data access layer.

`load_base_data()` returns a PaySim-schema DataFrame. It PREFERS a real Kaggle
CSV dropped into data/raw/; if none is found it falls back to a schema-identical
stand-in so the whole pipeline is runnable before the download completes.

When the real CSV arrives, drop it in data/raw/ and everything downstream works
unchanged — same columns, same target.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from config import (
    DATA_RAW,
    DATA_SYNTH,
    PAYSIM_CSV,
    PAYSIM_DIR,
    PAYSIM_COLUMNS,
    FRAUD_TYPES,
    SEED,
    STANDIN_FRAUD_RATE,
    STANDIN_N_ROWS,
)

STANDIN_PATH = DATA_SYNTH / "paysim_standin.csv"


# ---------------------------------------------------------------------------
# Real CSV discovery + validation
# ---------------------------------------------------------------------------
def find_real_csv():
    """Return the configured PaySim CSV, or first matching CSV fallback."""
    candidates = []
    if PAYSIM_CSV.exists():
        candidates.append(PAYSIM_CSV)
    candidates.extend(sorted(PAYSIM_DIR.glob("*.csv")) if PAYSIM_DIR.exists() else [])
    candidates.extend(sorted(DATA_RAW.glob("*.csv")))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        if set(PAYSIM_COLUMNS).issubset(head.columns):
            return path
    return None


def validate_schema(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in PAYSIM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] is missing required PaySim columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Stand-in generator (schema-identical to real PaySim)
# ---------------------------------------------------------------------------
def make_standin(n: int = STANDIN_N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Generate a PaySim-schema stand-in that preserves the key REAL structure:

    * fraud occurs ONLY in TRANSFER / CASH_OUT
    * fraudulent origins are drained: oldbalanceOrg == amount, newbalanceOrig == 0
    * mule destinations often start/stay empty
    * isFlaggedFraud fires on the naive rule (TRANSFER & amount > 200_000)
    * fraud prevalence ~0.15% (severe imbalance)

    Accounts are drawn from a pool smaller than n so some repeat -> velocity
    features have *some* signal. NOTE: real PaySim origins are mostly unique, so
    velocity features must be re-validated on the real file.
    """
    rng = np.random.default_rng(seed)

    types = np.array(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"])
    type_probs = [0.338, 0.352, 0.220, 0.084, 0.006]
    typ = rng.choice(types, size=n, p=type_probs)

    step = rng.integers(1, 744, size=n)  # 1..743 hours ~ 30 days

    amount = np.round(rng.lognormal(mean=8.0, sigma=1.4, size=n), 2)

    # Account pool: ~2 transactions per account on average.
    n_accounts = max(1, n // 2)
    pool = np.array(
        ["C" + str(x) for x in rng.integers(10**8, 10**9, size=n_accounts)]
    )
    name_orig = rng.choice(pool, size=n)

    # Destinations: merchants (M...) for PAYMENT, else customers (C...).
    # Draw from pools SMALLER than n so destinations recur, mirroring the origin
    # pool above. Real PaySim reuses customer/mule and merchant destinations
    # heavily; the causal dest-history features (dest_txn_count_so_far,
    # dest_seen_before) and their M4/M5 guards depend on that reuse existing.
    n_dest_customers = max(1, n // 10)   # ~ many txns per customer destination
    n_dest_merchants = max(1, n // 20)   # merchants reused even more
    dest_customer_pool = np.array(
        ["C" + str(x) for x in rng.integers(10**8, 10**9, size=n_dest_customers)]
    )
    dest_merchant_pool = np.array(
        ["M" + str(x) for x in rng.integers(10**8, 10**9, size=n_dest_merchants)]
    )
    name_dest = np.where(
        typ == "PAYMENT",
        rng.choice(dest_merchant_pool, size=n),
        rng.choice(dest_customer_pool, size=n),
    )

    # Balances (legit defaults)
    old_org = np.round(rng.lognormal(mean=9.0, sigma=1.2, size=n), 2)
    outgoing = np.isin(typ, ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT"])
    new_org = np.where(outgoing, np.maximum(old_org - amount, 0.0), old_org + amount)
    new_org = np.round(new_org, 2)

    old_dest = np.round(rng.lognormal(mean=9.0, sigma=1.3, size=n), 2)
    # Merchants (PAYMENT) report 0 balances in PaySim.
    old_dest = np.where(typ == "PAYMENT", 0.0, old_dest)
    new_dest = np.where(typ == "PAYMENT", 0.0, np.round(old_dest + amount, 2))

    # ---- Fraud assignment: only eligible types, ~STANDIN_FRAUD_RATE overall ----
    is_fraud = np.zeros(n, dtype=int)
    eligible = np.isin(typ, FRAUD_TYPES)
    elig_idx = np.flatnonzero(eligible)
    n_fraud = max(1, int(round(n * STANDIN_FRAUD_RATE)))
    n_fraud = min(n_fraud, elig_idx.size)
    fraud_idx = rng.choice(elig_idx, size=n_fraud, replace=False)
    is_fraud[fraud_idx] = 1

    # Fraud signature: drain the origin, mule destination empty, larger amounts.
    amount[fraud_idx] = np.round(rng.lognormal(mean=10.5, sigma=1.0, size=n_fraud), 2)
    old_org[fraud_idx] = amount[fraud_idx]
    new_org[fraud_idx] = 0.0
    old_dest[fraud_idx] = 0.0
    new_dest[fraud_idx] = 0.0

    is_flagged = ((typ == "TRANSFER") & (amount > 200_000)).astype(int)

    df = pd.DataFrame(
        {
            "step": step,
            "type": typ,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "nameDest": name_dest,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    return df[PAYSIM_COLUMNS]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_base_data(sample_frac: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Load base transactions. Real CSV if present in data/raw/, else stand-in.

    sample_frac: if the real file is huge (~6.3M rows), pass e.g. 0.05 to work
                 on a stratified-by-fraud sample during development.
    """
    real = find_real_csv()
    if real is not None:
        if verbose:
            print(f"[base] Loading REAL PaySim CSV: {real.name}")
        df = pd.read_csv(real)
        validate_schema(df, real.name)
        df = df[PAYSIM_COLUMNS].copy()
        source = "real"
    else:
        if STANDIN_PATH.exists():
            if verbose:
                print(f"[base] No real CSV in data/raw/. Using cached stand-in: {STANDIN_PATH.name}")
            df = pd.read_csv(STANDIN_PATH)
        else:
            if verbose:
                print("[base] No real CSV in data/raw/. Generating PaySim stand-in...")
            df = make_standin()
            df.to_csv(STANDIN_PATH, index=False)
        source = "standin"

    if sample_frac is not None and 0 < sample_frac < 1:
        # Stratify on the label so we keep enough fraud in the sample.
        # GroupBy.sample (not .apply) keeps the grouping column under pandas 3.0.
        df = (
            df.groupby("isFraud", group_keys=False)
            .sample(frac=sample_frac, random_state=SEED)
            .reset_index(drop=True)
        )
        if verbose:
            print(f"[base] Sampled {sample_frac:.1%} (stratified) -> {len(df):,} rows")

    df.attrs["source"] = source
    if verbose:
        rate = df["isFraud"].mean()
        print(f"[base] rows={len(df):,}  fraud={df['isFraud'].sum():,} ({rate:.4%})  source={source}")
    return df


if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else None
    d = load_base_data(sample_frac=frac)
    print(d.head())
    print("\nType distribution:\n", d["type"].value_counts())
    print("\nFraud by type:\n", d.groupby("type")["isFraud"].mean())
