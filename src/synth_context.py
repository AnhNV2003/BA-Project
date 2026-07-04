"""Synthetic contextual / behavioural layer  (Module 1 — core deliverable).

WHY THIS EXISTS
---------------
PaySim gives transaction + balance + label but NO e-commerce risk context
(identity, device, IP, address, account history). This module adds that context
so the project is a real fraud-risk problem, not a "download-and-train" exercise.

THREE-LAYER ARCHITECTURE (defend this in the report)
----------------------------------------------------
Layer 1 — Customer IDENTITY  (Faker, per customer, cached once)
    Real `nameOrig` is 99.9% single-use, so we build our OWN fixed pool of
    synthetic customers and let Faker generate realistic identities
    (name, email handle, city, home lat/lng). Cached to disk => Faker runs once.

Layer 2 — Account RISK attributes  (numpy, per customer, conditioned on risk)
    After transactions are assigned to customers, a customer is "risky" if any
    of their transactions is fraud. Account age / country / disposable-email are
    drawn conditioned on that risk level so they stay CONSISTENT per customer.

Layer 3 — Transaction RISK signals  (numpy, per transaction, conditioned on isFraud)
    is_new_device, address mismatch, failed payment attempts, IP geo-distance
    (haversine from the customer's home), time-of-day + velocity features.

DESIGN PRINCIPLES
-----------------
* REALISM, NOT LEAKAGE: every conditional distribution differs between
  fraud/legit but OVERLAPS heavily; no single field separates the classes.
* TRANSPARENCY: all tunable parameters live in GEN below — the same source of
  truth used to generate docs/data_dictionary.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from config import SEED, DATA_SYNTH

# ---------------------------------------------------------------------------
# Generation parameters — SINGLE SOURCE OF TRUTH (also used by the data dict).
# For each conditional field: (legit_param, fraud_param). Distributions overlap.
# ---------------------------------------------------------------------------
GEN = {
    "n_customers": 200_000,        # size of the fixed synthetic customer pool

    # REALISM CONTROLS (prevent leakage / trivially-perfect separation):
    #   reveal_rate: a fraud fires each transaction-level red-flag only ~55% of
    #                the time (real fraud is partially "stealthy" — shows some
    #                flags, not all). This caps single-feature AND combined AUC.
    #   legit_noise: this fraction of legit transactions also display a red-flag
    #                (hard negatives -> realistic precision ceiling).
    "reveal_rate": 0.55,
    "legit_noise": 0.04,

    # account age in days ~ lognormal(mu, sigma); fraudsters skew YOUNG (overlap)
    "account_age_days": {"legit": (6.0, 0.9), "fraud": (5.0, 1.0)},

    # P(billing country on the platform's elevated-risk list) — per customer
    "high_risk_country": {"legit": 0.06, "fraud": 0.28},

    # P(customer uses a disposable / throwaway email) — per customer
    "is_disposable_email": {"legit": 0.04, "fraud": 0.22},

    # ---- transaction level (further softened by reveal_rate/legit_noise) ----
    "is_new_device": {"legit": 0.15, "fraud": 0.50},
    "shipping_billing_mismatch": {"legit": 0.08, "fraud": 0.42},
    "num_failed_payment_attempts": {"legit": 0.30, "fraud": 1.40},   # Poisson lambda

    # IP location offset from home (degrees) ~ lognormal -> haversine km.
    # Means closer + wider spread => heavy overlap (single-feature AUC ~0.8).
    "ip_offset_deg": {"legit": (0.2, 1.0), "fraud": (1.4, 1.1)},   # (mu, sigma)
}

_BROWSERS = ["Chrome", "Safari", "Edge", "Firefox", "Samsung Internet", "Opera"]
_OS = ["Windows", "Android", "iOS", "macOS", "Linux"]
_EMAIL_COMMON = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "proton.me"]
_EMAIL_DISPOSABLE = ["mailinator.com", "guerrillamail.com", "tempmail.com", "10minutemail.com", "trashmail.com"]
_COUNTRIES = ["US", "GB", "DE", "FR", "VN", "IN", "BR", "NG", "RU", "CN", "ID", "PH"]
_HIGH_RISK = ["NG", "RU", "CN", "ID"]
_LOW_RISK = [c for c in _COUNTRIES if c not in _HIGH_RISK]


# ---------------------------------------------------------------------------
# Layer 1 — Faker customer-identity master (generated once, cached to disk)
# ---------------------------------------------------------------------------
def get_customer_master(n_customers: int = GEN["n_customers"], seed: int = SEED,
                        verbose: bool = True) -> pd.DataFrame:
    """Fixed pool of synthetic customer identities. Faker runs once, then cached.

    Columns: customer_id, customer_name, email_handle, billing_city,
             home_lat, home_lng  (identity only — risk attributes added later).
    """
    cache = DATA_SYNTH / f"customer_master_{n_customers}.parquet"
    if cache.exists():
        if verbose:
            print(f"[synth] customer master cache hit: {cache.name}")
        return pd.read_parquet(cache)

    if verbose:
        print(f"[synth] generating {n_customers:,} Faker identities (one-time)...")
    fk = Faker()
    Faker.seed(seed)

    names, handles, cities, lats, lngs = [], [], [], [], []
    for _ in range(n_customers):
        names.append(fk.name())
        handles.append(fk.user_name())
        cities.append(fk.city())
        lat, lng = fk.latlng()
        lats.append(float(lat))
        lngs.append(float(lng))

    master = pd.DataFrame({
        "customer_id": np.char.add("U", np.arange(n_customers).astype(str)),
        "customer_name": names,
        "email_handle": handles,
        "billing_city": cities,
        "home_lat": np.round(lats, 5),
        "home_lng": np.round(lngs, 5),
    })
    try:
        master.to_parquet(cache, index=False)
    except Exception as e:               # pragma: no cover - parquet optional
        cache = cache.with_suffix(".csv")
        master.to_csv(cache, index=False)
        if verbose:
            print(f"[synth] parquet unavailable ({e}); cached CSV instead")
    if verbose:
        print(f"[synth] cached identity master -> {cache.name}")
    return master


def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Layers 2 + 3 — attach risk attributes + transaction risk signals
# ---------------------------------------------------------------------------
def add_synthetic_context(df: pd.DataFrame, n_customers: int = GEN["n_customers"],
                          seed: int = SEED, verbose: bool = True) -> pd.DataFrame:
    """Augment PaySim `df` (needs: step, isFraud) with the synthetic context."""
    rng = np.random.default_rng(seed)
    n = len(df)
    y = df["isFraud"].to_numpy()
    fraud_mask = y == 1

    master = get_customer_master(n_customers, seed, verbose)
    n_customers = len(master)

    out = df.copy()

    # -- assign each transaction to a synthetic customer --
    cust_idx = rng.integers(0, n_customers, size=n)
    out["customer_id"] = master["customer_id"].to_numpy()[cust_idx]

    # -- Layer 2: per-customer risk level (any fraud -> risky) --
    cust_risky = np.zeros(n_customers, dtype=bool)
    np.logical_or.at(cust_risky, cust_idx, fraud_mask)
    row_risky = cust_risky[cust_idx]

    mu = np.where(row_risky, GEN["account_age_days"]["fraud"][0], GEN["account_age_days"]["legit"][0])
    sg = np.where(row_risky, GEN["account_age_days"]["fraud"][1], GEN["account_age_days"]["legit"][1])
    out["account_age_days"] = np.clip(np.round(rng.lognormal(mu, sg)), 1, 3650).astype(int)

    p_hr = np.where(row_risky, GEN["high_risk_country"]["fraud"], GEN["high_risk_country"]["legit"])
    is_hr = rng.random(n) < p_hr
    out["high_risk_country"] = is_hr.astype(int)
    out["billing_country"] = np.where(
        is_hr,
        np.array(_HIGH_RISK)[rng.integers(0, len(_HIGH_RISK), n)],
        np.array(_LOW_RISK)[rng.integers(0, len(_LOW_RISK), n)],
    )

    p_disp = np.where(row_risky, GEN["is_disposable_email"]["fraud"], GEN["is_disposable_email"]["legit"])
    out["is_disposable_email"] = (rng.random(n) < p_disp).astype(int)

    # -- merge Faker identity, build email address --
    ident = master.set_index("customer_id").loc[out["customer_id"].to_numpy()].reset_index(drop=True)
    out["customer_name"] = ident["customer_name"].to_numpy()
    out["billing_city"] = ident["billing_city"].to_numpy()
    home_lat = ident["home_lat"].to_numpy()
    home_lng = ident["home_lng"].to_numpy()
    domain = np.where(
        out["is_disposable_email"].to_numpy() == 1,
        np.array(_EMAIL_DISPOSABLE)[rng.integers(0, len(_EMAIL_DISPOSABLE), n)],
        np.array(_EMAIL_COMMON)[rng.integers(0, len(_EMAIL_COMMON), n)],
    )
    out["email"] = pd.Series(ident["email_handle"].to_numpy(), index=out.index) + "@" + domain

    # -- Layer 3: transaction-level risk signals --
    # Each fraud reveals a given red-flag only with prob reveal_rate; a small
    # fraction of legit transactions also trip a flag (legit_noise). A FRESH
    # effective mask per feature decorrelates the flags, so neither any single
    # feature nor their combination separates the classes perfectly (realism).
    reveal, leak = GEN["reveal_rate"], GEN["legit_noise"]

    def eff_mask():
        return (fraud_mask & (rng.random(n) < reveal)) | ((~fraud_mask) & (rng.random(n) < leak))

    def bern(field):
        m = eff_mask()
        p = np.where(m, GEN[field]["fraud"], GEN[field]["legit"])
        return (rng.random(n) < p).astype(int)

    out["is_new_device"] = bern("is_new_device")
    out["shipping_billing_mismatch"] = bern("shipping_billing_mismatch")
    m_fail = eff_mask()
    lam = np.where(m_fail, GEN["num_failed_payment_attempts"]["fraud"], GEN["num_failed_payment_attempts"]["legit"])
    out["num_failed_payment_attempts"] = rng.poisson(lam).astype(int)

    out["browser"] = np.array(_BROWSERS)[rng.integers(0, len(_BROWSERS), n)]
    out["device_os"] = np.array(_OS)[rng.integers(0, len(_OS), n)]
    out["device_id"] = np.char.add("D", rng.integers(10**7, 10**8, n).astype(str))

    # IP geolocation: offset from home; fraud sits farther away -> haversine km.
    m_ip = eff_mask()
    omu = np.where(m_ip, GEN["ip_offset_deg"]["fraud"][0], GEN["ip_offset_deg"]["legit"][0])
    osg = np.where(m_ip, GEN["ip_offset_deg"]["fraud"][1], GEN["ip_offset_deg"]["legit"][1])
    off = rng.lognormal(omu, osg, n)                      # magnitude in degrees
    ang = rng.uniform(0, 2 * np.pi, n)
    ip_lat = np.clip(home_lat + off * np.sin(ang), -90, 90)
    ip_lng = home_lng + off * np.cos(ang)
    out["ip_billing_distance_km"] = np.round(_haversine_km(home_lat, home_lng, ip_lat, ip_lng), 1)

    # time-derived
    out["hour_of_day"] = (out["step"] % 24).astype(int)
    out["day_index"] = (out["step"] // 24).astype(int)
    out["is_night"] = out["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    out = _add_velocity_features(out)

    if verbose:
        added = [c for c in out.columns if c not in df.columns]
        print(f"[synth] added {len(added)} columns to {n:,} rows")
    return out


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer behaviour from the assignment + step (O(n) two-pointer).

    account_txn_total / account_txn_index / time_since_last_hours / txn_count_last_24h
    """
    order = np.lexsort((df["step"].to_numpy(), df["customer_id"].to_numpy()))
    cust = df["customer_id"].to_numpy()[order]
    step = df["step"].to_numpy()[order]
    m = len(df)
    total = np.empty(m, np.int32); idx = np.empty(m, np.int32)
    since = np.empty(m, np.int32); cnt24 = np.empty(m, np.int32)

    i = 0
    while i < m:                       # contiguous block per customer
        j = i
        while j < m and cust[j] == cust[i]:
            j += 1
        left = i
        for k in range(i, j):
            idx[k] = k - i
            total[k] = j - i
            since[k] = -1 if k == i else int(step[k] - step[k - 1])
            while step[k] - step[left] > 24:
                left += 1
            cnt24[k] = k - left
        i = j

    inv = np.empty(m, np.int64); inv[order] = np.arange(m)
    df = df.copy()
    df["account_txn_total"] = total[inv]
    df["account_txn_index"] = idx[inv]
    df["time_since_last_hours"] = since[inv]
    df["txn_count_last_24h"] = cnt24[inv]
    return df


if __name__ == "__main__":
    from data_base import load_base_data

    base = load_base_data(sample_frac=0.02)
    aug = add_synthetic_context(base)
    print("\nShape:", aug.shape)
    print("\nSample identities:")
    print(aug[["customer_name", "email", "billing_city", "billing_country", "ip_billing_distance_km"]].head())
    cols = ["account_age_days", "is_new_device", "shipping_billing_mismatch",
            "num_failed_payment_attempts", "ip_billing_distance_km",
            "is_disposable_email", "high_risk_country", "txn_count_last_24h"]
    tbl = aug.groupby("isFraud")[cols].mean().T
    tbl.columns = ["legit(0)", "fraud(1)"]
    print("\nFraud vs legit (differ but overlap):\n", tbl.round(3))
