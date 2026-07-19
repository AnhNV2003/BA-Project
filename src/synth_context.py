"""Synthetic contextual / behavioural layer for PaySim (Module 1).

PaySim provides transaction, balances, and labels, but lacks e-commerce risk
context such as identity, device, IP, address, account age, and failed payment
attempts. This module adds a transparent synthetic layer with overlapping
fraud/legit distributions so no synthetic field perfectly separates fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from config import DATA_SYNTH, SEED

GEN = {
    "n_customers": 200_000,
    "reveal_rate": 0.55,
    "legit_noise": 0.04,
    "account_age_days": {"legit": (6.0, 0.9), "fraud": (5.0, 1.0)},
    "high_risk_country": {"legit": 0.06, "fraud": 0.28},
    "is_disposable_email": {"legit": 0.04, "fraud": 0.22},
    "is_new_device": {"legit": 0.15, "fraud": 0.50},
    "shipping_billing_mismatch": {"legit": 0.08, "fraud": 0.42},
    "num_failed_payment_attempts": {"legit": 0.30, "fraud": 1.40},
    "ip_offset_deg": {"legit": (0.2, 1.0), "fraud": (1.4, 1.1)},
}

_BROWSERS = ["Chrome", "Safari", "Edge", "Firefox", "Samsung Internet", "Opera"]
_OS = ["Windows", "Android", "iOS", "macOS", "Linux"]
_EMAIL_COMMON = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "proton.me"]
_EMAIL_DISPOSABLE = ["mailinator.com", "guerrillamail.com", "tempmail.com", "10minutemail.com", "trashmail.com"]
_COUNTRIES = ["US", "GB", "DE", "FR", "VN", "IN", "BR", "NG", "RU", "CN", "ID", "PH"]
_HIGH_RISK = ["NG", "RU", "CN", "ID"]
_LOW_RISK = [c for c in _COUNTRIES if c not in _HIGH_RISK]

FIELD_META = {
    "customer_id": {
        "layer": "L1 Identity",
        "type": "string",
        "unit": "-",
        "range": f"U0..U{GEN['n_customers'] - 1}",
        "fraud_conditioned": "no",
        "logic": "Assigned from a fixed synthetic customer pool because real nameOrig is almost always single-use.",
    },
    "customer_name": {
        "layer": "L1 Identity",
        "type": "string",
        "unit": "-",
        "range": "Faker generated name",
        "fraud_conditioned": "no",
        "logic": "Faker name for review-queue/demo display only; not intended as a model feature.",
    },
    "email": {
        "layer": "L1+L2 Identity/Risk",
        "type": "string",
        "unit": "-",
        "range": "handle@domain",
        "fraud_conditioned": "via disposable flag",
        "logic": "Faker handle plus common or disposable domain depending on is_disposable_email.",
    },
    "billing_city": {
        "layer": "L1 Identity",
        "type": "string",
        "unit": "-",
        "range": "Faker city",
        "fraud_conditioned": "no",
        "logic": "Faker city for analyst-facing context only.",
    },
    "account_age_days": {
        "layer": "L2 Account Risk",
        "type": "int",
        "unit": "days",
        "range": "1..3650",
        "fraud_conditioned": "yes, softened",
        "logic": f"Lognormal days; legit={GEN['account_age_days']['legit']}, fraud={GEN['account_age_days']['fraud']}, applied through reveal/noise risk mask.",
    },
    "billing_country": {
        "layer": "L2 Account Risk",
        "type": "category",
        "unit": "ISO country",
        "range": ", ".join(_COUNTRIES),
        "fraud_conditioned": "yes, softened",
        "logic": "High-risk countries are NG/RU/CN/ID; selected by high_risk_country flag.",
    },
    "high_risk_country": {
        "layer": "L2 Account Risk",
        "type": "int",
        "unit": "flag",
        "range": "{0,1}",
        "fraud_conditioned": "yes, softened",
        "logic": f"Bernoulli high-risk country; legit={GEN['high_risk_country']['legit']}, fraud={GEN['high_risk_country']['fraud']}, softened by reveal/noise.",
    },
    "is_disposable_email": {
        "layer": "L2 Account Risk",
        "type": "int",
        "unit": "flag",
        "range": "{0,1}",
        "fraud_conditioned": "yes, softened",
        "logic": f"Bernoulli disposable email; legit={GEN['is_disposable_email']['legit']}, fraud={GEN['is_disposable_email']['fraud']}, softened by reveal/noise.",
    },
    "is_new_device": {
        "layer": "L3 Transaction Risk",
        "type": "int",
        "unit": "flag",
        "range": "{0,1}",
        "fraud_conditioned": "yes",
        "logic": f"Bernoulli new device; legit={GEN['is_new_device']['legit']}, fraud={GEN['is_new_device']['fraud']}, softened by reveal/noise.",
    },
    "shipping_billing_mismatch": {
        "layer": "L3 Transaction Risk",
        "type": "int",
        "unit": "flag",
        "range": "{0,1}",
        "fraud_conditioned": "yes",
        "logic": f"Bernoulli mismatch; legit={GEN['shipping_billing_mismatch']['legit']}, fraud={GEN['shipping_billing_mismatch']['fraud']}, softened by reveal/noise.",
    },
    "num_failed_payment_attempts": {
        "layer": "L3 Transaction Risk",
        "type": "int",
        "unit": "count",
        "range": ">=0",
        "fraud_conditioned": "yes",
        "logic": f"Poisson failed attempts; legit lambda={GEN['num_failed_payment_attempts']['legit']}, fraud lambda={GEN['num_failed_payment_attempts']['fraud']}, softened by reveal/noise.",
    },
    "browser": {
        "layer": "L3 Device",
        "type": "category",
        "unit": "-",
        "range": ", ".join(_BROWSERS),
        "fraud_conditioned": "no",
        "logic": "Uniform draw from common browsers for context.",
    },
    "device_os": {
        "layer": "L3 Device",
        "type": "category",
        "unit": "-",
        "range": ", ".join(_OS),
        "fraud_conditioned": "no",
        "logic": "Uniform draw from common operating systems for context.",
    },
    "device_id": {
        "layer": "L3 Device",
        "type": "string",
        "unit": "-",
        "range": "D########",
        "fraud_conditioned": "no",
        "logic": "Random device fingerprint id.",
    },
    "ip_billing_distance_km": {
        "layer": "L3 Transaction Risk",
        "type": "float",
        "unit": "km",
        "range": ">=0",
        "fraud_conditioned": "yes",
        "logic": f"Haversine distance from home to simulated IP; offset degrees legit={GEN['ip_offset_deg']['legit']}, fraud={GEN['ip_offset_deg']['fraud']}, softened by reveal/noise.",
    },
    "hour_of_day": {
        "layer": "L3 Time",
        "type": "int",
        "unit": "hour",
        "range": "0..23",
        "fraud_conditioned": "derived",
        "logic": "Derived from PaySim step modulo 24.",
    },
    "day_index": {
        "layer": "L3 Time",
        "type": "int",
        "unit": "day",
        "range": "0..30",
        "fraud_conditioned": "derived",
        "logic": "Derived from PaySim step // 24.",
    },
    "is_night": {
        "layer": "L3 Time",
        "type": "int",
        "unit": "flag",
        "range": "{0,1}",
        "fraud_conditioned": "derived",
        "logic": "1 when hour_of_day is 0..5.",
    },
    "account_txn_total": {
        "layer": "L3 Illustrative Velocity",
        "type": "int",
        "unit": "count",
        "range": ">=1",
        "fraud_conditioned": "no",
        "logic": "Total synthetic-customer transactions in the loaded dataset; illustrative because real nameOrig is mostly single-use.",
    },
    "account_txn_index": {
        "layer": "L3 Illustrative Velocity",
        "type": "int",
        "unit": "count",
        "range": ">=0",
        "fraud_conditioned": "no",
        "logic": "0-based transaction order for synthetic customer.",
    },
    "time_since_last_hours": {
        "layer": "L3 Illustrative Velocity",
        "type": "int",
        "unit": "hours",
        "range": "-1 or >=0",
        "fraud_conditioned": "no",
        "logic": "Hours since previous synthetic-customer transaction; -1 for first observed.",
    },
    "txn_count_last_24h": {
        "layer": "L3 Illustrative Velocity",
        "type": "int",
        "unit": "count",
        "range": ">=0",
        "fraud_conditioned": "no",
        "logic": "Prior synthetic-customer transactions in trailing 24 hours.",
    },
}


def get_customer_master(n_customers: int = GEN["n_customers"], seed: int = SEED, verbose: bool = True) -> pd.DataFrame:
    cache = DATA_SYNTH / f"customer_master_{n_customers}.parquet"
    if cache.exists():
        if verbose:
            print(f"[synth] customer master cache hit: {cache.name}")
        return pd.read_parquet(cache)

    if verbose:
        print(f"[synth] generating {n_customers:,} Faker identities")
    Faker.seed(seed)
    fake = Faker()
    rows = []
    for i in range(n_customers):
        lat, lng = fake.latlng()
        rows.append({
            "customer_id": f"U{i}",
            "customer_name": fake.name(),
            "email_handle": fake.user_name(),
            "billing_city": fake.city(),
            "home_lat": round(float(lat), 5),
            "home_lng": round(float(lng), 5),
        })
    master = pd.DataFrame(rows)
    master.to_parquet(cache, index=False)
    if verbose:
        print(f"[synth] cached identity master -> {cache.name}")
    return master


def _haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def _risk_signal_by_customer(cust_risky: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    reveal = GEN["reveal_rate"]
    noise = GEN["legit_noise"]
    return (cust_risky & (rng.random(len(cust_risky)) < reveal)) | ((~cust_risky) & (rng.random(len(cust_risky)) < noise))


def add_synthetic_context(df: pd.DataFrame, n_customers: int = GEN["n_customers"], seed: int = SEED, verbose: bool = True) -> pd.DataFrame:
    required = {"step", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PaySim synthetic context requires columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    n = len(df)
    fraud_mask = df["isFraud"].to_numpy(dtype=int) == 1
    base_cols = set(df.columns)
    out = df.copy()

    master = get_customer_master(n_customers, seed, verbose)
    n_customers = len(master)
    cust_idx = rng.integers(0, n_customers, size=n)
    out["customer_id"] = master["customer_id"].to_numpy()[cust_idx]

    cust_risky = np.zeros(n_customers, dtype=bool)
    np.logical_or.at(cust_risky, cust_idx, fraud_mask)
    cust_signal = _risk_signal_by_customer(cust_risky, rng)
    row_signal = cust_signal[cust_idx]

    mu = np.where(row_signal, GEN["account_age_days"]["fraud"][0], GEN["account_age_days"]["legit"][0])
    sigma = np.where(row_signal, GEN["account_age_days"]["fraud"][1], GEN["account_age_days"]["legit"][1])
    out["account_age_days"] = np.clip(np.round(rng.lognormal(mu, sigma)), 1, 3650).astype("int16")

    p_hr = np.where(row_signal, GEN["high_risk_country"]["fraud"], GEN["high_risk_country"]["legit"])
    is_hr = rng.random(n) < p_hr
    out["high_risk_country"] = is_hr.astype("int8")
    out["billing_country"] = np.where(
        is_hr,
        np.array(_HIGH_RISK)[rng.integers(0, len(_HIGH_RISK), n)],
        np.array(_LOW_RISK)[rng.integers(0, len(_LOW_RISK), n)],
    )

    p_disp = np.where(row_signal, GEN["is_disposable_email"]["fraud"], GEN["is_disposable_email"]["legit"])
    out["is_disposable_email"] = (rng.random(n) < p_disp).astype("int8")

    ident = master.iloc[cust_idx].reset_index(drop=True)
    out["customer_name"] = ident["customer_name"].to_numpy()
    out["billing_city"] = ident["billing_city"].to_numpy()
    home_lat = ident["home_lat"].to_numpy(dtype=float)
    home_lng = ident["home_lng"].to_numpy(dtype=float)
    domain = np.where(
        out["is_disposable_email"].to_numpy() == 1,
        np.array(_EMAIL_DISPOSABLE)[rng.integers(0, len(_EMAIL_DISPOSABLE), n)],
        np.array(_EMAIL_COMMON)[rng.integers(0, len(_EMAIL_COMMON), n)],
    )
    out["email"] = pd.Series(ident["email_handle"].to_numpy(), index=out.index) + "@" + domain

    reveal = GEN["reveal_rate"]
    noise = GEN["legit_noise"]

    def eff_mask():
        return (fraud_mask & (rng.random(n) < reveal)) | ((~fraud_mask) & (rng.random(n) < noise))

    def bern(field: str) -> np.ndarray:
        mask = eff_mask()
        p = np.where(mask, GEN[field]["fraud"], GEN[field]["legit"])
        return (rng.random(n) < p).astype("int8")

    out["is_new_device"] = bern("is_new_device")
    out["shipping_billing_mismatch"] = bern("shipping_billing_mismatch")

    fail_mask = eff_mask()
    lam = np.where(fail_mask, GEN["num_failed_payment_attempts"]["fraud"], GEN["num_failed_payment_attempts"]["legit"])
    out["num_failed_payment_attempts"] = rng.poisson(lam).astype("int16")

    out["browser"] = np.array(_BROWSERS)[rng.integers(0, len(_BROWSERS), n)]
    out["device_os"] = np.array(_OS)[rng.integers(0, len(_OS), n)]
    out["device_id"] = np.char.add("D", rng.integers(10**7, 10**8, n).astype(str))

    ip_mask = eff_mask()
    mu = np.where(ip_mask, GEN["ip_offset_deg"]["fraud"][0], GEN["ip_offset_deg"]["legit"][0])
    sigma = np.where(ip_mask, GEN["ip_offset_deg"]["fraud"][1], GEN["ip_offset_deg"]["legit"][1])
    offset = rng.lognormal(mu, sigma, n)
    angle = rng.uniform(0, 2 * np.pi, n)
    ip_lat = np.clip(home_lat + offset * np.sin(angle), -90, 90)
    ip_lng = home_lng + offset * np.cos(angle)
    out["ip_billing_distance_km"] = np.round(_haversine_km(home_lat, home_lng, ip_lat, ip_lng), 1).astype("float32")

    out["hour_of_day"] = (out["step"] % 24).astype("int8")
    out["day_index"] = (out["step"] // 24).astype("int16")
    out["is_night"] = out["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype("int8")

    out = _add_velocity_features(out)

    overlap = set(FIELD_META) & base_cols
    if overlap:
        raise AssertionError(f"Synthetic columns overlap base PaySim columns: {sorted(overlap)}")
    if verbose:
        added = [c for c in out.columns if c not in base_cols]
        print(f"[synth] added {len(added)} columns to {n:,} rows")
    return out


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    order = np.lexsort((df["step"].to_numpy(), df["customer_id"].astype(str).to_numpy()))
    cust = df["customer_id"].astype(str).to_numpy()[order]
    step = df["step"].to_numpy()[order]
    n = len(df)
    total = np.empty(n, np.int32)
    idx = np.empty(n, np.int32)
    since = np.empty(n, np.int32)
    cnt24 = np.empty(n, np.int32)

    i = 0
    while i < n:
        j = i
        while j < n and cust[j] == cust[i]:
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

    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    out = df.copy()
    out["account_txn_total"] = total[inv]
    out["account_txn_index"] = idx[inv]
    out["time_since_last_hours"] = since[inv]
    out["txn_count_last_24h"] = cnt24[inv]
    return out


if __name__ == "__main__":
    from data_base import load_base_data

    base = load_base_data(sample_frac=0.02)
    aug = add_synthetic_context(base)
    print("\nShape:", aug.shape)
    print(aug[["customer_name", "email", "billing_city", "billing_country", "ip_billing_distance_km"]].head())
