# Cleaning Report — E-Commerce Fraud Detection

Input rows: **954,393**, columns: **33**

## 1. Missing values

No missing values in any column. ✅

## 2. Duplicate base transactions

Exact duplicates on PaySim fields: **0**. None.

## 3. Invalid / zero amounts

- Negative amounts: **0** (removed)
- Zero amounts: **4** (kept — legal in PaySim, but flagged)

## 4. Amount outliers

99.9th percentile = 9,335,948. Rows above it: **955** (0.100%). **Kept** — extreme amounts are informative for fraud; we add a capped feature instead of dropping.

## 5. Known PaySim balance quirks (documented, NOT modified)

- Destination balances 0 before & after despite amount>0: **347,652** (merchant/mule accounts — expected).

- Rows where oldbalanceOrg − amount ≠ newbalanceOrig: **769,030** — this **balance-error is fraud signal**, kept as a feature.

## 6. Range validation (synthetic fields)

- hour_of_day ∈ 0..23: OK ✅

- account_age_days ≥ 1: OK ✅

- num_failed_payment_attempts ≥ 0: OK ✅

- ip_billing_distance_km ≥ 0: OK ✅

## Summary

- Rows before: **954,393** → after: **954,393** (removed 0, 0.000%).

- Decisions:
  - No rows removed; dataset already clean. Added flags: `flag_zero_amount`, `amount_capped`.
