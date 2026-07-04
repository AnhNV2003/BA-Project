# Data Dictionary — E-Commerce Fraud Detection

This dataset combines the **PaySim** base transactions (Kaggle:
`rupakroy/online-payments-fraud-detection-dataset`) with a **synthetic
contextual/behavioural layer** generated in `src/synth_context.py`.

- **Rows:** 6,362,620 (full) — modelling draft uses a stratified 15% sample.
- **Target:** `isFraud` (0/1). Prevalence ≈ **0.129%** (severe imbalance).
- **Reproducibility:** all generation is seeded (`SEED = 42` in `src/config.py`).

---

## 1. Base columns (PaySim — provided, not generated)

| Column | Type | Unit | Valid range | Description |
|---|---|---|---|---|
| `step` | int | hours | 1–743 | Simulation time; 1 step = 1 hour (~31 days total). |
| `type` | category | — | PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT | Transaction type. **Fraud only occurs in TRANSFER & CASH_OUT.** |
| `amount` | float | currency | ≥ 0 | Transaction amount. |
| `nameOrig` | string | — | `C########` | Origin customer id. **99.9% single-use** in real data. |
| `oldbalanceOrg` | float | currency | ≥ 0 | Origin balance before the transaction. |
| `newbalanceOrig` | float | currency | ≥ 0 | Origin balance after. For fraud, origin is typically drained to 0. |
| `nameDest` | string | — | `C…`/`M…` | Recipient id (`M…` = merchant, ~33.8% of rows). |
| `oldbalanceDest` | float | currency | ≥ 0 | Destination balance before. |
| `newbalanceDest` | float | currency | ≥ 0 | Destination balance after. |
| `isFraud` | int | — | {0,1} | **TARGET.** 1 = fraudulent transaction. |
| `isFlaggedFraud` | int | — | {0,1} | PaySim's naive rule flag (TRANSFER & amount > 200k). Rarely correct. |

---

## 2. Synthetic columns (generated)

Generation architecture (see `src/synth_context.py` docstring):
**L1 Identity** (Faker, per customer, cached) · **L2 Account risk** (numpy, per
customer, conditioned on customer risk) · **L3 Transaction risk** (numpy, per
transaction, conditioned on `isFraud`). All conditional distributions **differ
between fraud/legit but overlap heavily** — no field separates the classes
perfectly (verified in the modelling step: synthetic-only AUC < 1.0).

| Column | Layer | Type | Unit | Valid range | Fraud-cond.? | Generation logic / business assumption |
|---|---|---|---|---|---|---|
| `customer_id` | L1 | string | — | `U0…U199999` | no | Assigned from a fixed pool of 200,000 synthetic customers (real `nameOrig` is single-use, so it cannot carry account history). |
| `customer_name` | L1 | string | — | — | no | Faker `name()`. Identity/demo only — **not a model feature.** |
| `email` | L1+L2 | string | — | `handle@domain` | via disposable | Faker `user_name()` handle + domain: disposable pool if `is_disposable_email` else common provider. |
| `billing_city` | L1 | string | — | — | no | Faker `city()`. Context/demo only. |
| `account_age_days` | L2 | int | days | 1–3650 | yes | `lognormal(μ,σ)`; legit μ=6.0,σ=0.9 (~median 400d), fraud μ=4.2,σ=1.0 (~median 66d). Fraudsters skew to **young accounts**. |
| `billing_country` | L2 | category | — | 12 ISO codes | yes | High-risk set {NG,RU,CN,ID} chosen with P=0.35 for risky customers vs 0.05 legit; else a low-risk country. |
| `high_risk_country` | L2 | int | — | {0,1} | yes | 1 if `billing_country` ∈ {NG,RU,CN,ID}. Platform's elevated-risk list (illustrative). |
| `is_disposable_email` | L2 | int | — | {0,1} | yes | Bernoulli: P=0.30 risky vs 0.03 legit. Throwaway email = higher risk. |
| `is_new_device` | L3 | int | — | {0,1} | yes | Bernoulli: P=0.55 fraud vs 0.12 legit. Device never seen on this account. |
| `shipping_billing_mismatch` | L3 | int | — | {0,1} | yes | Bernoulli: P=0.45 fraud vs 0.06 legit. Ship-to ≠ bill-to. |
| `num_failed_payment_attempts` | L3 | int | count | ≥ 0 | yes | `Poisson(λ)`; λ=2.2 fraud vs 0.25 legit. Card-testing behaviour. |
| `ip_billing_distance_km` | L3 | float | km | ≥ 0 | yes | Haversine from the customer's home lat/lng to a transaction IP location; IP offset ~`lognormal`, fraud μ=2.3 vs legit μ=−1.5 (degrees). Geo-inconsistency signal. |
| `browser` | L3 | category | — | 6 values | no | Uniform over common browsers. Context only. |
| `device_os` | L3 | category | — | 5 values | no | Uniform over common OSes. Context only. |
| `device_id` | L3 | string | — | `D#######` | no | Random device fingerprint id. |
| `hour_of_day` | L3 | int | hour | 0–23 | derived | `step % 24`. |
| `day_index` | L3 | int | day | 0–30 | derived | `step // 24`. |
| `is_night` | L3 | int | — | {0,1} | derived | 1 if `hour_of_day` ∈ 0–5. Night-time transactions. |
| `account_txn_total` | L3 | int | count | ≥ 1 | no | Total transactions by this customer in the dataset (behavioural). |
| `account_txn_index` | L3 | int | count | ≥ 0 | no | 0-based order of this transaction within the customer's history. |
| `time_since_last_hours` | L3 | int | hours | −1 or ≥ 0 | no | Hours since the customer's previous transaction (−1 = first seen). |
| `txn_count_last_24h` | L3 | int | count | ≥ 0 | no | Customer's prior transactions in the preceding 24h (velocity). |

---

## 3. Design notes & known limitations (state these in the report)

1. **No leakage by construction.** Every conditional field overlaps across
   classes; the modelling step confirms a model trained on synthetic features
   *alone* achieves high-but-not-perfect AUC (no single-feature separation).
2. **Velocity is weak in this base.** Because real `nameOrig` is single-use, we
   synthesised a customer pool to make velocity computable, but fraud accounts
   are young ⇒ few prior transactions, so `txn_count_last_24h` carries little
   signal here. Re-evaluate after any change to the assignment model.
3. **Balances dominate on PaySim.** The base balance-error signature
   (origin drained to 0) is highly predictive on its own; synthetic context adds
   realism and robustness (and matters when balance info is unavailable), but is
   not expected to dominate feature importance.
4. **Prevalence preserved.** The modelling sample keeps the real ~0.129% fraud
   rate; class imbalance is handled at the modelling stage (class weights /
   SMOTE), never by distorting the base prevalence.
5. **Parameters are tunable** in `GEN` (`src/synth_context.py`) — this table is
   kept in sync with that single source of truth.
