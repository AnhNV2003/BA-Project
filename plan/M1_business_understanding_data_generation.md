# Plan: Task 1 (Module 1) — Business Understanding & Data Generation trên PaySim

## Context

Đề bài dùng PaySim ("Online Payments Fraud Detection Dataset",
Kaggle `rupakroy/online-payments-fraud-detection-dataset`) làm base dataset.

- Raw file: `data/online-payments-fraud-detection-dataset/PS_20174392719_1491204439457_log.csv`
- Full size: 6,362,620 transactions
- Raw schema: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`
- Fraud rows: 8,213
- Fraud rate: approximately 0.129%
- Fraud appears only in `TRANSFER` and `CASH_OUT`

## Module 1 Requirements

1. Business understanding: define the fraud-detection business problem and KPIs.
2. Generate synthetic contextual data using Python/Faker.
3. Produce a complete data dictionary for raw and synthetic fields.

Required KPIs:

- Fraud rate
- False-positive rate
- Financial loss avoided

Required synthetic context examples:

- customer account age
- device/browser fingerprint
- shipping-vs-billing mismatch
- failed payment attempts
- IP-to-billing distance
- time-of-day pattern

## Active Files

| File | Role |
|---|---|
| `src/config.py` | PaySim paths, schema, target, and business-cost assumptions |
| `src/data_base.py` | Load the real PaySim CSV, validate schema, and stratified-sample by target |
| `src/synth_context.py` | Add Faker identity, account-risk, transaction-risk, time, device, and illustrative velocity context |
| `src/build_dataset.py` | Orchestrate load → synth context → write processed artifacts |
| `src/make_data_dictionary.py` | Generate/check data dictionary from the implemented schema |
| `src/verify_task1.py` | Verify M1 output quality and basic anti-leakage checks |
| `docs/business_understanding.md` | Business problem, KPIs, and cost model |
| `docs/data_dictionary.md` | Raw + synthetic field definitions |

## Design

### 1. Load PaySim

`src/data_base.py` reads the configured CSV path first:

`data/online-payments-fraud-detection-dataset/PS_20174392719_1491204439457_log.csv`

It validates the required PaySim columns and uses a 15% stratified sample by
default. This keeps the natural fraud prevalence while avoiding a slow default
build. Use `python src/build_dataset.py --full` only when a full 6.36M-row
artifact is required.

### 2. Add Synthetic Context

The raw PaySim table contains transaction and balance fields but lacks
e-commerce context. `src/synth_context.py` adds:

- identity/display context: `customer_id`, `customer_name`, `email`, `billing_city`
- account risk: `account_age_days`, `billing_country`, `high_risk_country`, `is_disposable_email`
- device/risk context: `is_new_device`, `shipping_billing_mismatch`, `num_failed_payment_attempts`, `browser`, `device_os`, `device_id`, `ip_billing_distance_km`
- time context: `hour_of_day`, `day_index`, `is_night`
- illustrative synthetic-customer velocity: `account_txn_total`, `account_txn_index`, `time_since_last_hours`, `txn_count_last_24h`

Important constraint: real `nameOrig` is almost always single-use, so
origin-account velocity is not treated as a strong real PaySim signal. Repeated
`nameDest` behavior is a better candidate for later feature engineering, but
that belongs to M4 rather than synthetic data generation.

### 3. Business Understanding

`docs/business_understanding.md` defines:

- business problem: flag online payment fraud while controlling customer friction
- fraud rate KPI
- false-positive rate KPI
- financial loss avoided KPI
- cost assumptions for missed fraud, false alarms, and manual review

### 4. Data Dictionary

`docs/data_dictionary.md` must document:

- every raw PaySim field
- every synthetic field
- type, unit, valid range, generation logic, and business assumption

`src/make_data_dictionary.py --check` verifies that documented synthetic fields
match the processed dataset.

## Outputs

- `data/processed/transactions_context.parquet`
- `data/processed/sample_preview.csv`
- `docs/business_understanding.md`
- `docs/data_dictionary.md`

## Verification

Run:

```bash
python src/build_dataset.py
python src/make_data_dictionary.py --check
python src/verify_task1.py
```

Expected checks:

- raw PaySim schema is complete
- default sample preserves fraud prevalence around 0.129%
- synthetic context columns are present
- synthetic context columns have valid ranges and no unexpected nulls
- individual synthetic risk flags do not perfectly reveal the target

## Notes for Later Modules

For M4, create historical features from repeated `nameDest` accounts using only
past transactions at scoring time. Candidate features include:

- destination seen before
- destination transaction count so far
- destination total/mean amount so far
- number of unique senders so far
- transaction count by type so far

These are feature-engineering tasks, not Module 1 synthetic-data tasks.
