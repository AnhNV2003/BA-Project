# Plan: Task 2 (Module 2) — Exploratory Data Analysis trên PaySim

## Context

M1 sinh `data/processed/transactions_context.parquet` từ PaySim base cộng
synthetic contextual data. M2 khám phá dữ liệu để định hướng M3 cleaning và M4
feature engineering.

Raw PaySim facts cần kiểm chứng lại trong EDA:

- 6.36M dòng ở raw full dataset; processed default là 15% stratified sample.
- Fraud rate khoảng 0.129%.
- Fraud chỉ xuất hiện ở `TRANSFER` và `CASH_OUT`.
- `nameOrig` gần như single-use, nên originator velocity không phải signal thật mạnh.
- `nameDest` lặp nhiều hơn, phù hợp để tạo historical destination features ở M4.
- Balance reconciliation signals rất mạnh trong PaySim, nên model sau này cần so sánh thêm realistic feature set.

## M2 Requirements

1. Phân tích distributions, outliers, correlations.
2. Định lượng class imbalance và fraud rate theo transaction types, channels, time windows.
3. Visualize fraud patterns: amount vs fraud likelihood, balance signature, geographic/context mismatch, destination reuse.

## Active File

| File | Role |
|---|---|
| `src/eda.py` | Load `transactions_context.parquet`, sinh figures vào `docs/figures/paysim/`, ghi `docs/eda_summary.md` |

## Figures

Expected output figures:

- `01_class_imbalance.png`
- `02_fraud_by_type.png`
- `03_amount_by_class.png`
- `04_amount_decile_fraud_rate.png`
- `05_time_windows.png`
- `06_synthetic_risk_flags.png`
- `07_numeric_context_by_class.png`
- `08_balance_signature.png`
- `09_destination_reuse_pattern.png`
- `10_context_channels.png`
- `11_corr_with_target.png`

## Analysis Checklist

### Distributions / Outliers / Correlations

- Amount distribution by class using `log1p(amount)`.
- Amount quantiles p50/p90/p99/p99.9 and max.
- Zero-amount count.
- Numeric correlations with `isFraud`.

### Imbalance / Type / Channel / Time

- Class imbalance count on log scale.
- Fraud rate by `type`.
- Fraud rate by synthetic channels: browser, device OS, billing country.
- Fraud rate by synthetic flags: new device, mismatch, disposable email, high-risk country, night.
- Fraud rate by `hour_of_day` and `day_index`.

### Fraud Patterns

- Amount decile vs fraud rate.
- Balance signature: `errorBalanceOrig`, `errorBalanceDest`, `orig_drained`, `dest_was_empty`.
- Destination reuse pattern for `nameDest`.
- Context/geographic mismatch: `ip_billing_distance_km`, `shipping_billing_mismatch`, `high_risk_country`.

## Outputs

- `docs/figures/paysim/*.png`
- `docs/eda_summary.md`

## Verification

Run:

```bash
python src/eda.py
```

Expected sanity checks:

- `docs/eda_summary.md` exists and title is PaySim.
- Figures are written under `docs/figures/paysim/`.
- Fraud rate in summary is approximately 0.129% for the default stratified sample.
- Fraud-by-type table shows fraud only in `TRANSFER` and `CASH_OUT`.
- Destination reuse section reports repeated `nameDest` accounts and recommends historical aggregation in M4.

## Notes for Later Modules

- M2 is descriptive only; no data mutation.
- M3 should validate/clean PaySim without destroying balance signals.
- M4 should implement `nameDest` historical features using past transactions only.
- M5 should compare full PaySim features against a realistic feature set that excludes post-transaction balance reconciliation signals.
