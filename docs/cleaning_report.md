# Cleaning Report — PaySim E-Commerce Fraud Detection

## Scope

- Input: `data/processed/transactions_context.parquet`.

- Cleaning is conservative: validate and document, remove only exact duplicates or negative amounts if found.

- PaySim balance-reconciliation quirks are preserved because they are predictive signal.


## Before / After Summary

| metric               |         before |          after |   change |
|:---------------------|---------------:|---------------:|---------:|
| rows                 |    6.36262e+06 |    6.36262e+06 |        0 |
| columns              |   33           |   35           |        2 |
| fraud_rows           | 8213           | 8213           |        0 |
| fraud_rate           |    0.00129082  |    0.00129082  |        0 |
| missing_cells        |    0           |    0           |        0 |
| duplicate_base_rows  |    0           |    0           |        0 |
| negative_amount_rows |    0           |    0           |        0 |
| zero_amount_rows     |   16           |   16           |        0 |
| step_outside_1_743   |    0           |    0           |        0 |


## Missing Values

Before:

| column                      |   missing |   missing_pct |
|:----------------------------|----------:|--------------:|
| step                        |         0 |             0 |
| type                        |         0 |             0 |
| amount                      |         0 |             0 |
| nameOrig                    |         0 |             0 |
| oldbalanceOrg               |         0 |             0 |
| newbalanceOrig              |         0 |             0 |
| nameDest                    |         0 |             0 |
| oldbalanceDest              |         0 |             0 |
| newbalanceDest              |         0 |             0 |
| isFraud                     |         0 |             0 |
| isFlaggedFraud              |         0 |             0 |
| customer_id                 |         0 |             0 |
| account_age_days            |         0 |             0 |
| high_risk_country           |         0 |             0 |
| billing_country             |         0 |             0 |
| is_disposable_email         |         0 |             0 |
| customer_name               |         0 |             0 |
| billing_city                |         0 |             0 |
| email                       |         0 |             0 |
| is_new_device               |         0 |             0 |
| shipping_billing_mismatch   |         0 |             0 |
| num_failed_payment_attempts |         0 |             0 |
| browser                     |         0 |             0 |
| device_os                   |         0 |             0 |
| device_id                   |         0 |             0 |
| ip_billing_distance_km      |         0 |             0 |
| hour_of_day                 |         0 |             0 |
| day_index                   |         0 |             0 |
| is_night                    |         0 |             0 |
| account_txn_total           |         0 |             0 |
| account_txn_index           |         0 |             0 |
| time_since_last_hours       |         0 |             0 |
| txn_count_last_24h          |         0 |             0 |

After:

| column                      |   missing |   missing_pct |
|:----------------------------|----------:|--------------:|
| step                        |         0 |             0 |
| type                        |         0 |             0 |
| amount                      |         0 |             0 |
| nameOrig                    |         0 |             0 |
| oldbalanceOrg               |         0 |             0 |
| newbalanceOrig              |         0 |             0 |
| nameDest                    |         0 |             0 |
| oldbalanceDest              |         0 |             0 |
| newbalanceDest              |         0 |             0 |
| isFraud                     |         0 |             0 |
| isFlaggedFraud              |         0 |             0 |
| customer_id                 |         0 |             0 |
| account_age_days            |         0 |             0 |
| high_risk_country           |         0 |             0 |
| billing_country             |         0 |             0 |
| is_disposable_email         |         0 |             0 |
| customer_name               |         0 |             0 |
| billing_city                |         0 |             0 |
| email                       |         0 |             0 |
| is_new_device               |         0 |             0 |
| shipping_billing_mismatch   |         0 |             0 |
| num_failed_payment_attempts |         0 |             0 |
| browser                     |         0 |             0 |
| device_os                   |         0 |             0 |
| device_id                   |         0 |             0 |
| ip_billing_distance_km      |         0 |             0 |
| hour_of_day                 |         0 |             0 |
| day_index                   |         0 |             0 |
| is_night                    |         0 |             0 |
| account_txn_total           |         0 |             0 |
| account_txn_index           |         0 |             0 |
| time_since_last_hours       |         0 |             0 |
| txn_count_last_24h          |         0 |             0 |
| flag_zero_amount            |         0 |             0 |
| amount_capped               |         0 |             0 |


## Duplicate Base Transactions

- Exact duplicates on PaySim base columns: **0**.


## Category Validation

Expected sets:

| column          | valid_values                                           |
|:----------------|:-------------------------------------------------------|
| type            | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER            |
| browser         | Chrome, Safari, Edge, Firefox, Samsung Internet, Opera |
| device_os       | Windows, Android, iOS, macOS, Linux                    |
| billing_country | US, GB, DE, FR, VN, IN, BR, NG, RU, CN, ID, PH         |

Before category value-set:

| column          |   nunique | values                                                 |
|:----------------|----------:|:-------------------------------------------------------|
| type            |         5 | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER            |
| browser         |         6 | Chrome, Edge, Firefox, Opera, Safari, Samsung Internet |
| device_os       |         5 | Android, Linux, Windows, iOS, macOS                    |
| billing_country |        12 | BR, CN, DE, FR, GB, ID, IN, NG, PH, RU, US, VN         |

After category value-set:

| column          |   nunique | values                                                 |
|:----------------|----------:|:-------------------------------------------------------|
| type            |         5 | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER            |
| browser         |         6 | Chrome, Edge, Firefox, Opera, Safari, Samsung Internet |
| device_os       |         5 | Android, Linux, Windows, iOS, macOS                    |
| billing_country |        12 | BR, CN, DE, FR, GB, ID, IN, NG, PH, RU, US, VN         |

### `type` invalid values

No inconsistency found.

### `browser` invalid values

No inconsistency found.

### `device_os` invalid values

No inconsistency found.

### `billing_country` invalid values

No inconsistency found.


## Amount / Step Validation

- Negative amount rows: **0**.

- Zero amount rows: **16**; kept and flagged with `flag_zero_amount`.

- Rows outside `step` range 1..743: **0**.

| metric          |           value |
|:----------------|----------------:|
| amount_p99      |     1.61598e+06 |
| amount_p999     |     8.9568e+06  |
| amount_max      |     9.24455e+07 |
| rows_above_p99  | 63627           |
| rows_above_p999 |  6363           |


## Balance Reconciliation Quirks

| check                                                |    rows | decision   |
|:-----------------------------------------------------|--------:|:-----------|
| destination balances 0 before/after while amount > 0 | 2317276 | preserve   |
| origin balance reconciliation error                  | 5125552 | preserve   |
| destination balance reconciliation error             | 4364924 | preserve   |


## Synthetic Field Range Validation

| check                            | status   |
|:---------------------------------|:---------|
| hour_of_day in 0..23             | OK       |
| account_age_days >= 1            | OK       |
| num_failed_payment_attempts >= 0 | OK       |
| ip_billing_distance_km >= 0      | OK       |


## Decisions

- No duplicate base transactions found.
- No category normalization or invalid-category repair was needed.
- No negative amount rows found.
- Kept zero-amount rows and added `flag_zero_amount`.
- Kept amount outliers and added `amount_capped` at p99.9 for optional robust modelling.
- All `step` values are inside expected PaySim range 1..743.
- Preserved balance-reconciliation quirks as predictive PaySim signal.
