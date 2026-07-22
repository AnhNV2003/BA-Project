# EDA Summary — PaySim E-Commerce Fraud Detection

## Dataset Snapshot

- Rows: **100,000**.

- Fraud rows: **150** (**0.1500%**).

- Total fraudulent amount: **9,428,249.55**.

- Step range: **1..743** hours.

- Synthetic/context sections use processed sample: **100,000** rows, **150** fraud rows.

- Raw-sensitive sections (`type`, `amount`, balance, `nameDest` reuse) use the full PaySim CSV to avoid sampling-compressed reuse counts.


## Fraud by Transaction Type

Fraud appears only in `TRANSFER` and `CASH_OUT` in PaySim.

| type     |   fraud |   count |   fraud_rate_pct |
|:---------|--------:|--------:|-----------------:|
| CASH_OUT |     126 |   35325 |           0.3567 |
| TRANSFER |      24 |    8422 |           0.285  |
| CASH_IN  |       0 |   21963 |           0      |
| DEBIT    |       0 |     600 |           0      |
| PAYMENT  |       0 |   33690 |           0      |


## Amount Distribution / Outliers

|       |    amount |
|------:|----------:|
| 0.5   |   3025.04 |
| 0.9   |  18022.5  |
| 0.99  |  78355.8  |
| 0.999 | 237029    |

- Max amount: **3,238,389.08**.

- Zero-amount rows in full raw data: **0**.

| amount_decile          |   sum |   count |   fraud_rate_pct |
|:-----------------------|------:|--------:|-----------------:|
| (3.009, 500.11]        |     0 |   10001 |             0    |
| (500.11, 932.526]      |     0 |    9999 |             0    |
| (932.526, 1446.714]    |     0 |   10000 |             0    |
| (1446.714, 2113.452]   |     2 |   10000 |             0.02 |
| (2113.452, 3025.035]   |     0 |   10000 |             0    |
| (3025.035, 4301.704]   |     0 |   10000 |             0    |
| (4301.704, 6254.654]   |     1 |   10000 |             0.01 |
| (6254.654, 9707.726]   |     5 |   10000 |             0.05 |
| (9707.726, 18022.49]   |    22 |   10000 |             0.22 |
| (18022.49, 3238389.08] |   120 |   10000 |             1.2  |


## Time Windows

- `hour_of_day` and `day_index` are derived from PaySim `step` using the same M1 formula.

- Late simulation days can have tiny volume, so extreme daily fraud rates should be treated as a simulation quirk.

|   day_index |   fraud |   count |   fraud_rate_pct |
|------------:|--------:|--------:|-----------------:|
|          30 |       6 |    3233 |           0.1856 |


## Synthetic Risk Flags

| signal                    |   rate_if_0_pct |   rate_if_1_pct |
|:--------------------------|----------------:|----------------:|
| is_new_device             |          0.1241 |          0.2845 |
| shipping_billing_mismatch |          0.1193 |          0.4432 |
| is_disposable_email       |          0.1343 |          0.4697 |
| high_risk_country         |          0.1276 |          0.4596 |
| is_night                  |          0.1374 |          0.1876 |


## Balance Signature

PaySim balance reconciliation is very predictive, so later modelling should also test a realistic feature set without post-transaction balance leakage-like signals.

| feature              |   directionless_auc |
|:---------------------|--------------------:|
| abs_errorBalanceDest |              0.9793 |
| orig_drained         |              0.8843 |
| abs_errorBalanceOrig |              0.8361 |
| dest_was_empty       |              0.8313 |

| signal         |   legit_rate_pct |   fraud_rate_pct |
|:---------------|-----------------:|-----------------:|
| orig_drained   |          23.1387 |              100 |
| dest_was_empty |          33.7406 |              100 |


## Destination Reuse / Mule Pattern

- This section is computed on the full raw PaySim file. A 15% row sample compresses cross-row reuse and undercounts max/repeated `nameDest` values.

- Unique destination accounts: **14,975**.

- Destination accounts with at least 2 transactions: **14,851**.

- Destination accounts with at least 4 transactions: **13,497**.

- Max transactions for one destination: **19**.

- This supports M4 historical aggregation on `nameDest` using past transactions only.

| reuse_bucket   |   destinations |   total_txns |   fraud_destinations |   fraud_txns |   avg_senders |   fraud_dest_rate_pct |
|:---------------|---------------:|-------------:|---------------------:|-------------:|--------------:|----------------------:|
| 1              |            124 |          124 |                    0 |            0 |       1       |                0      |
| 2-3            |           1354 |         3649 |                    6 |            6 |       2.69498 |                0.4431 |
| 4-10           |          12366 |        82695 |                  115 |          118 |       6.68696 |                0.93   |
| 11+            |           1131 |        13532 |                   25 |           26 |      11.9637  |                2.2104 |

|   dest_is_customer |   sum |   count |      mean |   fraud_rate_pct |
|-------------------:|------:|--------:|----------:|-----------------:|
|                  0 |     0 |   33690 | 0         |           0      |
|                  1 |   150 |   66310 | 0.0022621 |           0.2262 |


## Channels / Context

### browser

| browser          |   fraud |   count |   fraud_rate_pct |
|:-----------------|--------:|--------:|-----------------:|
| Opera            |      30 |   16702 |           0.1796 |
| Safari           |      28 |   16794 |           0.1667 |
| Firefox          |      26 |   16644 |           0.1562 |
| Samsung Internet |      25 |   16747 |           0.1493 |
| Chrome           |      21 |   16574 |           0.1267 |
| Edge             |      20 |   16539 |           0.1209 |

### device_os

| device_os   |   fraud |   count |   fraud_rate_pct |
|:------------|--------:|--------:|-----------------:|
| Linux       |      36 |   20180 |           0.1784 |
| Windows     |      30 |   19949 |           0.1504 |
| Android     |      29 |   20034 |           0.1448 |
| macOS       |      28 |   19944 |           0.1404 |
| iOS         |      27 |   19893 |           0.1357 |

### billing_country

| billing_country   |   fraud |   count |   fraud_rate_pct |
|:------------------|--------:|--------:|-----------------:|
| NG                |      17 |    1691 |           1.0053 |
| ID                |       7 |    1676 |           0.4177 |
| RU                |       4 |    1692 |           0.2364 |
| CN                |       3 |    1686 |           0.1779 |
| BR                |      19 |   11676 |           0.1627 |
| PH                |      18 |   11537 |           0.156  |
| VN                |      17 |   11726 |           0.145  |
| IN                |      17 |   11868 |           0.1432 |
| GB                |      15 |   11617 |           0.1291 |
| FR                |      12 |   11566 |           0.1038 |
| DE                |      12 |   11606 |           0.1034 |
| US                |       9 |   11659 |           0.0772 |


## Top Numeric Correlations with Target

|                             |   pearson_r |
|:----------------------------|------------:|
| abs_errorBalanceDest        |      0.1906 |
| errorBalanceDest            |      0.1906 |
| amount                      |      0.0862 |
| log_amount                  |      0.0718 |
| orig_drained                |      0.0704 |
| oldbalanceOrg               |      0.06   |
| dest_was_empty              |      0.0542 |
| isFlaggedFraud              |      0.0449 |
| num_failed_payment_attempts |      0.041  |
| ip_billing_distance_km      |      0.0316 |
| dest_is_customer            |      0.0276 |
| shipping_billing_mismatch   |      0.0245 |
| high_risk_country           |      0.0215 |
| is_disposable_email         |      0.0183 |
| newbalanceOrig              |     -0.0178 |
