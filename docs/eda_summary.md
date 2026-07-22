# EDA Summary — PaySim E-Commerce Fraud Detection

## Dataset Snapshot

- Rows: **6,362,620**.

- Fraud rows: **8,213** (**0.1291%**).

- Total fraudulent amount: **12,056,415,427.84**.

- Step range: **1..743** hours.

- Synthetic/context sections use processed sample: **6,362,620** rows, **8,213** fraud rows.

- Raw-sensitive sections (`type`, `amount`, balance, `nameDest` reuse) use the full PaySim CSV to avoid sampling-compressed reuse counts.


## Fraud by Transaction Type

Fraud appears only in `TRANSFER` and `CASH_OUT` in PaySim.

| type     |   fraud |            count |   fraud_rate_pct |
|:---------|--------:|-----------------:|-----------------:|
| TRANSFER |    4097 | 532909           |           0.7688 |
| CASH_OUT |    4116 |      2.2375e+06  |           0.184  |
| CASH_IN  |       0 |      1.39928e+06 |           0      |
| DEBIT    |       0 |  41432           |           0      |
| PAYMENT  |       0 |      2.1515e+06  |           0      |


## Amount Distribution / Outliers

|       |           amount |
|------:|-----------------:|
| 0.5   |  74871.9         |
| 0.9   | 365423           |
| 0.99  |      1.61598e+06 |
| 0.999 |      8.9568e+06  |

- Max amount: **92,445,516.64**.

- Zero-amount rows in full raw data: **16**.

| amount_decile             |   sum |   count |   fraud_rate_pct |
|:--------------------------|------:|--------:|-----------------:|
| (-0.001, 4501.3]          |   148 |  636263 |           0.0233 |
| (4501.3, 9866.158]        |   128 |  636261 |           0.0201 |
| (9866.158, 18092.028]     |   148 |  636262 |           0.0233 |
| (18092.028, 36371.35]     |   365 |  636262 |           0.0574 |
| (36371.35, 74871.94]      |   617 |  636262 |           0.097  |
| (74871.94, 122563.784]    |   592 |  636262 |           0.093  |
| (122563.784, 176801.919]  |   581 |  636262 |           0.0913 |
| (176801.919, 246611.22]   |   503 |  636262 |           0.0791 |
| (246611.22, 365423.309]   |   719 |  636262 |           0.113  |
| (365423.309, 92445516.64] |  4412 |  636262 |           0.6934 |


## Time Windows

- `hour_of_day` and `day_index` are derived from PaySim `step` using the same M1 formula.

- Late simulation days can have tiny volume, so extreme daily fraud rates should be treated as a simulation quirk.

|   day_index |   fraud |   count |   fraud_rate_pct |
|------------:|--------:|--------:|-----------------:|
|          30 |     282 |     282 |              100 |


## Synthetic Risk Flags

| signal                    |   rate_if_0_pct |   rate_if_1_pct |
|:--------------------------|----------------:|----------------:|
| is_new_device             |          0.1018 |          0.2678 |
| shipping_billing_mismatch |          0.1059 |          0.3533 |
| is_disposable_email       |          0.1164 |          0.3649 |
| high_risk_country         |          0.114  |          0.3202 |
| is_night                  |          0.0995 |          1.7726 |


## Balance Signature

PaySim balance reconciliation is very predictive, so later modelling should also test a realistic feature set without post-transaction balance leakage-like signals.

| feature              |   directionless_auc |
|:---------------------|--------------------:|
| abs_errorBalanceOrig |              0.9208 |
| orig_drained         |              0.8687 |
| dest_was_empty       |              0.6134 |
| abs_errorBalanceDest |              0.5476 |

| signal         |   legit_rate_pct |   fraud_rate_pct |
|:---------------|-----------------:|-----------------:|
| orig_drained   |          23.8035 |          97.5527 |
| dest_was_empty |          42.475  |          65.1528 |


## Destination Reuse / Mule Pattern

- This section is computed on the full raw PaySim file. A 15% row sample compresses cross-row reuse and undercounts max/repeated `nameDest` values.

- Unique destination accounts: **2,722,362**.

- Destination accounts with at least 2 transactions: **459,658**.

- Destination accounts with at least 4 transactions: **325,315**.

- Max transactions for one destination: **113**.

- This supports M4 historical aggregation on `nameDest` using past transactions only.

| reuse_bucket   |    destinations |       total_txns |   fraud_destinations |   fraud_txns |   avg_senders |   fraud_dest_rate_pct |
|:---------------|----------------:|-----------------:|---------------------:|-------------:|--------------:|----------------------:|
| 1              |      2.2627e+06 |      2.2627e+06  |                 2673 |         2673 |       1       |                0.1181 |
| 2-3            | 134343          | 325975           |                 1260 |         1265 |       2.42644 |                0.9379 |
| 4-10           | 195029          |      1.23398e+06 |                 2031 |         2042 |       6.32718 |                1.0414 |
| 11+            | 130286          |      2.53996e+06 |                 2205 |         2233 |      19.4952  |                1.6924 |

|   dest_is_customer |   sum |       count |       mean |   fraud_rate_pct |
|-------------------:|------:|------------:|-----------:|-----------------:|
|                  0 |     0 | 2.1515e+06  | 0          |            0     |
|                  1 |  8213 | 4.21112e+06 | 0.00195031 |            0.195 |


## Channels / Context

### browser

| browser          |   fraud |       count |   fraud_rate_pct |
|:-----------------|--------:|------------:|-----------------:|
| Chrome           |    1437 | 1.06126e+06 |           0.1354 |
| Safari           |    1432 | 1.06028e+06 |           0.1351 |
| Opera            |    1362 | 1.06113e+06 |           0.1284 |
| Firefox          |    1360 | 1.06119e+06 |           0.1282 |
| Samsung Internet |    1322 | 1.05947e+06 |           0.1248 |
| Edge             |    1300 | 1.05929e+06 |           0.1227 |

### device_os

| device_os   |   fraud |       count |   fraud_rate_pct |
|:------------|--------:|------------:|-----------------:|
| macOS       |    1672 | 1.27317e+06 |           0.1313 |
| Windows     |    1644 | 1.27278e+06 |           0.1292 |
| Linux       |    1642 | 1.27174e+06 |           0.1291 |
| iOS         |    1641 | 1.27315e+06 |           0.1289 |
| Android     |    1614 | 1.27178e+06 |           0.1269 |

### billing_country

| billing_country   |   fraud |   count |   fraud_rate_pct |
|:------------------|--------:|--------:|-----------------:|
| NG                |     378 |  116373 |           0.3248 |
| ID                |     376 |  116675 |           0.3223 |
| CN                |     371 |  116462 |           0.3186 |
| RU                |     368 |  116727 |           0.3153 |
| PH                |     874 |  736482 |           0.1187 |
| DE                |     858 |  737319 |           0.1164 |
| GB                |     855 |  736409 |           0.1161 |
| IN                |     851 |  737772 |           0.1153 |
| BR                |     838 |  736880 |           0.1137 |
| US                |     834 |  736283 |           0.1133 |
| VN                |     808 |  737735 |           0.1095 |
| FR                |     802 |  737503 |           0.1087 |


## Top Numeric Correlations with Target

|                             |   pearson_r |
|:----------------------------|------------:|
| amount                      |      0.0767 |
| orig_drained                |      0.0621 |
| is_night                    |      0.0614 |
| errorBalanceDest            |      0.0551 |
| abs_errorBalanceDest        |      0.0538 |
| isFlaggedFraud              |      0.0441 |
| time_since_last_hours       |      0.0407 |
| log_amount                  |      0.0406 |
| day_index                   |      0.0326 |
| num_failed_payment_attempts |      0.0318 |
| step                        |      0.0316 |
| hour_of_day                 |     -0.0314 |
| dest_is_customer            |      0.0257 |
| ip_billing_distance_km      |      0.0249 |
| shipping_billing_mismatch   |      0.0201 |
