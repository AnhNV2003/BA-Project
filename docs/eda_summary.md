# EDA Summary — PaySim E-Commerce Fraud Detection

## Dataset Snapshot

- Rows: **6,362,620**.

- Fraud rows: **8,213** (**0.1291%**).

- Total fraudulent amount: **12,056,415,427.84**.

- Step range: **1..743** hours.

- Synthetic/context sections use processed sample: **954,393** rows, **1,232** fraud rows.

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
| is_new_device             |          0.1014 |          0.2699 |
| shipping_billing_mismatch |          0.1058 |          0.3537 |
| is_disposable_email       |          0.1164 |          0.3801 |
| high_risk_country         |          0.1154 |          0.3121 |
| is_night                  |          0.1004 |          1.7446 |


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

| browser          |   fraud |   count |   fraud_rate_pct |
|:-----------------|--------:|--------:|-----------------:|
| Safari           |     229 |  158737 |           0.1443 |
| Samsung Internet |     206 |  158795 |           0.1297 |
| Chrome           |     207 |  159763 |           0.1296 |
| Edge             |     201 |  158806 |           0.1266 |
| Opera            |     198 |  158726 |           0.1247 |
| Firefox          |     191 |  159566 |           0.1197 |

### device_os

| device_os   |   fraud |   count |   fraud_rate_pct |
|:------------|--------:|--------:|-----------------:|
| Windows     |     263 |  189988 |           0.1384 |
| Linux       |     258 |  190989 |           0.1351 |
| macOS       |     244 |  191773 |           0.1272 |
| iOS         |     236 |  190765 |           0.1237 |
| Android     |     231 |  190878 |           0.121  |

### billing_country

| billing_country   |   fraud |   count |   fraud_rate_pct |
|:------------------|--------:|--------:|-----------------:|
| RU                |      62 |   16699 |           0.3713 |
| ID                |      53 |   16687 |           0.3176 |
| NG                |      47 |   16492 |           0.285  |
| CN                |      45 |   16437 |           0.2738 |
| IN                |     159 |  111335 |           0.1428 |
| US                |     136 |  111353 |           0.1221 |
| DE                |     129 |  110079 |           0.1172 |
| BR                |     126 |  110341 |           0.1142 |
| FR                |     126 |  111347 |           0.1132 |
| GB                |     122 |  111523 |           0.1094 |
| PH                |     116 |  110697 |           0.1048 |
| VN                |     111 |  111403 |           0.0996 |


## Top Numeric Correlations with Target

|                             |   pearson_r |
|:----------------------------|------------:|
| amount                      |      0.0882 |
| errorBalanceDest            |      0.0646 |
| isFlaggedFraud              |      0.0637 |
| abs_errorBalanceDest        |      0.063  |
| orig_drained                |      0.0612 |
| is_night                    |      0.06   |
| log_amount                  |      0.0427 |
| hour_of_day                 |     -0.0334 |
| num_failed_payment_attempts |      0.0333 |
| day_index                   |      0.0315 |
| step                        |      0.0304 |
| time_since_last_hours       |      0.0272 |
| dest_is_customer            |      0.0257 |
| ip_billing_distance_km      |      0.0244 |
| shipping_billing_mismatch   |      0.0201 |
