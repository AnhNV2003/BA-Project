# EDA Summary — E-Commerce Fraud Detection

- Rows: **954,393** | Fraud: **1,232** (**0.1291%**) — severe imbalance.

## Fraud by transaction type

Fraud occurs **only in TRANSFER & CASH_OUT** (a structural PaySim fact).

| type     |   mean |   sum |   count |
|:---------|-------:|------:|--------:|
| TRANSFER |  0.769 |   614 |   79832 |
| CASH_OUT |  0.184 |   618 |  335057 |
| CASH_IN  |  0     |     0 |  210493 |
| DEBIT    |  0     |     0 |    6215 |
| PAYMENT  |  0     |     0 |  322796 |

## Synthetic risk flags (fraud rate when flag on vs off)

| signal                    |   rate_if_0(%) |   rate_if_1(%) |
|:--------------------------|---------------:|---------------:|
| is_new_device             |          0.104 |          0.259 |
| shipping_billing_mismatch |          0.105 |          0.366 |
| is_disposable_email       |          0.106 |          0.67  |
| high_risk_country         |          0.099 |          0.581 |
| is_night                  |          0.1   |          1.745 |

## Top correlations with isFraud

|                             |   pearson_r |
|:----------------------------|------------:|
| amount                      |       0.088 |
| isFlaggedFraud              |       0.064 |
| is_night                    |       0.06  |
| hour_of_day                 |      -0.033 |
| high_risk_country           |       0.032 |
| num_failed_payment_attempts |       0.032 |
| day_index                   |       0.031 |
| is_disposable_email         |       0.031 |
| step                        |       0.03  |
| time_since_last_hours       |       0.027 |
