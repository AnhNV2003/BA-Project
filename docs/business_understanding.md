# Business Understanding — PaySim Online Payments Fraud Detection

## Business Problem

The payment platform needs to score online money-transfer transactions and
decide whether to approve, block, or route them for manual review.

The core business trade-off is:

- missed fraud creates direct financial loss,
- false alarms create customer friction and review cost,
- manual review capacity is limited.

PaySim is a deliberately imbalanced fraud dataset: the full file has 6,362,620
transactions and 8,213 frauds, a fraud rate of about 0.129%.

## KPI Definitions

| KPI | Formula | Business meaning |
|---|---|---|
| Fraud rate | `N_fraud / N_transactions` | Base prevalence the model must handle. PaySim full: 8,213 / 6,362,620 = 0.129%. |
| False-positive rate | `FP / (FP + TN)` | Share of legitimate transactions incorrectly flagged. Lower means less customer friction. |
| Financial loss avoided % | `sum(amount for TP) / sum(amount for all fraud)` | Share of fraud money captured before loss. |

## Cost Model

The modelling stages use configurable assumptions from `src/config.py`:

- False negative cost: `COST_FALSE_NEGATIVE * amount`
- False positive cost: `COST_FALSE_POSITIVE` per wrongly flagged legitimate transaction
- Manual review cost: `COST_MANUAL_REVIEW` per flagged transaction

Expected cost:

```text
sum(amount for false negatives) * COST_FALSE_NEGATIVE
+ N_false_positive * COST_FALSE_POSITIVE
+ N_flagged * COST_MANUAL_REVIEW
```

Baseline do-nothing loss:

```text
sum(amount for all fraudulent transactions)
```

## Module 1 Data Decision

The raw PaySim table has transaction and balance information but no
e-commerce context such as customer age, device fingerprint, browser, failed
payment attempts, shipping/billing mismatch, or IP distance. Module 1 therefore
adds a synthetic contextual layer generated with Python/Faker.

Important limitations:

- Real `nameOrig` is almost always single-use, so origin-account velocity is not
  a reliable real signal.
- Repeated `nameDest` accounts are more promising for later mule-account feature
  engineering, but that belongs to Module 4 rather than synthetic generation.
- PaySim balance fields are very predictive; later modelling should compare a
  stricter "realistic" feature set that excludes post-transaction balance
  reconciliation.
