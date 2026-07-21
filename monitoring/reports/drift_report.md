# Monitoring — Drift Report (draft)

Reference: day ≤ 15 (51,670 rows) · Current: day > 15 (48,330 rows)

Retraining trigger: any monitored PSI ≥ **0.25**.

## PSI table (natural split vs simulated-campaign split)

| feature                     |   psi_natural |   psi_drifted | trigger_natural   | trigger_drifted   |
|:----------------------------|--------------:|--------------:|:------------------|:------------------|
| amount                      |         0     |         0.078 | stable            | stable            |
| account_age_days            |         0     |         0     | stable            | stable            |
| ip_billing_distance_km      |         0     |         0.286 | stable            | SIGNIFICANT       |
| num_failed_payment_attempts |         0     |         0.445 | stable            | SIGNIFICANT       |
| is_new_device               |         0     |         0     | stable            | stable            |
| hour_of_day                 |         0.001 |         0.001 | stable            | stable            |
| PREDICTION_SCORE_logreg     |         0.004 |         0.099 | stable            | stable            |
| PREDICTION_SCORE_rf         |         0     |         0     | stable            | stable            |
| PREDICTION_SCORE_xgb        |         0.018 |         0.47  | stable            | SIGNIFICANT       |

## Retraining decision

- Natural split: **no retraining needed** (all < 0.25).
- Simulated fraud campaign: **RETRAIN** — drift on: ip_billing_distance_km, num_failed_payment_attempts, PREDICTION_SCORE_xgb.
