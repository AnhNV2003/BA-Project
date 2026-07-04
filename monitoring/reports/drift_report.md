# Monitoring — Drift Report (draft)

Reference: day ≤ 10 (539,797 rows) · Current: day > 10 (414,596 rows)

Retraining trigger: any monitored PSI ≥ **0.25**.

## PSI table (natural split vs simulated-campaign split)

| feature                     |   psi_natural |   psi_drifted | trigger_natural   | trigger_drifted   |
|:----------------------------|--------------:|--------------:|:------------------|:------------------|
| amount                      |         0.008 |         0.127 | stable            | moderate          |
| account_age_days            |         0     |         0     | stable            | stable            |
| ip_billing_distance_km      |         0     |         0.28  | stable            | SIGNIFICANT       |
| num_failed_payment_attempts |         0     |         0.45  | stable            | SIGNIFICANT       |
| is_new_device               |         0     |         0     | stable            | stable            |
| hour_of_day                 |         0.009 |         0.009 | stable            | stable            |
| PREDICTION_SCORE            |         0.01  |         0.063 | stable            | stable            |

## Retraining decision

- Natural split: **no retraining needed** (all < 0.25).
- Simulated fraud campaign: **RETRAIN** — drift on: ip_billing_distance_km, num_failed_payment_attempts.
