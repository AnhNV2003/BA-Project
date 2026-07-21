# Model Development — PaySim Fraud Detection

## Methodology

- `prepare_feature_frame(full_df)` runs once before split, preserving causal `nameDest` history across train/validation/test.

- `FeatureTransformer` fits frequency maps, imputer, scaler, and feature schema on train only.

- Threshold and best model are selected on validation expected cost. Test is evaluated once for the selected configuration.

- Cost uses missed fraud amount, false-positive friction cost, and review cost for all flagged transactions.

- **Leaky groups ['all', 'base'] are excluded from deployable bundle selection** (post-transaction balances are unavailable at authorization time and near-deterministically encode the label). They remain below as a leaky upper-bound reference.


## Split

|   rows |   train_rows |   val_rows |   test_rows |   train_fraud_rate |   val_fraud_rate |   test_fraud_rate |   val_dest_seen_before_rate |   val_max_dest_txn_count_so_far |   train_step_min |   train_step_max |   val_step_min |   val_step_max |   test_step_min |   test_step_max |
|-------:|-------------:|-----------:|------------:|-------------------:|-----------------:|------------------:|----------------------------:|--------------------------------:|-----------------:|-----------------:|---------------:|---------------:|----------------:|----------------:|
| 100000 |        70000 |      15000 |       15000 |             0.0014 |       0.00173333 |        0.00173333 |                      0.9932 |                              16 |                1 |              520 |            520 |            631 |             631 |             743 |


## Best Deployable Configuration (non-leaky groups only)

| model_key   | model_name    | feature_group   | is_leaky   | matrix   |   n_features |   val_auc_pr |   val_roc_auc |   val_threshold |   val_expected_cost |   val_precision |   val_recall |   val_f1 |   val_loss_avoided_pct |   val_flagged_rate |   test_auc_pr |   test_roc_auc |   test_expected_cost |   test_precision |   test_recall |   test_f1 |   test_loss_avoided_pct |   test_flagged_rate |
|:------------|:--------------|:----------------|:-----------|:---------|-------------:|-------------:|--------------:|----------------:|--------------------:|----------------:|-------------:|---------:|-----------------------:|-------------------:|--------------:|---------------:|---------------------:|-----------------:|--------------:|----------:|------------------------:|--------------------:|
| rf          | Random Forest | realistic       | False      | tree     |           44 |            1 |             1 |          0.1008 |                  78 |               1 |            1 |        1 |                    100 |         0.00173333 |             1 |              1 |                   78 |                1 |             1 |         1 |                     100 |          0.00173333 |


## Results (all groups; `is_leaky=True` = upper-bound reference, not deployable)

| model_name          | feature_group   | is_leaky   |   n_features |   val_auc_pr |   val_roc_auc |   val_threshold |   val_expected_cost |   val_precision |   val_recall |     val_f1 |   val_loss_avoided_pct |   test_auc_pr |   test_roc_auc |   test_expected_cost |   test_precision |   test_recall |    test_f1 |   test_loss_avoided_pct |
|:--------------------|:----------------|:-----------|-------------:|-------------:|--------------:|----------------:|--------------------:|----------------:|-------------:|-----------:|-----------------------:|--------------:|---------------:|---------------------:|-----------------:|--------------:|-----------:|------------------------:|
| Random Forest       | realistic       | False      |           44 |   1          |      1        |         0.1008  |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                 78   |       1          |      1        | 1          |                100      |
| XGBoost             | realistic       | False      |           44 |   0.997253   |      0.999995 |         0.01098 |                 134 |      0.928571   |     1        | 0.962963   |               100      |     1         |       1        |                162   |       0.896552   |      1        | 0.945455   |                100      |
| Logistic Regression | realistic       | False      |           44 |   0.691      |      0.998243 |         0.51996 |                7918 |      0.0849673  |     1        | 0.156627   |               100      |     0.493369  |       0.995613 |              62985.1 |       0.0842105  |      0.923077 | 0.154341   |                 95.5129 |
| XGBoost             | dest            | False      |           16 |   0.0279281  |      0.925821 |         0.001   |              126602 |      0.00837802 |     0.961538 | 0.0166113  |                97.3735 |     0.017591  |       0.92011  |             197549   |       0.00918897 |      0.884615 | 0.018189   |                 89.6678 |
| Logistic Regression | dest            | False      |           16 |   0.0193027  |      0.913125 |         0.5     |              137033 |      0.00877514 |     0.923077 | 0.017385   |                96.3285 |     0.0113032 |       0.833889 |             265246   |       0.00559211 |      0.653846 | 0.0110894  |                 85.4304 |
| Logistic Regression | synth           | False      |           21 |   0.0184358  |      0.882227 |         0.13074 |              170010 |      0.00426579 |     1        | 0.00849534 |               100      |     0.0838021 |       0.880768 |             290758   |       0.0039676  |      0.923077 | 0.00790123 |                 90.1563 |
| Random Forest       | dest            | False      |           16 |   0.0271053  |      0.83435  |         0.001   |              276921 |      0.0148322  |     0.730769 | 0.0290742  |                85.4752 |     0.0115872 |       0.769927 |             349025   |       0.0130612  |      0.615385 | 0.0255795  |                 74.5709 |
| XGBoost             | synth           | False      |           21 |   0.0050647  |      0.715209 |         0.001   |              621663 |      0.00330907 |     0.769231 | 0.00658979 |                72.7621 |     0.0243523 |       0.829476 |             224456   |       0.00410914 |      0.961538 | 0.00818331 |                 95.5836 |
| Random Forest       | synth           | False      |           21 |   0.00802977 |      0.747122 |         0.001   |              823043 |      0.00653846 |     0.653846 | 0.0129474  |                54.8571 |     0.0288476 |       0.799043 |             389706   |       0.00683453 |      0.730769 | 0.0135424  |                 74.7956 |
| Logistic Regression | all             | True       |           50 |   1          |      1        |         0.02096 |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                 78   |       1          |      1        | 1          |                100      |
| Logistic Regression | base            | True       |           18 |   1          |      1        |         0.02096 |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                 78   |       1          |      1        | 1          |                100      |
| Random Forest       | all             | True       |           50 |   1          |      1        |         0.0509  |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                106   |       0.962963   |      1        | 0.981132   |                100      |
| Random Forest       | base            | True       |           18 |   1          |      1        |         0.14072 |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                106   |       0.962963   |      1        | 0.981132   |                100      |
| XGBoost             | all             | True       |           50 |   1          |      1        |         0.01098 |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                 78   |       1          |      1        | 1          |                100      |
| XGBoost             | base            | True       |           18 |   1          |      1        |         0.001   |                  78 |      1          |     1        | 1          |               100      |     1         |       1        |                 78   |       1          |      1        | 1          |                100      |


## Notes

- Leaky groups (`base`/`all`) may reach AUC-PR≈1.0 because PaySim post-transaction balances encode strong reconciliation signals; treat these as an upper bound, not a deployable result.

- `realistic` and `dest` groups reflect deployable pre-authorization signal and drive the saved bundle.
