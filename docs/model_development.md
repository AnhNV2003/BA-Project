# Model Development — PaySim Fraud Detection

## Methodology

- `prepare_feature_frame(full_df)` runs once before split, preserving causal `nameDest` history across train/validation/test.

- `FeatureTransformer` fits frequency maps, imputer, scaler, and feature schema on train only.

- Threshold and best model are selected on validation expected cost. Test is evaluated once for the selected configuration.

- Cost uses missed fraud amount, false-positive friction cost, and review cost for all flagged transactions.

- **Leaky groups ['all', 'base'] are excluded from deployable bundle selection** (post-transaction balances are unavailable at authorization time and near-deterministically encode the label). They remain below as a leaky upper-bound reference.


## Split

|        rows |   train_rows |   val_rows |   test_rows |   train_fraud_rate |   val_fraud_rate |   test_fraud_rate |   val_dest_seen_before_rate |   val_max_dest_txn_count_so_far |   train_step_min |   train_step_max |   val_step_min |   val_step_max |   test_step_min |   test_step_max |
|------------:|-------------:|-----------:|------------:|-------------------:|-----------------:|------------------:|----------------------------:|--------------------------------:|-----------------:|-----------------:|---------------:|---------------:|----------------:|----------------:|
| 6.36262e+06 |  4.45383e+06 |     954393 |      954393 |        0.000817947 |      0.000588856 |        0.00419953 |                     0.57568 |                             109 |                1 |              323 |            323 |            378 |             378 |             743 |


## Best Deployable Configuration (non-leaky groups only)

| model_key   | model_name   | feature_group   | is_leaky   | matrix   |   n_features |   val_auc_pr |   val_roc_auc |   val_threshold |   val_expected_cost |   val_precision |   val_recall |    val_f1 |   val_loss_avoided_pct |   val_flagged_rate |   test_auc_pr |   test_roc_auc |   test_expected_cost |   test_precision |   test_recall |   test_f1 |   test_loss_avoided_pct |   test_flagged_rate |
|:------------|:-------------|:----------------|:-----------|:---------|-------------:|-------------:|--------------:|----------------:|--------------------:|----------------:|-------------:|----------:|-----------------------:|-------------------:|--------------:|---------------:|---------------------:|-----------------:|--------------:|----------:|------------------------:|--------------------:|
| xgb         | XGBoost      | realistic       | False      | tree     |           44 |     0.816123 |       0.99916 |         0.38024 |              541275 |       0.0450472 |     0.983986 | 0.0861505 |                99.9719 |          0.0128626 |      0.908377 |       0.998762 |          1.18296e+07 |         0.251002 |      0.968313 |  0.398665 |                 99.8182 |           0.0162009 |


## Results (all groups; `is_leaky=True` = upper-bound reference, not deployable)

| model_name          | feature_group   | is_leaky   |   n_features |   val_auc_pr |   val_roc_auc |   val_threshold |   val_expected_cost |   val_precision |   val_recall |    val_f1 |   val_loss_avoided_pct |   test_auc_pr |   test_roc_auc |   test_expected_cost |   test_precision |   test_recall |   test_f1 |   test_loss_avoided_pct |
|:--------------------|:----------------|:-----------|-------------:|-------------:|--------------:|----------------:|--------------------:|----------------:|-------------:|----------:|-----------------------:|--------------:|---------------:|---------------------:|-----------------:|--------------:|----------:|------------------------:|
| XGBoost             | realistic       | False      |           44 |     0.816123 |      0.99916  |         0.38024 |    541275           |      0.0450472  |     0.983986 | 0.0861505 |                99.9719 |      0.908377 |       0.998762 |          1.18296e+07 |        0.251002  |      0.968313 | 0.398665  |                 99.8182 |
| Logistic Regression | realistic       | False      |           44 |     0.265479 |      0.989085 |         0.28044 |         3.25533e+06 |      0.00570584 |     0.982206 | 0.0113458 |                99.9255 |      0.586418 |       0.98997  |          5.36757e+06 |        0.0248795 |      0.996756 | 0.0485472 |                 99.9846 |
| Random Forest       | realistic       | False      |           44 |     0.747627 |      0.968732 |         0.001   |         5.02999e+06 |      0.0185451  |     0.941281 | 0.0363736 |                99.4358 |      0.811911 |       0.970234 |          2.8398e+07  |        0.0864758 |      0.946357 | 0.158471  |                 99.5688 |


## Notes

- Leaky groups (`base`/`all`) may reach AUC-PR≈1.0 because PaySim post-transaction balances encode strong reconciliation signals; treat these as an upper bound, not a deployable result.

- `realistic` and `dest` groups reflect deployable pre-authorization signal and drive the saved bundle.
