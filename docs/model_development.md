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

| model_name          | feature_group   | is_leaky   |   n_features |   val_auc_pr |   val_roc_auc |   val_threshold |   val_expected_cost |   val_precision |   val_recall |     val_f1 |   val_loss_avoided_pct |   test_auc_pr |   test_roc_auc |   test_expected_cost |   test_precision |   test_recall |    test_f1 |   test_loss_avoided_pct |
|:--------------------|:----------------|:-----------|-------------:|-------------:|--------------:|----------------:|--------------------:|----------------:|-------------:|-----------:|-----------------------:|--------------:|---------------:|---------------------:|-----------------:|--------------:|-----------:|------------------------:|
| XGBoost             | realistic       | False      |           44 |  0.816123    |      0.99916  |         0.38024 |    541275           |     0.0450472   |     0.983986 | 0.0861505  |                99.9719 |    0.908377   |       0.998762 |          1.18296e+07 |       0.251002   |      0.968313 | 0.398665   |                 99.8182 |
| Logistic Regression | realistic       | False      |           44 |  0.265479    |      0.989085 |         0.28044 |         3.25533e+06 |     0.00570584  |     0.982206 | 0.0113458  |                99.9255 |    0.586418   |       0.98997  |          5.36757e+06 |       0.0248795  |      0.996756 | 0.0485472  |                 99.9846 |
| Random Forest       | realistic       | False      |           44 |  0.747627    |      0.968732 |         0.001   |         5.02999e+06 |     0.0185451   |     0.941281 | 0.0363736  |                99.4358 |    0.811911   |       0.970234 |          2.8398e+07  |       0.0864758  |      0.946357 | 0.158471   |                 99.5688 |
| XGBoost             | synth           | False      |           21 |  0.271434    |      0.948667 |         0.07086 |         1.08305e+07 |     0.00145417  |     0.998221 | 0.00290411 |                99.9944 |    0.255048   |       0.925568 |          2.00346e+08 |       0.0109901  |      0.971307 | 0.0217343  |                 96.987  |
| Logistic Regression | synth           | False      |           21 |  0.0639299   |      0.940689 |         0.13074 |         1.11752e+07 |     0.00140635  |     1        | 0.00280876 |               100      |    0.216858   |       0.952125 |          1.09666e+07 |       0.0101406  |      1        | 0.0200776  |                100      |
| Logistic Regression | dest            | False      |           16 |  0.00123216  |      0.657761 |         0.17066 |         2.64526e+07 |     0.000596095 |     0.998221 | 0.00119148 |                99.9847 |    0.0109555  |       0.669471 |          3.00548e+07 |       0.00422448 |      0.998253 | 0.00841335 |                 99.9425 |
| XGBoost             | dest            | False      |           16 |  0.00146176  |      0.67775  |         0.001   |         2.66868e+07 |     0.000589344 |     1        | 0.00117799 |               100      |    0.00960592 |       0.64838  |          5.25268e+07 |       0.004208   |      0.996756 | 0.00838061 |                 99.5881 |
| Random Forest       | dest            | False      |           16 |  0.000809821 |      0.628044 |         0.001   |         1.01764e+08 |     0.000863182 |     0.786477 | 0.00172447 |                88.3765 |    0.0057161  |       0.620004 |          1.05924e+09 |       0.00566227 |      0.803643 | 0.0112453  |                 83.4996 |
| Random Forest       | synth           | False      |           21 |  0.295407    |      0.813596 |         0.001   |         2.35553e+08 |     0.00510586  |     0.66726  | 0.0101342  |                68.9587 |    0.131406   |       0.839639 |          1.65417e+09 |       0.029636   |      0.734281 | 0.0569725  |                 73.8833 |
| Random Forest       | all             | True       |           50 |  1           |      1        |         0.17066 |      1686           |     1           |     1        | 1          |               100      |    0.999963   |       1        |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| Random Forest       | base            | True       |           18 |  1           |      1        |         0.44012 |      1686           |     1           |     1        | 1          |               100      |    0.999998   |       1        |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| XGBoost             | all             | True       |           50 |  1           |      1        |         0.50998 |      1686           |     1           |     1        | 1          |               100      |    0.999827   |       0.999998 |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| XGBoost             | base            | True       |           18 |  1           |      1        |         0.96906 |      1686           |     1           |     1        | 1          |               100      |    0.999809   |       0.999997 |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| Logistic Regression | base            | True       |           18 |  0.533488    |      0.997609 |         0.57984 |    739934           |     0.0208705   |     1        | 0.0408876  |               100      |    0.774419   |       0.995321 |          7.35273e+07 |       0.118736   |      0.992515 | 0.212098   |                 98.8505 |
| Logistic Regression | all             | True       |           50 |  0.611976    |      0.998175 |         0.55988 |    974995           |     0.0269746   |     0.989324 | 0.0525172  |                99.9453 |    0.85349    |       0.997223 |          7.38078e+07 |       0.114405   |      0.988772 | 0.205082   |                 98.8466 |


## Notes

- Leaky groups (`base`/`all`) may reach AUC-PR≈1.0 because PaySim post-transaction balances encode strong reconciliation signals; treat these as an upper bound, not a deployable result.

- `realistic` and `dest` groups reflect deployable pre-authorization signal and drive the saved bundle.
