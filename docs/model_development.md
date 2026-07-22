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
| xgb         | XGBoost      | realistic       | False      | tree     |           44 |      0.80633 |      0.999046 |         0.24052 |              683120 |       0.0286406 |     0.989324 | 0.0556696 |                99.9796 |          0.0203407 |      0.906883 |       0.998748 |          6.32824e+06 |         0.172949 |      0.981038 |  0.294058 |                 99.9084 |           0.0238214 |


## Results (all groups; `is_leaky=True` = upper-bound reference, not deployable)

| model_name          | feature_group   | is_leaky   |   n_features |   val_auc_pr |   val_roc_auc |   val_threshold |   val_expected_cost |   val_precision |   val_recall |     val_f1 |   val_loss_avoided_pct |   test_auc_pr |   test_roc_auc |   test_expected_cost |   test_precision |   test_recall |    test_f1 |   test_loss_avoided_pct |
|:--------------------|:----------------|:-----------|-------------:|-------------:|--------------:|----------------:|--------------------:|----------------:|-------------:|-----------:|-----------------------:|--------------:|---------------:|---------------------:|-----------------:|--------------:|-----------:|------------------------:|
| XGBoost             | realistic       | False      |           44 |  0.80633     |      0.999046 |         0.24052 |    683120           |     0.0286406   |     0.989324 | 0.0556696  |                99.9796 |    0.906883   |       0.998748 |          6.32824e+06 |       0.172949   |      0.981038 | 0.294058   |                 99.9084 |
| Logistic Regression | realistic       | False      |           44 |  0.249108    |      0.989035 |         0.28044 |         3.26398e+06 |     0.00568767  |     0.982206 | 0.0113099  |                99.9255 |    0.586731   |       0.989973 |          5.3718e+06  |       0.0248561  |      0.996756 | 0.0485027  |                 99.9846 |
| Random Forest       | realistic       | False      |           44 |  0.744497    |      0.972119 |         0.001   |         4.39165e+06 |     0.0176794   |     0.948399 | 0.0347118  |                99.5266 |    0.812072   |       0.97378  |          2.62832e+07 |       0.0842521  |      0.953343 | 0.154822   |                 99.6029 |
| XGBoost             | synth           | False      |           21 |  0.267625    |      0.947041 |         0.08084 |         1.05192e+07 |     0.00149731  |     0.998221 | 0.00299014 |                99.9944 |    0.243114   |       0.922426 |          2.49068e+08 |       0.011308   |      0.960828 | 0.0223529  |                 96.2105 |
| Logistic Regression | synth           | False      |           21 |  0.0645356   |      0.940669 |         0.13074 |         1.11724e+07 |     0.00140671  |     1        | 0.00280946 |               100      |    0.217216   |       0.952129 |          1.09649e+07 |       0.0101422  |      1        | 0.0200807  |                100      |
| Logistic Regression | dest            | False      |           16 |  0.0012107   |      0.657759 |         0.17066 |         2.64526e+07 |     0.000596094 |     0.998221 | 0.00119148 |                99.9847 |    0.010941   |       0.66947  |          3.00548e+07 |       0.00422448 |      0.998253 | 0.00841335 |                 99.9425 |
| XGBoost             | dest            | False      |           16 |  0.0013044   |      0.673596 |         0.001   |         2.66603e+07 |     0.00058993  |     1        | 0.00117916 |               100      |    0.00894157 |       0.63095  |          4.11464e+07 |       0.00420666 |      0.997505 | 0.00837799 |                 99.7686 |
| Random Forest       | dest            | False      |           16 |  0.000806977 |      0.625786 |         0.001   |         1.13523e+08 |     0.000852319 |     0.784698 | 0.00170279 |                86.8332 |    0.00574441 |       0.624775 |          9.2977e+08  |       0.005887   |      0.802645 | 0.0116883  |                 85.5371 |
| Random Forest       | synth           | False      |           21 |  0.288912    |      0.820089 |         0.001   |         2.64262e+08 |     0.00521656  |     0.681495 | 0.0103539  |                65.1422 |    0.137955   |       0.847376 |          1.5488e+09  |       0.0301775  |      0.750749 | 0.0580227  |                 75.55   |
| Random Forest       | all             | True       |           50 |  1           |      1        |         0.22056 |      1686           |     1           |     1        | 1          |               100      |    0.999993   |       1        |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| Random Forest       | base            | True       |           18 |  1           |      1        |         0.47006 |      1686           |     1           |     1        | 1          |               100      |    0.99996    |       1        |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| XGBoost             | all             | True       |           50 |  1           |      1        |         0.51996 |      1686           |     1           |     1        | 1          |               100      |    0.999768   |       0.999995 |     764965           |       0.99975    |      0.999501 | 0.999626   |                 99.9881 |
| XGBoost             | base            | True       |           18 |  1           |      1        |         0.96906 |      1686           |     1           |     1        | 1          |               100      |    0.999778   |       0.999992 |     764937           |       1          |      0.999501 | 0.99975    |                 99.9881 |
| Logistic Regression | base            | True       |           18 |  0.522598    |      0.997617 |         0.57984 |    735090           |     0.0210054   |     1        | 0.0411465  |               100      |    0.773785   |       0.995296 |          7.36499e+07 |       0.11925    |      0.992265 | 0.212913   |                 98.8485 |
| Logistic Regression | all             | True       |           50 |  0.603778    |      0.998172 |         0.55988 |    975835           |     0.0269354   |     0.989324 | 0.0524429  |                99.9453 |    0.85334    |       0.997219 |          7.37384e+07 |       0.114315   |      0.989022 | 0.204943   |                 98.8477 |


## Notes

- Leaky groups (`base`/`all`) may reach AUC-PR≈1.0 because PaySim post-transaction balances encode strong reconciliation signals; treat these as an upper bound, not a deployable result.

- `realistic` and `dest` groups reflect deployable pre-authorization signal and drive the saved bundle.
