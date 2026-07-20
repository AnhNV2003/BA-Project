# Feature Engineering — PaySim Fraud Detection

## Source / Split

- Source: `data/processed/transactions_clean.parquet` with **6,362,620 rows**.

- Train/validation/test rows: **4,453,834 / 954,393 / 954,393**.

- Step windows: train 1..323; validation 323..378; test 378..743.

- Transformer state is fit on train only; train-fitted frequency maps, imputer, and scaler are reused for validation/test.

- Destination-history features are precomputed once on the full chronological frame before splitting, so validation/test rows can see train history but never future rows.


## Engineered Features

- Base PaySim: amount, log amount, cents, transfer/cash-out flags, balance reconciliation errors, drained/empty balance flags.

- Destination history: past-only counts, amount sums/means/std, unique senders, type counts, recency, amount ratio/z-score, past frequency.

- Synthetic/context: account age, device/risk flags, IP distance, time features, illustrative synthetic-customer velocity.

- Categorical encoding: fixed one-hot for `type`; train-frequency encoding for `browser`, `device_os`, and `billing_country`.

- Dropped from model matrices: raw IDs/display fields (`nameOrig`, `nameDest`, `customer_id`, `customer_name`, `email`, `billing_city`, `device_id`).


## Feature Groups

| group     |   n_features | purpose                                        |
|:----------|-------------:|:-----------------------------------------------|
| base      |           18 | full PaySim amount/balance signature           |
| dest      |           16 | past-only destination-account behaviour        |
| synth     |           21 | M1 synthetic/contextual risk                   |
| realistic |           44 | authorization-time-style features plus context |
| all       |           50 | full feature set                               |


## Transformer Output

| matrix   |   n_features | scaling                        |
|:---------|-------------:|:-------------------------------|
| tree     |           50 | none                           |
| linear   |           50 | median impute + StandardScaler |

- Constant columns dropped on train: **0**.

- Destination-history features: **16**.

- Encoded categorical features: **8**.


## Imbalance Handling Guidance for M5

- Full fraud prevalence: **0.1291%**.

- Train fraud prevalence: **0.0818%**.

- Recommended model weight default: `scale_pos_weight=1221.57` or class weights.

- SMOTE/undersampling, if used, must be applied only inside the train fold after transformation.


## Leakage Checks

| check                                                              | status   |
|:-------------------------------------------------------------------|:---------|
| first transaction for each destination has dest_txn_count_so_far=0 | PASS     |
| time_since_dest_last_seen=-1 for unseen destinations               | PASS     |
| raw display/id columns absent from feature matrix                  | PASS     |
| transformer rejects raw subsets without precomputed dest-history   | PASS     |


## Destination History Distribution Check

| split      |   dest_seen_before_rate |   max_dest_txn_count_so_far |
|:-----------|------------------------:|----------------------------:|
| validation |                  0.5757 |                         109 |
| test       |                  0.5644 |                         112 |


## Notes

- `nameDest`/`nameOrig` raw strings are not used as features; only past-derived aggregates or train-frequency encodings are used.

- Full balance features are intentionally separated from `realistic` features because PaySim post-transaction balances can make the task artificially easy.
