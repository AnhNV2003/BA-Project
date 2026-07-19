# Plan: Task 4 (Module 4) — Feature Engineering trên PaySim

## Context

Input là cleaned PaySim từ M3. M4 tạo model-ready features và so sánh nhóm tín
hiệu base PaySim với synthetic context + destination history.

## Feature Groups (trong `src/features.py`)

- `BASE_NUMERIC`: amount, log_amount, amount_cents, balances, balance-reconciliation errors, drained/empty flags, transfer/cash-out flags.
- `BASE_REALISTIC`: chỉ trường gần authorization-time (bỏ post-transaction balance) — để synthetic/dest context có đất dụng võ.
- `SYNTH_NUMERIC`: context M1 (account age, device/risk flags, IP distance, time, velocity minh họa).
- `DEST_HISTORY_NUMERIC`: feature lịch sử `nameDest` (mục 1).
- `all`: base + dest + synth + encoded categorical.

## 1. Risk-signal features từ `nameDest` (điểm mạnh — past-only)

Nguyên tắc: `nameDest` lặp nhiều hơn hẳn `nameOrig` (~single-use) → là entity mule. Khai thác bằng **aggregation past-only**, **KHÔNG dùng ID thô làm feature**.

**1a. Historical aggregation** (chỉ từ giao dịch quá khứ):
- `dest_seen_before`, `dest_txn_count_so_far`, `dest_amount_sum_so_far`, `dest_amount_mean_so_far`, `dest_amount_std_so_far`, `dest_unique_senders_so_far`, count theo từng type so far.

**1b. Recency của entity** (analog "days/time since"):
- `time_since_dest_last_seen` (giờ kể từ giao dịch trước của cùng `nameDest`, dùng `step`; -1 nếu lần đầu).

**1c. Group-normalized amount anomaly**:
- `amount_to_dest_mean_ratio` = amount / mean past; `amount_dest_zscore` so với mean/std past.

**1d. Frequency encoding thay ID thô**:
- `dest_freq_so_far`, `orig_freq_so_far` (past-only); tuyệt đối không đưa chuỗi ID thô vào ma trận.

**1e. Amount-decimal**:
- `amount_cents` = phần thập phân của amount (round-number vs lẻ — card-testing).

> **BẮT BUỘC — tính trên FULL chronological, KHÔNG per-subset:** các feature dest-history phải được tính một lần trên toàn bộ dữ liệu đã sort theo `step` (causal: mỗi dòng chỉ dùng quá khứ → an toàn, không leak), rồi mới split. Nếu tính riêng từng subset trong `transform()`, val/test sẽ mất lịch sử train của mỗi dest (bị reset về 0 → `dest_seen_before` ~6.5% thay vì ~35%) gây train/serve skew, làm hỏng chính feature nameDest. Chỉ freq-encoding categorical + imputer + scaler mới fit-on-train.

## 2. Encode categorical + scale numeric

- **Categorical**: `type` one-hot schema cố định; `browser`/`device_os`/`billing_country` bằng **train-frequency encoding** (`*_freq_train`).
- **Scaling**: `StandardScaler` cho nhóm linear (LogReg) + median impute; tree (RF/XGB) giữ NaN, không scale.
- Loại khỏi ma trận: display/id (`customer_id`, `customer_name`, `email`, `billing_city`, `device_id`, `nameOrig`, `nameDest` thô).

## 3. Xử lý mất cân bằng + feature selection

- **Imbalance (0.129%)**: mặc định class weights / `scale_pos_weight` (áp ở M5). SMOTE/undersampling nếu dùng **chỉ trên train fold** sau transform.
- **Feature selection**: bỏ near-constant trên train; các nhóm base/realistic/synth/dest/all làm thí nghiệm feature-set ở M5.

## Transformer (fit-on-train)

`FeatureTransformer.fit(train_df)` học freq maps + imputer + scaler + danh sách feature (đã bỏ constant). `transform(df, group)`:
- `tree`: reindex cột, giữ NaN.
- `linear`: reindex + median impute + StandardScaler.

## Outputs
- `models/feature_transformer.joblib`
- `data/processed/features_preview.parquet`
- `docs/feature_engineering.md`
- `docs/feature_groups.md` (liệt kê chính xác cột × nhóm base/dest/synth/realistic/all + cờ leaky/authorization-time; sinh tự động bằng `python src/make_feature_groups_doc.py` từ `features.py`+`train_validate.py` để chống drift).

## Verification
- Feature matrices chỉ numeric; display/id vắng mặt.
- **Dest-history tính trên full chronological** → `dest_seen_before`/`dest_txn_count_so_far` của val/test khớp bản full (không bị reset per-subset).
- Past-only sanity: giao dịch đầu của mỗi dest có `dest_txn_count_so_far=0`, `time_since_dest_last_seen=-1`.
- Freq maps/imputer/scaler fit train-only; linear matrix không còn NaN.
- Schema feature ổn định giữa train/val/test.
