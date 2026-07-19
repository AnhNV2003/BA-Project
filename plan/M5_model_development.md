# Plan: Task 5 (Module 5) — Model Development trên PaySim

## Context

Dùng features PaySim từ `src/features.py` để train & so sánh classifiers.
Script: `src/train_validate.py`.

## Feature pipeline (BẮT BUỘC — khớp fix M4)

- **`prepare_feature_frame(full_df)` MỘT LẦN trước khi split** (dest-history causal past-only tính trên full chronological — nếu tính per-subset sẽ tái diễn bug train/serve skew: val/test mất lịch sử train của nameDest).
- Dùng `FeatureTransformer.fit(train)/transform(val,test)` (freq-encode/impute/scale fit train-only).
- Feature-group experiments (base/dest/synth/realistic/all): **subset cột trên output transformer**, KHÔNG dùng legacy `feature_matrix` (nó recompute dest-history per-subset + để freq cols = 0).
- `tree` group (XGB/RF... ) giữ NaN; `linear` group (LogReg) đã impute+scale.

## Split rigor (tránh test-peeking)

- Chia **train / val / test** (vd 60/20/20; stratified theo `isFraud`, hoặc time-based theo `step`) — không chỉ train/test.
- **Chọn operating threshold trên VAL** (min expected cost), **đánh giá TEST đúng 1 lần**.
- **Chọn model tốt nhất theo VAL** (không dùng test để chọn) — bản `train_validate.py` hiện tại đang chọn threshold + model trên test, phải sửa.
- **Loại nhóm leaky khỏi bundle deploy**: `LEAKY_GROUPS = {base, all}` chứa balance sau giao dịch (`newbalance*`/`errorBalance*`/`orig_drained`) — mã hóa nhãn gần tất định, không có ở authorization-time. Chúng vẫn train + hiện trong bảng so sánh làm **upper-bound reference** (`is_leaky=True`), nhưng **bundle deploy chỉ chọn best trong nhóm non-leaky** (`realistic`/`dest`/`synth`). Nếu chỉ train toàn nhóm leaky → raise lỗi rõ ràng.
- Cân nhắc so sánh thêm split **time-based theo `step`** (train quá khứ → test tương lai) để kiểm tra drift; bản gốc dùng random stratified.

## Models

- Logistic Regression, Random Forest, XGBoost (LightGBM thêm nếu cần).

## Metrics (hợp imbalance)

- AUC-PR, ROC-AUC, Precision / Recall / F1.
- **Cost-based**: FN = missed fraud `amount`; FP = friction/manual handling; review cost/giao dịch flagged. `loss_avoided` = tiền fraud chặn được / tổng tiền fraud.

## Imbalance handling

- Mặc định class weights / `scale_pos_weight`.
- Tùy chọn SMOTE / undersampling **chỉ trên train fold** (val/test không resample) — so sánh với class-weights.

## Feature Experiments

- `base`: full PaySim balance features.
- `synth`: synthetic context only.
- `realistic`: base gần authorization-time + synthetic (không có post-transaction balance).
- `all`: base + synthetic.

## Outputs

- `models/fraud_model.joblib` (bundle: model + transformer + feature_group + threshold + metrics + `leaky_excluded_from_selection` — cho M6; **model non-leaky**).
- `docs/model_development.md` + `docs/model_results.csv` (bảng so sánh val + test, có cột `is_leaky`).
- `docs/cost_model_worksheet.md` (kịch bản tự nghiên cứu để suy 3 hệ số cost — cơ sở cho threshold selection).
- Console metrics từ `python src/train_validate.py`.

## Verification

- **`prepare_feature_frame` chạy trước split**; transform val/test giữ dest-history full (dest_seen_before val ~35%, không bị reset ~6.5%).
- Threshold + chọn model **theo val**, test đánh giá 1 lần (kiểm code path).
- Synthetic-only có tín hiệu nhưng **không hoàn hảo** (AUC < 1).
- Full PaySim (`all`/`base`) có thể rất cao vì balance áp đảo → **trung thực nêu rõ**; nhóm `realistic` cho con số deployable hơn.
- Report threshold, expected cost, loss avoided; val→test gap hợp lý.
