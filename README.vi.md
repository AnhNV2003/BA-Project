# Phát hiện Gian lận Thanh toán Thời gian thực cho Sàn E-Commerce

Bài tập lớn nhóm môn Business Analytics — một sản phẩm phân tích end-to-end:
**sinh dữ liệu → EDA → làm sạch → mô hình → API + hàng đợi review → giám sát.**

Câu hỏi kinh doanh: *giao dịch nào có khả năng gian lận, và làm sao cân bằng giữa
chặn gian lận và không gây phiền khách hàng hợp lệ?*

> 🇬🇧 Bản tiếng Anh: [README.md](README.md)

---

## Dữ liệu

- **Nguồn nền (bắt buộc):** PaySim — Kaggle `rupakroy/online-payments-fraud-detection-dataset`.
  Tải file CSV vào thư mục **`data/raw/`** (không commit). 6.362.620 dòng, tỉ lệ
  gian lận **0,129%**, gian lận **chỉ xảy ra ở TRANSFER & CASH_OUT**.
- **Dữ liệu synthetic (phần mở rộng bắt buộc):** danh tính (Faker) + tín hiệu rủi ro
  (thiết bị mới, lệch địa chỉ ship/bill, số lần thanh toán fail, khoảng cách IP,
  tuổi tài khoản, …), sinh bởi `src/synth_context.py`. Mọi cột được ghi rõ trong
  **`docs/data_dictionary.md`**.

## 📥 Lấy dữ liệu

Các file nặng (CSV gốc, parquet đã xử lý, model bundle) **không** commit vào git —
tự sinh lại ở máy local. Chỉ cần 1 lệnh:

```bash
pip install -r requirements.txt
./scripts/get_data.sh          # tải qua Kaggle CLI + build lại toàn bộ (M1->M5)
```

Chưa cấu hình Kaggle API token? Xem **[docs/data_setup.md](docs/data_setup.md)**
để tải tay, dùng bản mẫu nhanh hơn (`./scripts/get_data.sh --sample 0.15`), hoặc
tái sử dụng artifact đồng đội đã build sẵn thay vì build lại từ đầu.

- **Nguồn nền (bắt buộc):** PaySim — Kaggle `rupakroy/online-payments-fraud-detection-dataset`.
  6.362.620 dòng, tỉ lệ gian lận **0,129%**, gian lận **chỉ xảy ra ở TRANSFER & CASH_OUT**.
- **Dữ liệu synthetic (phần mở rộng bắt buộc):** danh tính (Faker) + tín hiệu rủi ro
  (thiết bị mới, lệch địa chỉ ship/bill, số lần thanh toán fail, khoảng cách IP,
  tuổi tài khoản, …), sinh bởi `src/synth_context.py`. Mọi cột được ghi rõ trong
  **`docs/data_dictionary.md`** / **`docs/feature_groups.md`**.

> ⚠️ Thư viện đã cài trong venv `SeminarProject/.venv` (Python 3.14). Nếu VSCode báo
> "package chưa cài" thì chỉ cần chọn đúng interpreter đó — không phải lỗi.

## Chạy pipeline (từ thư mục gốc của repo)

Sau khi có CSV trong `data/raw/` (qua `get_data.sh` hoặc tải tay):

| # | Lệnh | Kết quả |
|---|---|---|
| 1 | `python src/build_dataset.py --full` | `data/processed/transactions_context.parquet` + file preview |
| 2 | `python src/eda.py` | `docs/figures/*.png` + `docs/eda_summary.md` |
| 3 | `python src/cleaning.py` | `data/processed/transactions_clean.parquet` + `docs/cleaning_report.md` |
| 4 | `python src/features.py` | `models/feature_transformer.joblib` + `docs/feature_engineering.md` |
| 5 | `python src/train_validate.py` | `models/fraud_model.joblib` + `docs/model_development.md` |
| 6 | `uvicorn api.main:app --reload` | API chấm điểm → http://127.0.0.1:8000/docs |
| 7 | `streamlit run app/streamlit_app.py` | hàng đợi review cho chuyên viên rủi ro |
| 8 | `python monitoring/drift.py` | `monitoring/reports/drift_report.md` + `evidently_drift.html` |

Hoặc chạy chung bước 1-5: `./train.sh full` (toàn bộ 6.36M dòng) / `./train.sh sample 0.15`.

## Bản đồ Module → file

| Module | File |
|---|---|
| M1 Hiểu bài toán & sinh dữ liệu | `src/synth_context.py`, `src/build_dataset.py`, `docs/data_dictionary.md` |
| M2 EDA | `src/eda.py` |
| M3 Làm sạch dữ liệu | `src/cleaning.py` |
| M4 Feature engineering | `src/features.py` |
| M5 Phát triển mô hình | `src/train_validate.py` |
| M6 Triển khai | `api/main.py`, `app/streamlit_app.py` |
| M7 Giám sát | `monitoring/drift.py` |
| M8 Báo cáo & thuyết trình | _CHƯA LÀM_ |

## Kết quả chính (full data, 6.362.620 dòng, prevalence 0,129%)

- **Model deploy được:** XGBoost trên nhóm feature `realistic` (chỉ tín hiệu có ở
  authorization-time) — test **AUC-PR 0,91**, ROC-AUC 0,999, recall 96,8%, loss
  avoided 99,8%.
- **Nhóm leaky (không deploy):** nhóm `base`/`all` đạt AUC-PR≈1,0 vì các cột balance
  sau giao dịch của PaySim mã hóa nhãn gần tất định. 2 nhóm này bị loại khỏi bundle
  lưu — xem `docs/feature_groups.md` để biết chính xác cột nào leaky/deploy được,
  và `docs/model_development.md` / `docs/model_results.csv` để so sánh đầy đủ.
- Feature lịch sử `nameDest` được tính một lần trên toàn bộ dữ liệu theo thời gian
  trước khi chia train/val/test để tránh train/serve skew.
- Trọng số chi phí (`COST_FALSE_NEGATIVE/FALSE_POSITIVE/MANUAL_REVIEW` trong
  `src/config.py`) là giả định kinh doanh — xem `docs/cost_model_worksheet.md` để
  suy ra có căn cứ thay vì đặt bừa.

## Kiến trúc synthetic 3 lớp (để bảo vệ trong báo cáo)

1. **Lớp danh tính** (Faker, theo từng khách, cache 1 lần): tên, email, thành phố,
   toạ độ nhà. Vì `nameOrig` thật dùng-một-lần (99,9% duy nhất) nên ta tự tạo pool
   200k khách hàng tổng hợp.
2. **Lớp thuộc tính rủi ro tài khoản** (numpy, theo khách, điều kiện theo mức rủi ro):
   tuổi tài khoản, quốc gia, email dùng-một-lần — nhất quán cho mỗi khách.
3. **Lớp tín hiệu rủi ro giao dịch** (numpy, theo giao dịch, điều kiện theo `isFraud`):
   device mới, lệch ship/bill, số lần fail, khoảng cách IP (haversine), giờ + velocity.
   Cơ chế *reveal* (mỗi fraud chỉ lộ ~55% red-flag) đảm bảo không leakage.

## Trạng thái

- [x] **P1** — sinh dữ liệu, data dictionary, EDA, làm sạch
- [x] **P2 (nháp)** — features, so sánh mô hình, kiểm tra leakage, chọn threshold theo chi phí
- [x] **P3 (nháp)** — API chấm điểm, giao diện review-queue, giám sát drift
- [ ] Deploy lên cloud free tier (Hugging Face Spaces / Render) — **bắt buộc có link sống**
- [ ] Build full-data + thí nghiệm mất cân bằng (SMOTE vs class weights)
- [ ] Báo cáo + slide M8
- [ ] Điền các mốc ngày nộp bài
