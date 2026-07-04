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

## Cài đặt

```bash
pip install -r requirements.txt
# sau đó tải CSV PaySim vào data/raw/
```

> ⚠️ Thư viện đã cài trong venv `SeminarProject/.venv` (Python 3.14). Nếu VSCode báo
> "package chưa cài" thì chỉ cần chọn đúng interpreter đó — không phải lỗi.

## Chạy pipeline (từ thư mục gốc của repo)

| # | Lệnh | Kết quả |
|---|---|---|
| 1 | `python src/build_dataset.py` (`--full` cho toàn bộ 6.36M) | `data/processed/transactions_context.parquet` + file preview |
| 2 | `python src/eda.py` | `docs/figures/*.png` + `docs/eda_summary.md` |
| 3 | `python src/cleaning.py` | `data/processed/transactions_clean.parquet` + `docs/cleaning_report.md` |
| 4 | `python src/train_validate.py` | `models/fraud_model.joblib` + metrics / kiểm tra leakage / threshold |
| 5 | `uvicorn api.main:app --reload` | API chấm điểm → http://127.0.0.1:8000/docs |
| 6 | `streamlit run app/streamlit_app.py` | hàng đợi review cho chuyên viên rủi ro |
| 7 | `python monitoring/drift.py` | `monitoring/reports/drift_report.md` + `evidently_drift.html` |

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

## Kết quả chính (mẫu phân tầng 15%, giữ prevalence thật 0,129%)

- **Mô hình:** XGBoost, **AUC-PR 0,997** (dùng tất cả feature).
- **Lớp synthetic đã kiểm định — không leakage:** feature synthetic mạnh nhất
  AUC 0,79; mô hình chỉ-synthetic AUC 0,88 (cao nhưng < 1,0).
- **Lưu ý:** các feature đối soát số dư của PaySim gần như tất định nên bài toán
  gốc rất dễ. Có sẵn **kịch bản "realistic"** tuỳ chọn (bỏ các feature đó →
  `groups="realistic"` trong `features.py`) để có trade-off precision/recall/chi phí
  thực sự nếu nhóm muốn phân tích sâu hơn.
- Mọi giả định điều chỉnh được nằm ở `src/config.py` (`SEED`, trọng số chi phí) và
  `GEN` trong `src/synth_context.py`.

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
