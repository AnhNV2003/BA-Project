# Plan: Task 3 (Module 3) — Data Cleaning trên PaySim

## Context

Input chính là `data/processed/transactions_context.parquet` từ M1. PaySim khá
sạch (0 NaN, 0 duplicate, 0 negative amount, chỉ 16 zero-amount trên full), nên
M3 tập trung vào **validation + documentation + before/after**, không phá tín hiệu
fraud (balance reconciliation là signal, không phải rác).

## Scope (map đúng 3 việc đề bài)

### 1. Missing values / duplicates
- Validate missing values (kỳ vọng 0) — ghi nhận theo cột.
- Check duplicate base transactions trên `PAYSIM_COLUMNS` (giữ bản đầu nếu có).

### 2. Inconsistent categories (đề bài yêu cầu — bổ sung)
- `type`: validate thuộc {PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT}.
- Synthetic categoricals: `browser`, `device_os`, `billing_country` — validate thuộc tập giá trị generator (`_BROWSERS`/`_OS`/`_COUNTRIES` trong `synth_context.py`); chuẩn hóa strip/case nếu cần.
- Ghi nhận mọi giá trị lạ (kể cả khi không tìm thấy — vẫn document là "no inconsistency found").

### 3. Outliers / invalid values
- `amount`: check negative (0) / zero (16); giữ, thêm cờ `flag_zero_amount`.
- Document amount outliers (p99/p99.9, max 92M) — **KHÔNG clip** (amount lớn là tín hiệu fraud, fraud giá trị cao); thêm helper `amount_capped` (document, không xóa dòng).
- `step`: validate 1..743 (timestamp hợp lệ).
- Preserve balance-reconciliation quirks (errorBalance*, orig_drained) — **document, không sửa** (predictive signal).

## Bàn giao cho M4 (guardrail, không phải feature engineering)

- **Giữ lại** `nameOrig`, `nameDest`, `step` (đừng drop như ID/PII) — M4 cần để build historical `nameDest` features past-only.
- Dedup trên `PAYSIM_COLUMNS` **bảo vệ aggregation** M4 (giao dịch trùng sẽ thổi phồng `dest_txn_count`).

## Active File

| File | Role |
|---|---|
| `src/cleaning.py` | Load processed PaySim, validate/clean conservatively, write cleaned parquet + before/after report |

## Outputs

- `data/processed/transactions_clean.parquet`
- `docs/cleaning_report.md` — **phải có bảng before/after**: row count, missing per col, category value-set, zero/neg amount, outlier count, + danh sách "Decisions".

## Verification

```bash
python src/cleaning.py
```

- Row count giữ nguyên trừ khi có dup/neg thật; fraud prevalence không đổi.
- Report có **before/after** cho từng bước; title là PaySim.
- Category validation ghi rõ (kể cả "no inconsistency").
- Balance-error rows được document + preserve (không bị "sửa").
