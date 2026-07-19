# Kịch bản xác định hệ số Cost Model (worksheet tự nghiên cứu)

> Mục tiêu: thay vì đặt bừa `COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE /
> COST_MANUAL_REVIEW`, bạn tự đi tra cứu → ghép vào công thức → ra con số **có
> nguồn dẫn**, biện luận được trong báo cáo. Điền vào các ô `___` rồi cập nhật
> `src/config.py`.

---

## 0. Nguyên tắc nền (đọc trước khi tra cứu)

**a. Công thức tổng** (đang dùng ở `src/train_validate.py`):
```
Net cost = COST_FALSE_NEGATIVE × Σ amount(FN)   # tiền mất do bỏ sót fraud (biến theo số tiền)
         + COST_FALSE_POSITIVE × count(FP)        # phí chặn nhầm khách thật (đếm)
         + COST_MANUAL_REVIEW  × count(flagged)   # phí review MỌI ca bị gắn cờ (đếm)
```

**b. Mỗi outcome ↔ chi phí thực:**
| Kết cục | Ý nghĩa | Chi phí gánh |
|---|---|---|
| FN | fraud lọt lưới | ~toàn bộ `amount` × hệ số FN |
| FP | khách tốt bị chặn | friction (FP) + review |
| TP | fraud bị bắt | chỉ review (tránh được `amount`) |
| TN | khách tốt cho qua | 0 |

**c. CHỈ TỈ LỆ quan trọng.** Nhân cả 3 hệ số với cùng một số → threshold tối ưu
KHÔNG đổi. Nên đừng ám ảnh giá trị tuyệt đối; điều quyết định là **tương quan**
giữa "tiền fraud mất" và "phí 1 FP / 1 review". Vì thế nên chọn 1 hệ số làm
**mỏ neo** (khuyến nghị: review cost, dễ tính nhất) rồi suy các hệ số kia theo nó.

**d. Đơn vị phải đồng nhất.** FN nhân với `amount` (VND/USD), nên FP và review
cũng phải cùng đơn vị tiền tệ đó. Chốt 1 đơn vị (vd USD) cho cả 3.

---

## 1. `COST_MANUAL_REVIEW` — làm ĐẦU TIÊN (mỏ neo, dễ nhất)

**Ý nghĩa:** chi phí để 1 analyst xử lý 1 giao dịch bị gắn cờ.

**Công thức:**
```
COST_MANUAL_REVIEW = (lương giờ có tải đầy đủ) / (số ca xử lý mỗi giờ)
lương giờ có tải đầy đủ = lương cơ bản/giờ × hệ số overhead
```

**Cần tra 3 con số:**
| Ký hiệu | Cần tìm | Gợi ý nguồn | Giá trị bạn tìm được |
|---|---|---|---|
| `S` | Lương/giờ analyst fraud/AML/risk ops | VietnamWorks, ITviec, TopCV, Glassdoor (lọc "Fraud Analyst"/"AML Analyst") → chia cho ~176 giờ/tháng | `___` |
| `O` | Hệ số overhead (bảo hiểm, chỗ ngồi, quản lý, phần mềm) | Chuẩn ngành 1.25–1.4× lương gross | `___` |
| `R` | Số ca 1 analyst review được mỗi giờ | Ước từ SLA đội (vd 3–8 phút/ca → 8–20 ca/giờ); nêu giả định rõ | `___` |

**Ghép số:** `COST_MANUAL_REVIEW = (S × O) / R = ___`

> Ví dụ minh họa (bạn thay bằng số thật): S=$18/giờ, O=1.3, R=8 → (18×1.3)/8 ≈ **$2.9/ca**.

---

## 2. `COST_FALSE_NEGATIVE` — hệ số nhân với `amount`

**Ý nghĩa:** bỏ sót 1 fraud mất bao nhiêu so với số tiền giao dịch.
Vì công thức đã nhân với `amount`, hệ số này = **phần trăm không thu hồi được**
(+ phí cố định quy về tỉ lệ nếu muốn).

**Công thức:**
```
COST_FALSE_NEGATIVE = (1 − recovery_rate) + (phí_cố_định_mỗi_vụ / amount_trung_bình_fraud)
```
- `recovery_rate` = tỉ lệ tiền lấy lại được sau khi fraud xảy ra.
- Phần thứ 2 (phí chargeback/điều tra) thường nhỏ so với amount PaySim → có thể bỏ qua, giữ mô hình đơn giản.

**Cần tra:**
| Ký hiệu | Cần tìm | Gợi ý nguồn | Giá trị |
|---|---|---|---|
| `recovery_rate` | Tỉ lệ thu hồi cho **transfer/push-payment fraud** (giống PaySim: tiền đã chuyển đi) | UK Finance "Annual Fraud Report" (APP scam recovery), FBI IC3 Report, báo cáo NHNN/ngân hàng nếu có | `___` |
| `fee` | Phí chargeback/điều tra mỗi vụ (nếu tính) | Biểu phí Visa/Mastercard chargeback ($15–100), Chargebacks911 | `___` |
| `avg_fraud_amt` | Số tiền fraud trung bình — **lấy TỪ DATA của bạn** | `docs/eda_summary.md` / tự tính trên PaySim | `___` |

**Ghép số:** `COST_FALSE_NEGATIVE = (1 − recovery_rate) + fee/avg_fraud_amt = ___`

> Với push-payment fraud, recovery thường THẤP → hệ số gần 1.0. Nếu bạn tìm được
> recovery_rate ≈ 0 → giữ **1.0** là hợp lý và bảo thủ.

---

## 3. `COST_FALSE_POSITIVE` — khó nhất, cần biện luận kỹ

**Ý nghĩa:** chặn nhầm 1 khách tốt phá hủy bao nhiêu giá trị.

**Công thức:**
```
COST_FALSE_POSITIVE = margin_mất + P(churn) × CLV + support_cost
  margin_mất  = giá trị đơn TB × biên lợi nhuận gộp   (doanh thu mất ngay lần đó)
  P(churn)    = xác suất khách bỏ đi sau khi bị chặn nhầm
  CLV         = customer lifetime value
  support_cost= phí chăm sóc/khiếu nại cho ca bị chặn nhầm
```

**Cần tra:**
| Ký hiệu | Cần tìm | Gợi ý nguồn | Giá trị |
|---|---|---|---|
| `AOV` | Average order/transaction value | data của bạn hoặc benchmark ngành | `___` |
| `margin%` | Biên lợi nhuận gộp của ngành | báo cáo tài chính ngành/ecommerce benchmark | `___` |
| `P(churn)` | % khách rời bỏ sau 1 false decline | Javelin "false declines", LexisNexis "True Cost of Fraud", Aite-Novarica | `___` |
| `CLV` | Giá trị vòng đời khách | công thức CLV hoặc benchmark | `___` |
| `support` | Phí xử lý khiếu nại/ca | dùng lại logic mục 1 (thời gian CSKH × lương) | `___` |

**Ghép số:** `COST_FALSE_POSITIVE = AOV×margin% + P(churn)×CLV + support = ___`

> Cảnh báo: literature cho thấy false decline có thể tốn HÀNG CHỤC USD/vụ vì mất
> khách. Nếu con số bạn ra quá nhỏ (< review cost) là dấu hiệu chưa tính churn/CLV.

---

## 4. Kiểm tra tương quan (quan trọng cho việc hiểu kết quả)

Sau khi có 3 số, tính **"bao nhiêu tiền fraud = 1 false positive"**:
```
break-even amount = COST_FALSE_POSITIVE / COST_FALSE_NEGATIVE
```
→ Nếu 1 giao dịch nghi ngờ có `amount` lớn hơn ngưỡng này thì "thà chặn nhầm còn
hơn bỏ sót". So sánh với **median fraud amount** trong data của bạn:
- median fraud amount PaySim rất lớn (~$441k) ≫ break-even → model sẽ **chặn rất
  rộng** (recall cao, precision thấp). Đây là hệ quả HỢP LÝ của cost, không phải
  lỗi model. Ghi rõ điều này trong báo cáo.

Điền: break-even = `___`; median fraud amount = `___`; nhận xét: `___`.

---

## 5. Sensitivity analysis (để báo cáo "khoa học", thừa nhận bất định)

Vì 3 hệ số là giả định, đừng chỉ báo cáo 1 điểm. Quét tỉ lệ và cho thấy kết luận
robust:

1. Cố định `COST_FALSE_NEGATIVE` và `COST_MANUAL_REVIEW`.
2. Cho `COST_FALSE_POSITIVE` chạy qua nhiều giá trị (vd 10, 25, 50, 100, 200).
3. Với mỗi giá trị: chạy lại chọn threshold trên validation → ghi threshold,
   precision, recall, flagged_rate, expected_cost.
4. Vẽ đường: FP-cost (trục x) vs recall & precision & flagged_rate (trục y).
5. Chỉ ra "vùng ổn định": khoảng FP-cost mà operating point không đổi nhiều.

> Muốn tự động hóa bước này, xem mục 7 (tôi có thể viết script `src/cost_sensitivity.py`).

---

## 6. Nguồn tham khảo nên tra (tìm ấn bản MỚI NHẤT)

**Chi phí fraud tổng / hệ số nhân:**
- LexisNexis Risk Solutions — *True Cost of Fraud Study* (chi phí mỗi $1 fraud)
- Nilson Report — thiệt hại gian lận thẻ toàn cầu
- Federal Reserve — *Payments Study*

**Recovery rate / push-payment (giống PaySim):**
- UK Finance — *Annual Fraud Report* (APP scam, tỉ lệ hoàn tiền)
- FBI IC3 — *Internet Crime Report*
- Báo cáo NHNN / hiệp hội ngân hàng VN (nếu có, để nội địa hóa)

**False decline / churn (cho FP):**
- Javelin Strategy & Research — *false declines*
- Aite-Novarica Group — merchant fraud & declines

**Phí chargeback:**
- Biểu phí chargeback của Visa/Mastercard; Chargebacks911

**Lương analyst (nội địa hóa cho FP & review):**
- VietnamWorks, ITviec, TopCV, Glassdoor — "Fraud Analyst" / "AML Analyst"

**Gợi ý câu tìm kiếm:**
- `"true cost of fraud" LexisNexis 2024 filetype:pdf`
- `APP fraud recovery rate UK Finance annual report`
- `false decline customer churn rate Javelin`
- `Visa chargeback fee schedule`
- `mức lương fraud analyst Việt Nam`

---

## 7. Chốt kết quả → cập nhật code

Sau khi có số, sửa `src/config.py` (dòng 55–60) và **ghi nguồn ngay trong comment**:
```python
# Nguồn: <báo cáo/URL> — recovery_rate=<...>, ...
COST_FALSE_NEGATIVE = ___
# Nguồn: AOV=<...>, margin=<...>, P(churn)=<...>, CLV=<...>
COST_FALSE_POSITIVE = ___
# Nguồn: lương=<...>, overhead=<...>, throughput=<...> ca/giờ
COST_MANUAL_REVIEW  = ___
```
Rồi chạy lại: `./train.sh` (hoặc `./train.sh full`) để threshold/kết quả cập nhật
theo cost mới.

**Checklist báo cáo:**
- [ ] Mỗi hệ số có công thức + con số + nguồn dẫn.
- [ ] Nêu rõ đơn vị tiền tệ thống nhất.
- [ ] Giải thích tương quan → vì sao model chặn rộng (precision thấp).
- [ ] Có bảng/biểu sensitivity analysis.
- [ ] Nêu giới hạn: đây là giả định, số thật cần dữ liệu vận hành nội bộ.
