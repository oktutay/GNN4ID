# Báo cáo: Sửa Data Leak & Resample trong pipeline XG-NID

**Ngày:** 2026-06-21
**Phạm vi:** pipeline xử lý dữ liệu CIC-IoT2023 của dự án (`GNN4ID/`)

---

## 1. Tóm tắt

Khi kiểm tra (verify) pipeline, phát hiện **3 vấn đề methodology** gây leak và resample không hợp lý.
Đối chiếu git cho thấy các lỗi này **đến từ chính paper XG-NID và code gốc của tác giả**
(`upstream = Yasir-ali-farrukh/GNN4ID`), dự án kế thừa nguyên xi — **không phải do dự án tạo ra**.

| # | Vấn đề | Mức độ | Nguồn gốc | Trạng thái |
|---|--------|--------|-----------|------------|
| 1 | Test trùng byte với train (random split + không dedup) | Nghiêm trọng | paper + code gốc | ✅ Đã sửa (data hiện có) + ✅ sửa mã nguồn |
| 2 | Oversample bằng nhân bản dòng thô (`duplicate_rows`) | Nghiêm trọng | code gốc | ✅ Thay bằng `class_weight` |
| 3 | Rolling features tính trước khi split (ghi đè file) | Vừa | notebook gốc | ✅ Sửa mã nguồn (cần raw để chạy) |

**Kết quả đo được sau khi làm sạch data hiện có:**
- Train: 160,000 → **112,343** dòng (loại 47,657 bản sao giả, ~29.8%).
- Test: 25,557 → **22,211** dòng (loại **3,346 dòng leak = 13.09%**, riêng DoS leak 46.5%).
- Leak test↔train: 13.09% → **0%**.

---

## 2. Nguyên nhân chi tiết

### Vấn đề 1 — Test trùng byte với train (leak nghiêm trọng nhất)

**Cơ chế.** `split_csv()` ([Functions.py](Utility/Functions.py)) chia train/test bằng
`df.sample(random_state=42)` — **random split ở mức dòng**, áp dụng riêng từng sub-type.
CIC-IoT2023 là dữ liệu tấn công nên sinh **rất nhiều flow byte-identical**. Random split trên
tập đầy bản sao + **không dedup** ⇒ một flow có thể vừa ở train vừa ở test.

`split_csv` chỉ đảm bảo disjoint *index*, không disjoint *nội dung*. Đo trực tiếp trên 2 file cuối:

```
Test rows identical to a train row: 3346 / 25557 = 13.09%
   Label 5 (Dos):       46.55%   <-- gần một nửa test là bản sao của train
   Label 2 (Spoofing):  29.85%
   Label 3 (Recon):       6.10%
```

**Hệ quả.** Model chỉ cần "nhớ" train là nhận ra ~13% test (tới 46% với DoS) ⇒ accuracy/F1 bị thổi phồng.

**Đối chiếu paper.** Section 3.2.1: *"Initially, 20% of the data samples were set aside to form the
test set..."* — random %, **không nhắc** dedup hay split theo group/thời gian. Table 4 khớp chính
xác data của dự án (WebBased test=1,090; BruteForce test=467).

---

### Vấn đề 2 — Oversample bằng nhân bản dòng thô (resample không hợp lý)

**Cơ chế.** `duplicate_rows()` kéo lớp thiểu số lên 20k bằng **copy nguyên dòng** (concat df N lần +
random pick phần dư). Không có augmentation, chỉ là bản sao y hệt. Đo trên train gốc:

```
train exact-duplicate rows: 47657 / 160000  (~29.8%)
   Label 7 (BruteForce): 20000 dòng nhưng chỉ  1869 unique  (90.7% là bản sao)
   Label 1 (WebBased):   20000 dòng nhưng chỉ  4358 unique  (78.2%)
   Label 5 (Dos):                              12632 unique  (36.8%)
   Label 2 (Spoofing):                         14591 unique  (27.0%)
```

**Hệ quả.** "Cân bằng 20k/lớp" là **đa dạng giả**; BruteForce thực chất chỉ có 1,869 flow lặp ~10
lần ⇒ overfit lớp thiểu số; `train_acc` in trong vòng lặp bị thổi phồng vì lặp graph y hệt; còn làm
tăng xác suất tồn tại "bản sao test" ở Vấn đề 1.

**Đối chiếu.** Paper chỉ nói mơ hồ *"oversampling techniques"*. Code gốc (`duplicate_rows`, commit của
Yasir-ali-farrukh) cho thấy đó là nhân bản thô.

---

### Vấn đề 3 — Rolling features tính trước khi split

**Cơ chế.** Notebook gốc gọi `additional_features(file)` trên **toàn bộ file sub-type (sort theo thời
gian)** rồi **ghi đè** ([Additional_Features.py:68](Utility/Additional_Features.py#L68)), *trước* khi
`split_csv`. Rolling là cửa sổ trailing (chỉ quá khứ) nên **không leak tương lai**, nhưng vector đặc
trưng của một flow test vẫn được tính từ các flow lân cận có thể nằm trong train ⇒ ghép cặp train/test.
Việc ghi đè còn khiến **không thể tách lại split sạch** về sau.

**Khắc phục đúng (thứ tự):** `split_csv()` → `additional_features(train)` và `additional_features(test)`
**độc lập từng tập**.

---

### Điểm dự án này đã làm TỐT HƠN code gốc

`split_csv` gốc của tác giả có **bug no-op** ở nhánh Benign (`df[...]` không gán lại vào `df`) ⇒ thực tế
**không loại** benign flow chạm MAC attacker dù paper nói có. Dự án đã fix thành
`df = df[~src_is_attacker & ~dst_is_attacker]`. Lọc theo MAC attacker (paper Table 3) cũng đúng.

---

## 3. Những gì đã sửa

### 3.1. Sửa được NGAY trên data hiện có — `clean_existing_data.py` (mới)

Script salvage chạy trên 2 CSV combined hiện có (`df_class_8_{train,test}.csv`):

- **Defect A (Vấn đề 2):** `drop_duplicates()` trên train ⇒ trả về **flow unique thật**.
- **Defect B (Vấn đề 1):** loại khỏi test mọi dòng **byte-identical** với một dòng train.
- Xuất `class_weights.json` (công thức balanced) thay cho nhân bản.
- Ghi `df_class_8_train_clean.csv`, `df_class_8_test_clean.csv`.

Kết quả thực tế (đã chạy):

```
                train_before -> train_unique | test_before -> test_clean
  Benign          20000 -> 19916 (99.6%)     | 4000 -> 3976  (-24)
  WebBased        20000 ->  4358 (21.8%)      | 1090 -> 1089  (-1)
  Spoofing        20000 -> 14591 (73.0%)      | 4000 -> 2806  (-1194)
  Recon           20000 -> 19036 (95.2%)      | 4000 -> 3756  (-244)
  Mirai           20000 -> 20000 (100%)       | 4000 -> 4000  (-0)
  Dos             20000 -> 12632 (63.2%)      | 4000 -> 2138  (-1862)
  DDos            20000 -> 19941 (99.7%)      | 4000 -> 3979  (-21)
  BruteForce      20000 ->  1869 (9.3%)       |  467 ->  467  (-0)

  TOTAL train 160000 -> 112343 | test 25557 -> 22211 (leak removed 3346 = 13.09%)
  class_weights = {0:0.705, 1:3.222, 2:0.962, 3:0.738, 4:0.702, 5:1.112, 6:0.704, 7:7.514}
```

### 3.2. Thay nhân bản bằng `class_weight` (Vấn đề 2)

- [Utility/Training.py](Utility/Training.py): `train()` / `train_with_edge_Att()` nhận tham số
  `class_weight`; loss đổi sang `F.nll_loss(pred, label, weight=...)` khi có (giữ nguyên hành vi cũ
  nếu `None`).
- [train.py](train.py): thêm `--class-weights <path>` (đọc `class_weights.json` → tensor) và truyền vào hàm train.
- [build_graphs.py](build_graphs.py): tự ưu tiên CSV `*_clean.csv` nếu tồn tại.

### 3.3. Sửa mã nguồn pipeline cho lần chạy-lại-từ-raw (Vấn đề 1 & 3)

- [Utility/Functions.py](Utility/Functions.py) `split_csv()`: thêm `dedup=True` (loại bản sao **trước**
  split) và `temporal=True` (split theo thời gian: 80% sớm→train, 20% muộn→test, dùng
  `bidirectional_first_seen_ms`; tự fallback random nếu thiếu cột). Giết tận gốc Vấn đề 1.
- [Utility/Functions.py](Utility/Functions.py) `Combining_classes()`: **bỏ** `duplicate_rows`; chỉ dedup
  + cap lớp đa số, giữ nguyên flow unique của lớp thiểu số.
- [Utility/Functions.py](Utility/Functions.py) `duplicate_rows()`: gắn nhãn **DEPRECATED** + cảnh báo.
- [Utility/Additional_Features.py](Utility/Additional_Features.py) `additional_features()`: thêm
  `out_file=` để **không ghi đè** file gốc, kèm cảnh báo thứ tự **phải chạy sau split, từng tập riêng**.

---

## 4. Những gì KHÔNG thể sửa trên data hiện có (và lý do)

`raw/` rỗng và các CSV combined đã **drop** `src_ip/dst_ip/src_mac/dst_mac/timestamp`. Do đó:

- **Group/temporal split thực thụ** trên data hiện có: không còn khóa thời gian/IP/MAC ⇒ chỉ làm được
  dedup + loại leak exact-match (đã làm). **Near-duplicate** leak còn sót lại chỉ trị được khi chạy lại
  từ raw bằng `split_csv(temporal=True)`.
- **Tính lại rolling sau split**: cột NFStream gốc đã mất, rolling đã "nướng" vào file ⇒ phải chạy lại
  từ pcap/NFStream CSV với pipeline đã sửa.
- **Retrain/đánh giá lại metric tại đây**: môi trường chưa cài `torch`/`torch_geometric`. Code đã sẵn
  sàng để chạy (xem mục 5).

> Để khôi phục đầy đủ Vấn đề 1 & 3 cần **chạy lại từ raw NFStream CSV** (có IP/MAC/timestamp) qua
> pipeline đã sửa: `split_csv()` → `additional_features(train/test riêng)` → `Combining_classes()`.

---

## 5. Cách chạy lại & đánh giá metric (Vấn đề 4)

```bash
# 1) Làm sạch data hiện có (đã chạy; tạo *_clean.csv + class_weights.json)
python3 clean_existing_data.py

# 2) Build graph từ CSV sạch (tự ưu tiên *_clean.csv)
python3 build_graphs.py

# 3) Train với class weight thay vì nhân bản, rồi đánh giá trên test sạch
python3 train.py --data-root ../data/CIC_IoT2023_Processed_Data \
    --skip-processing --model edge \
    --class-weights ../data/CIC_IoT2023_Processed_Data/class_weights.json
```

**Kỳ vọng:** accuracy/F1 trên test sạch sẽ **thấp hơn** số trong báo cáo cũ (đã loại 13% test ăn gian +
bỏ nhân bản). Đây mới là con số phản ánh đúng khả năng tổng quát hóa, và là điểm có thể nêu như
**critique/đóng góp** so với paper XG-NID gốc.

---

## 6. Danh sách file thay đổi

| File | Loại | Nội dung |
|------|------|----------|
| `clean_existing_data.py` | mới | Dedup train + loại leak test, xuất CSV sạch + class weights |
| `BAOCAO_FIX_DATALEAK_RESAMPLE.md` | mới | Báo cáo này |
| `Utility/Training.py` | sửa | Thêm `class_weight` cho loss |
| `train.py` | sửa | `--class-weights`, nạp & truyền tensor |
| `build_graphs.py` | sửa | Ưu tiên CSV `*_clean.csv` |
| `Utility/Functions.py` | sửa | `split_csv` (dedup+temporal); `Combining_classes` (bỏ nhân bản); `duplicate_rows` deprecated |
| `Utility/Additional_Features.py` | sửa | `out_file=` không ghi đè + cảnh báo chạy sau split |
