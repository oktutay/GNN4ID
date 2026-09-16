# Báo cáo hoàn thiện pre-process GNN4ID / XG-NID (v2, 16/09/2026)

*(Đường dẫn file trong báo cáo tính từ thư mục `GNN4ID/`; các báo cáo khác nằm cùng thư mục `docs/`; EDA và khảo sát ở `XG_NID/`.)*

Phạm vi: rà lại toàn bộ khối tiền xử lý CIC-IoT2023 của bản fork `XG_NID/GNN4ID` (Flow and Feature
Generator → Explainable Feature Extractor → Data preprocessing), đối chiếu với paper XG-NID, với code gốc
của tác giả (upstream `Yasir-ali-farrukh/GNN4ID`, commit `551d1f1`) và với dữ liệu tác giả công bố trên
Google Drive; sửa lỗi, viết lại pipeline một lệnh, kiểm chứng trên 2 pcap local và đóng gói zip để chạy ở
máy khác.

## 1. Kết luận ngắn

| # | Câu hỏi | Trả lời |
| --- | --- | --- |
| 1 | Code có outdate so với tác giả không? | **Không.** `main` local == upstream tip `551d1f1` (23/02/2026). Thay đổi code cuối của tác giả là 22/11/2024 (`idle_timeout` 5 → 120). Khác biệt giữa local và tác giả là do bản viết lại 21/06 (16 cột Table 1 → 69 feature), không do upstream mới hơn. |
| 2 | "350 packet hay 350 flow?" | Paper **không ghi số 350** ở bất kỳ đâu; Table 1 và Algorithm 1 chỉ nói "rolling **time** window W", Sec 3.1.2 gợi ý "over the last minute". Số 350 và đơn vị **dòng flow** là của code tác giả (`rolling(window=350)` trên CSV flow, sort theo `bidirectional_first_seen_ms`, group theo `dst_ip` và cặp src-dst). Người dùng đúng về code; giả định "paper = 350 packet" không có trong paper. |
| 3 | 76 hay 82 flow feature? | Cả hai đúng ở hai giai đoạn khác nhau: **76 = NFStream raw (77 cột − `id`)** đúng như paper "76 flow-level + 14 packet-level"; sau khi drop 29 cột định danh + 7 one-hot + 28 rolling + `packet_size_variation` ⇒ **82** = CSV Drive = checkpoint. Bản 21/06 chỉ sinh 16 cột ⇒ 69, không tương thích. |
| 4 | Có leak không? | Bản 21/06 đã sửa leak trùng lặp (13% test trùng train) nhưng **tạo ra một lỗi mới**: rolling tính SAU khi undersample train ⇒ cùng cột hai nghĩa giữa train/test (mục 2.1). Bản v2 sửa lại thứ tự và thêm dedup mức feature toàn cục. |
| 5 | Split rồi mới up/down-sampling? | **Có**: cap/undersample sau split; oversample chỉ train, sau dedup (Table 4 của paper); test không bao giờ nhân bản; `class_weights.json` luôn được xuất để chạy biến thể không nhân bản. |

## 2. Lỗi tìm thấy và cách sửa

### 2.1 Rolling tính sau khi undersample (lỗi do bản fix 21/06 tạo ra) — **nghiêm trọng**
`split_csv` (temporal) sample train về 20k dòng/file rồi `additional_features(train)` mới chạy ⇒ cửa sổ
350 dòng của train trải trên chuỗi đã **thưa** (350 dòng sample ≈ hàng chục nghìn flow thật), còn test tính
trên 4000 flow **dày liên tiếp** cuối file ⇒ cùng một cột (vd `Rolling_TCP_Requests_Destination`) mang hai
đơn vị; giá trị bão hoà 350 (xuất hiện ở 99% dòng DoS trong data tác giả) không bao giờ có ở train; đầu mỗi
file test còn cold-start (`min_periods=1`).

Rolling là **causal** (chỉ nhìn các flow trước đó) và **không dùng nhãn**, nên tính trên toàn file rồi mới
temporal split **không leak**: một dòng test chỉ phụ thuộc vào quá khứ của chính nó (đúng điều kiện deploy),
không dòng train nào phụ thuộc vào test. Thứ tự chốt: extract → **features (toàn file)** → split (temporal)
→ combine → dedup → oversample train → class8. Xem docstring `Utility/Additional_Features.py`.

### 2.2 Schema rolling không khớp tác giả
Bản 21/06 sinh 16 cột tên Table 1; tác giả sinh 28 cột (`*_Destination` / `*_SourceDestination`) +
`packet_size_variation`. **Khôi phục đúng 28 cột, đúng tên, đúng thứ tự** (`Utility/Schema.py::ROLLING_28`),
kiểm chứng **bit-identical** với code upstream trên 2 pcap local (`Debug/verify_preprocessing.py` check 5,
0/15,313 dòng lệch). Tuỳ chọn `schema='table1'` giữ 15 cột theo đích với tên Table 1.

Đồng thời ghi rõ 3 điểm code tác giả lệch paper (giữ nguyên để tái lập, có cờ để đổi): (a) UDP/TCP/ICMP/
HTTP/DNS/vuln đếm **flow** còn ACK/FIN/RST/PSH/SYN đếm **packet** (`count_mode='packets'` để nhất quán);
(b) `Unique_Ports_In_SourceDestinationIP` đếm `dst_port` (paper: source ports); (c) `Rolling_vulnerable_port`
group theo cặp (paper: theo đích).

### 2.3 Tên file CIC-IoT2023 trên Linux — 2 sub-attack bị rơi, benign bị gán sai
`name_mapping` cũ có key `'DDos-SlowLoris'` (thư mục thật `DDoS-SlowLoris`) và value
`'Webbased-BrwserHijack'`; `Combining_classes` glob `'DDos*'`/`'WebBased*'` phân biệt hoa thường trên Linux
(Windows thì không, nên tác giả không thấy) ⇒ SlowLoris và BrowserHijacking bị bỏ âm thầm. File benign
thật là `BenignTraffic*.pcap` (thư mục `Benign_Final`) không khớp key `'Benign'` ⇒ `split_csv` coi benign
là **attack** (giữ flow chạm MAC attacker) và `label_dict['BenignTraffic_N']` KeyError.
**Sửa:** `resolve_class_subtype()` — longest-prefix, không phân biệt hoa thường, trên 34 tên thư mục CIC +
tên canonical của tác giả; `assert_all_resolvable()` chặn file lạ ngay từ đầu; test `tests/test_resolver.py`.

### 2.4 `Combining_classes`
Biến `file`/`name_file` rò từ vòng lặp; `os.remove` file test (không chạy lại được); sample ngẫu nhiên
không giữ tỷ lệ sub-attack dù paper nói "proportional representation across subclasses". Dữ liệu Drive
đo được: **DDoS train/test = 100% ICMP, 99.8% flow đủ 20 packet; DoS = 100% TCP, 1–2 packet, 0 payload;
Mirai = 100% UDP, 20 packet** ⇒ một lookup 3 feature `(proto, pk==20, pk<=2)` đã đúng 65.5% (train).
**Sửa:** 3 pass (train: dedup → cap theo tỷ lệ sub-attack; test: loại dòng trùng train của MỌI lớp → cap;
train: oversample theo tỷ lệ), không xoá file, không biến rò, chịu lớp vắng.

### 2.5 Dedup mức raw gần vô dụng
`drop_duplicates()` trên dòng thô (có `id`, timestamp, port) hầu như không bắt được flood trùng; dedup có
nghĩa phải ở **feature-view sau khi drop 29 cột** (chính là cách `clean_existing_data.py` tìm ra 13% test
trùng train). **Sửa:** `row_keys()` (hash 64-bit trên feature-view) dùng ở `split_csv` (train pool) và
`Combining_classes` (toàn cục); `dedup_across_splits()` dùng chung.

### 2.6 Notebook vẫn theo thứ tự cũ, ghi đè in-place
6 notebook gọi `additional_features(file)` ghi đè file thô (chạy lần 2 crash vì `get_dummies` đã ăn
`protocol`), rồi `split_csv` ghi đè tiếp. **Sửa:** mọi stage ghi thư mục riêng (`features/`, `split/`,
`combined/`), guard idempotent trong `additional_features`, notebook được sửa bằng nbformat (mục 4).

### 2.7 Nhỏ hơn
`GNN4ID_Model.ipynb`/`3_*_pcap.ipynb` dựng test dataset từ glob **train** (chỉ vô hại nhờ
`skip_processing=True`) — sửa dùng `Files_test`. `Feature_extractor` chỉ chạy được với `Destination_path`
có dấu `/` cuối — sửa `os.path.join`. Cờ `--n-dissections` (L7 NFStream: SNI, JA3, user-agent) và
`--include-packetflag` (8 cờ TCP vào packet node) được đưa ra CLI; mặc định giữ nguyên paper.

## 3. Pipeline v2 — `run_preprocessing.py`

```
1 extract   pcap → raw/<Class>-<Short>_<n>.csv       NFStream limit=20 idle=120 n_dissections=0
2 features  raw → features/<stem>.csv                 28 rolling + packet_size_variation + 7 one-hot, TOÀN file
3 split     features → split/{train,test}/<stem>_*    lọc MAC → 20% cuối = test pool (cap 4000, chọn ngẫu nhiên
                                                      trong pool hoặc --test-pick tail) → 80% đầu = train pool
                                                      → dedup feature-view → cap 20000/file
4 combine   split → combined/{train,test}/<Class>_*   dedup → cap 20k/4k theo tỷ lệ sub-attack → loại test∩train
                                                      (toàn cục) → oversample train lên 20k → class_weights.json
5 class8    combined → df_class_8_{train,test}.csv    drop 29 cột định danh → assert header 97 cột == Drive
6 graphs    (tuỳ chọn) NIDSDataset → processed/*.pt
```
Mỗi stage có thư mục riêng, chạy lại được, `--force` tính lại, `--reset` xoá `out`; `manifest.json` ghi
config (git sha, version thư viện) + số dòng từng file/stage/lớp/sub-attack + `checks`.

Cờ chính: `--window 350 --window-unit flows|time|packets`, `--count-mode author|packets|flows`,
`--schema author82|table1`, `--rolling-scope all|filtered` (mặc định `all` = như tác giả, cửa sổ chứa cả flow
nền chưa lọc MAC), `--test-pick random|tail`, `--no-oversample`, `--dedup-test-within`, `--n-dissections`,
`--keep-l7`, `--include-packetflag`, `--workers`, `--classes Benign,DDos,...`.

Tương thích ngược: `build_graphs.py`, `clean_existing_data.py`, `train.py --skip-processing` và 2 checkpoint
trong `checkpoints/` giữ nguyên hành vi (flow node vẫn 82-d, header 97 cột giống hệt Drive — kiểm chứng
`verify_preprocessing.py` check 1c). `additional_features(file)`/`split_csv(path)`/`Combining_classes(dir, classes,
label_dict=)` giữ chữ ký cũ (kwarg cũ được ánh xạ).

## 4. File đã thay đổi / thêm

| File | Việc |
| --- | --- |
| `Utility/Schema.py` (mới) | hằng số schema: 46/14/28/7 cột, header 97, 29 cột drop, 9 cột L7, 36 tên CIC → lớp, MAC attacker, port list |
| `Utility/Additional_Features.py` (viết lại) | `compute_rolling_features()` + `additional_features(out_file=)`; 3 đơn vị cửa sổ, `count_mode`, `schema`, guard idempotent; engine `BaseIndexer` (bit-identical với tác giả) |
| `Utility/Functions.py` | resolver tên file, `mac_filter`, `row_keys`/`dedup_across_splits`, `proportional_cap`/`oversample_train`, `split_csv` (không rolling, không in-place), `Combining_classes` 3 pass, `build_class8_csvs`, `NIDSDataset` drop cột L7 + nhãn qua resolver |
| `Utility/Feature_extractor_flow_packet_combined.py` | `extract_pcap()` + cờ `--n-dissections --limit --idle-timeout --active-timeout --bpf --out-name` |
| `Utility/Explainer.py` | import `FLOW_FEATURE_NAMES_82` từ Schema |
| `run_preprocessing.py` (mới) | entry-point 6 stage + logging + manifest |
| `build_graphs.py`, `train.py` | `--include-packetflag` |
| `Debug/trace_pipeline.py`, `Debug/autobreak.py`, `.vscode/launch.json` (2 bản), `Debug/README_TRACE.md` | thứ tự stage mới (features → split), cờ mới, breakpoint mới |
| `Debug/verify_preprocessing.py` (mới) | 9 nhóm assertion trên thư mục output |
| `tests/` (mới) | 17 unit test (`test_rolling_units`, `test_resolver`, `test_split_combine`) + bản gốc của tác giả làm fixture |
| 6 notebook | `rename_files()` qua resolver, features → `features/`, split → `split/`, `Combining_classes(... oversample=True, out_dir=combined)`, `build_class8_csvs()` thay 4 cell concat/drop, sửa bug test glob, `FLOW_FEATURE_NAMES_82`; output đã xoá |
| `README.md`, `docs/FEATURES_VA_Y_NGHIA.md` | mục "Preprocessing v2", flow node 82 cột, bảng 28 cột rolling |
| `make_preprocess_zip.py` (mới) | đóng gói code (không data/checkpoint) thành `dist/gnn4id_preprocess_<sha>_<ngày>.zip` |

## 5. Kiểm chứng đã chạy (máy này, 2 pcap: XSS, DictionaryBruteForce)

- `python -m unittest discover -s tests` → **17/17 OK** (flows == code tác giả kể cả tại timestamp trùng;
  time == `pandas.rolling('60s')` trên data không trùng timestamp; packets = suffix nhỏ nhất ≥ N; resolver
  36 tên × 9 hậu tố; split/combine tổng hợp với dòng trùng cấy vào test pool bị loại; header 97).
- `run_preprocessing.py --train-cap 400 --test-cap 100`: XSS 4270 → 222 (MAC) → test 44 / train 178 → 400;
  BruteForce 11043 → 2184 → test 100 / train 1747 → 400; class8 train 800×97, test 144×97;
  `checks: test_keys_in_train=0`.
- `Debug/verify_preprocessing.py --drive-train df_class_8_train.csv` → **14/14 PASS**, trong đó header
  == Drive (strip `\r`), rolling của split == của `features/` (join theo `id`), **code tác giả 551d1f1 tái lập
  100% `features/`** (0 dòng lệch / 15,313), ranh giới thời gian chặt, oversample chỉ tăng train.
- Biến thể đều chạy và giữ header 97 cột: `--window 60s --window-unit time`, `--window-unit packets
  --count-mode packets`, `--no-oversample` (train 578 unique, `class_weights.json` {1: 1.62, 7: 0.72}),
  `--n-dissections 20 --keep-l7` (raw 100 cột, class8 106 cột), `--schema table1` (class8 84 cột).
- `build_graphs.py` → flow.x (1, 82), packet.x (≤20, 1500); `--include-packetflag` → (≤20, 1508);
  `train.py --skip-processing --epochs 1` chạy cả 2 biến thể; `Debug/trace_pipeline.py --stage all --reset`
  hết 6 stage với thứ tự mới.

## 6. Chạy trên máy khác

```bash
unzip gnn4id_preprocess_<sha>_<ngày>.zip -d GNN4ID && cd GNN4ID
conda create -n xmfgnn python=3.10 -y && conda activate xmfgnn && pip install -r requirement.txt
# giải nén PCAP.zip của CIC-IoT2023 (34 thư mục, ~548 GB) rồi:
python run_preprocessing.py --pcap-dir /data/CIC_IoT2023/PCAP --out-dir /data/xgnid_pp --workers 4
python Debug/verify_preprocessing.py --out-dir /data/xgnid_pp
python build_graphs.py /data/xgnid_pp && python train.py --data-root /data/xgnid_pp --skip-processing
```
Lưu ý: (1) tên file phải resolve được — nếu archive có tên lạ, `assert_all_resolvable` sẽ liệt kê và bạn thêm
vào `Utility/Schema.py::CIC_SUBTYPES`; (2) file DDoS nhiều triệu flow: stage `features` đọc cả file
(cột `udps.payload_data` nặng) — cần RAM tương ứng hoặc chạy `--classes` từng nhóm; `--train-pool-cap-per-file`
giữ mọi thứ sau split nhỏ; (3) số benign sẽ khác Drive (1,306,976) vì lọc MAC benign nay hoạt động và
`BenignTraffic*` resolve đúng; (4) tại timestamp trùng nhau, thứ tự dòng theo quicksort của pandas — dùng cùng
lời gọi sort như tác giả nên cùng input cho cùng output.

## 7. Còn mở
- `--low-memory` (đọc 2 pass) chưa làm; `Unique_Ports_*` dùng `rolling.apply` Python như tác giả (chậm nhất).
- Cửa sổ thời gian `60s` là lựa chọn ngoài paper (paper chỉ "last minute") — để tuỳ chọn, không mặc định.
- Các vấn đề dữ liệu (payload lộ tên lớp, TLS ciphertext, rolling bão hoà theo lớp, DoS/DDoS tách theo
  thời gian) không phải lỗi code, được phân tích ở `CIC_IoT2023_EDA_v2_VI.md`.
