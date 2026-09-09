# Trace pipeline XG-NID bằng 2 file pcap

Bộ này để **đọc/hiểu code chạy theo thứ tự nào**, không phải để train. Đầu vào là 2 file
trong `data/Debug and Trace/`:

- `DictionaryBruteForce.pcap` (38 MB) → class `BruteForce` (label 7)
- `XSS.pcap` (12 MB) → class `WebBased` (label 1)

2 file pcap gốc **không bị xoá** (notebook `GNN4ID.ipynb` có `os.remove(pcap)` sau khi extract,
driver ở đây bỏ bước đó). Mọi output nằm trong `data/Debug and Trace/trace_out/`.

---

## 1. Cần gì

| Thứ | Giá trị |
|---|---|
| Extension VS Code | **đã cài 2026-09-09**: `ms-python.python` 2026.4.0, `ms-python.debugpy` 2026.6.0, `ms-python.vscode-pylance` 2026.3.1, `ms-python.vscode-python-envs` 1.36.0, `ms-toolsai.jupyter` 2025.9.1 (+ renderers/keymap/cell-tags). Sau khi cài phải **Reload Window** (Ctrl+Shift+P → *Developer: Reload Window*) rồi Ctrl+Shift+P → *Python: Select Interpreter* → chọn `xmfgnn` |
| Python interpreter | `/home/tutay/anaconda3/envs/xmfgnn/bin/python` (conda env `xmfgnn`: nfstream 6.5.3, torch 2.3.1+cu121, torch_geometric 2.5.3) — env `base` **không** có nfstream/torch |
| cwd khi chạy | `.../XG_NID/GNN4ID` (để `from Utility.Functions import *` chạy được) |

`.vscode/` có ở cả `XG_NID/` và `XG_NID/GNN4ID/` nên mở folder nào làm workspace cũng chạy được.
Path trong config dùng **cú pháp biến của VS Code**, không hard-code:

| Biến | Nghĩa |
|---|---|
| `${workspaceFolder}` | folder đang mở (bản ở `XG_NID/` dùng `${workspaceFolder}/GNN4ID`, bản ở `GNN4ID/` dùng `${workspaceFolder}`) |
| `${config:python.defaultInterpreterPath}` | interpreter khai trong `.vscode/settings.json` — **path tuyệt đối duy nhất còn lại**, đổi env chỉ cần sửa 1 dòng đó |
| `${file}` | file đang mở (config *debug the current file*) |
| `${command:pickProcess}` | hộp chọn PID (config *attach*) |
| `${userHome}` | `/home/tutay` (chỗ ghi checkpoint tạm của config `train.py`) |

---

## 2. Cách 1 — tự đặt breakpoint rồi step (trace thủ công)

Config để **chỉ đi trong code của tác giả**:

```jsonc
"justMyCode": true,                                              // bỏ qua nfstream / pandas / torch
"rules": [{"path": "${workspaceFolder}/GNN4ID/Debug/**", "include": false}]   // bỏ qua cả driver
```

→ F11 (step into) chỉ dừng trong `GNN4ID/Utility/*.py`, `train.py`, `build_graphs.py`.
Không phải Step Out liên tục để thoát khỏi thư viện, cũng không lạc vào `Debug/trace_pipeline.py`.

Cách làm: mở file trong `Utility/`, **click lề trái đúng dòng dưới đây** (đặt ở dòng `def` sẽ
không dính — phải là lệnh đầu tiên trong thân hàm), chọn config, F5.

| Config (F5) | Breakpoint | Đứng ở đó xem gì |
|---|---|---|
| `XG-NID 1: extract` | `Feature_extractor_flow_packet_combined.py:10` `on_init`<br>`:63` `on_update` | 1 packet → 1 dòng trong `flow.udps.*`; `packet.payload_size`, `self.limit` (=20 packet/flow) |
| `XG-NID 2: split` | `Functions.py:579` `split_csv` | lọc MAC attacker → dedup → chia 80/20 theo thời gian: `df.shape` sau mỗi bước, `n_test` |
| `XG-NID 3: features` | `Additional_Features.py:66` `additional_features`<br>`:21` `_rolling_sum` | sort theo `bidirectional_first_seen_ms`, groupby `_dst_id`, sinh 16 cột `Rolling_*` |
| `XG-NID 4: combine` | `Functions.py:472` `Combining_classes` | gộp file cùng class, gán `Label` = `label_dict[name_file]`, cap số dòng |
| `XG-NID 5: graphs` | `Functions.py:112` `process`<br>`:220` `_get_flow_node_features`<br>`:243` `_get_packet_node_features`<br>`:285` `_get_contain_edge_features`<br>`:306` `_get_link_edge_features`<br>`:318` `_get_contain_edge_index`<br>`:332` `_get_link_edge_index` | 1 dòng CSV → 1 `HeteroData`: flow node (82 số) + packet node (20×1500 byte payload) + contain edge (20) + link edge (19) |
| `XG-NID 6: model` | `Model.py:68` `HeteroGNN.forward`<br>`Training.py:35` `train`<br>`:80` `test_cm` | 2×GATConv trên 2 loại edge → BN → pool → MLP 8 class |
| `XG-NID 0: cả 6 stage` | bất kỳ dòng nào ở trên | chạy tuần tự cả pipeline, dữ liệu nhỏ |
| `XG-NID: build_graphs.py` / `train.py` | như trên | pipeline thật trên dataset đã xử lý |

Thứ tự stage: `extract → split → features → combine → graphs → model` (mục 4 giải thích dữ liệu
biến đổi ra sao). Muốn hiểu tuần tự thì trace lần lượt config 1 → 6.

### Tự trace bằng tay — phím và mẹo

| Phím | Việc |
|---|---|
| F9 | bật/tắt breakpoint ở dòng đang đứng |
| F5 | start / continue |
| F10 | step over (chạy hết dòng, không vào trong hàm) |
| F11 | step into (vào trong hàm — `justMyCode:true` nên **chỉ** vào code của tác giả, nfstream/pandas/torch/`Debug/` bị step over) |
| Shift+F11 | step out (chạy hết hàm hiện tại rồi dừng ở chỗ gọi) |
| Ctrl+Shift+F5 | restart |
| Shift+F5 | stop |

4 thứ trong panel **Run and Debug** dùng để đọc thứ tự hàm mà không cần tracer tự động:

1. **Call Stack** — đang ở đâu trong chuỗi gọi. Đây chính là "hàm nào gọi hàm nào" theo thời gian thực;
   click từng dòng để xem biến local của frame đó.
2. **Function breakpoint** — trong panel *Breakpoints* bấm `+` rồi nhập tên hàm, ví dụ
   `NIDSDataset._get_packet_node_features` hoặc `additional_features`. Không cần mở file, cứ hàm đó
   chạy là dừng.
3. **Logpoint** (trace không cần dừng) — click phải vào lề trái → *Add Logpoint*, nhập message có
   `{biểu thức}`; mỗi lần dòng đó chạy thì in ra Debug Console mà **không dừng chương trình**:
   - trong `on_init` (`Feature_extractor_flow_packet_combined.py:13`):
     `{flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} proto={flow.protocol}`
   - trong `process()`, dòng đầu thân vòng lặp (`Functions.py:168`, `flow_node_feats = ...`):
     `row {index} label={flow["Label"]} packets={len(flow["udps.payload_data"])}`
4. **Conditional breakpoint / Hit count** — click phải breakpoint → *Edit Breakpoint*. Cần vì
   `on_update` chạy ~147 000 lần:
   - Expression: `packet.payload_size > 500`, hoặc `flow.bidirectional_packets == 19`
   - Hit count: `500` (chỉ dừng ở lần thứ 500)

Biểu thức nên gõ vào **Debug Console** / thêm vào **Watch** khi đã dừng:

| Đang dừng ở | Gõ thử |
|---|---|
| `My_Custom.on_init` / `on_update` | `packet.ip_size`, `packet.payload_size`, `packet.ip_packet[-packet.payload_size:].hex()[:40]`, `len(flow.udps.payload_data)` |
| `split_csv` | `df.shape`, `df['src_mac'].isin(attacker_macs).sum()`, `df['bidirectional_first_seen_ms'].head()` |
| `additional_features` | `data.shape`, `[c for c in data.columns if c.startswith('Rolling')]`, `data.groupby('_dst_id').size().head()` |
| `NIDSDataset.process` | `self.data.shape`, `flow.index.tolist()[:10]`, `flow['udps.delta_time'][:5]` |
| `_get_packet_node_features` | `len(flow['udps.payload_data'])`, `bytes.fromhex(flow['udps.payload_data'][0])[:16]` |
| `HeteroGNN.forward` | `{k: tuple(v.shape) for k, v in x.items()}`, `batch.batch_dict['flow']`, `batch.y` |
| `Training.train` | `pred.shape`, `pred.exp().max(1)`, `label`, `loss.item()` |

Config `XG-NID: xem cả driver nữa (không lọc Debug/, stopOnEntry)` dừng ngay dòng đầu tiên và
**không** lọc `Debug/`, dùng khi muốn xem driver gọi vào `Utility/` ở chỗ nào.

### Vì sao breakpoint trong `on_init`/`on_update` lại dính được

`NFStreamer` bình thường fork 1 process/core (`nfstream/streamer.py`, `self._mp_context.Process(...)`),
nên callback của plugin chạy ở **process con** → breakpoint hay bị miss. Driver mặc định dùng
`--extract-mode inproc`: gọi thẳng `nfstream.meter.meter_workflow(...)` (đúng function mà process con
chạy) trong process hiện tại, đọc flow ra từ 1 `queue.Queue`. Stack khi dừng trông như:

```
trace_pipeline.stage_extract → _flows_in_process → nfstream/meter.py:meter_workflow
   → meter.py:consume → flow.py:NFlow.__init__ → My_Custom.on_init
```

Muốn dùng đúng đường multiprocess của nfstream thì thêm `--extract-mode streamer`.

---

## 3. Cách 2 — chạy thẳng, đọc log thứ tự hàm

Không có debugger thì driver tự bật tracer (`sys.setprofile`) và ghi log:

```bash
cd /home/tutay/Tutay/Tutay_Sec/XG_NID/GNN4ID
/home/tutay/anaconda3/envs/xmfgnn/bin/python Debug/trace_pipeline.py --stage all --reset \
    --max-flows 3000 --max-graphs 120
```

Hoặc trong VS Code: **Terminal → Run Task → `XG-NID trace: all stages + call log`**.

Sinh ra 2 file trong `data/Debug and Trace/trace_out/logs/`:

- `run_<time>.log` — toàn bộ console (mỗi stage in ra file vào/ra, shape, tên cột thêm/bớt…)
- `trace_<time>.trace` — log gọi hàm theo đúng thứ tự thực thi, kèm 2 bảng tổng kết ở cuối

Một dòng trace đọc như sau:

```
[00042] t=  1.317s | | -> NIDSDataset._get_packet_node_features(flow=Series(len=97))   Utility/Functions.py:237
[00043] t=  1.319s | | <- NIDSDataset._get_packet_node_features = Tensor(shape=(20, 1500), dtype=float32)   (2.1 ms)
```

`| |` = độ sâu stack, `->` gọi vào, `<-` trả về (kèm giá trị trả về + thời gian). Mỗi hàm chỉ log
đầy đủ `--trace-max-calls` lần đầu (mặc định 3), sau đó chỉ đếm — nên log vẫn đọc được dù
`on_update` chạy 147k lần. Cuối file:

- **FUNCTIONS IN ORDER OF FIRST CALL** — đúng cái "các func hoạt động lần lượt thế nào"
- **HOTTEST FUNCTIONS** — thời gian dồn ở đâu

Option hay dùng: `--trace-max-calls 20` (log dày hơn), `--trace-echo` (in luôn ra terminal),
`--trace-deps` (trace cả trong nfstream / torch_geometric), `--trace off`.

---

## 4. 6 stage và dữ liệu chạy qua đâu

```
data/Debug and Trace/*.pcap
  │  (1) extract   NFStreamer + My_Custom(limit=20)      → 91 cột, mỗi flow giữ 20 packet
  ▼
trace_out/raw/BruteForce-Dictionary.csv , WebBased-XSS.csv       ← tên theo NAME_MAPPING
  │  (2) split_csv  lọc MAC attacker (paper Table 3) → dedup → chia theo thời gian 80/20
  ├──────────────► trace_out/raw/test/<name>_test.csv            (20% flow mới nhất)
  ▼                (file gốc bị GHI ĐÈ = phần train)
  │  (3) additional_features  chạy RIÊNG cho train và test (nếu chạy chung → leak)
  │      + 14 cột Rolling_*  + one-hot Exp_*/proto_*
  ▼
  │  (4) Combining_classes  dedup + cap + thêm cột Label
  ├──────────────► trace_out/raw/train/<Class>_train.csv , raw/test/<Class>_test.csv
  ▼      rồi notebook cell 11-14: concat + drop 29 cột định danh/biased
trace_out/df_class_8_train.csv , df_class_8_test.csv
  │  (5) NIDSDataset.process   1 dòng flow → 1 HeteroData
  ▼      flow node (1×N feature) | packet node (20×1500 byte payload)
  │      contain edge flow→packet [direction, ip_size, transport_size, payload_size]
  │      link edge packet_i→packet_i+1 [delta_time] , rồi T.ToUndirected()
trace_out/processed/data_*.pt , data_test_*.pt
  │  (6) HeteroGNN_Edge (2× GATConv + BN + LeakyReLU → global_mean_pool → MLP → log_softmax)
  ▼      Training.train (Adam + ReduceLROnPlateau) → Training.test_cm (confusion matrix)
```

Chạy 1 stage riêng (dùng lại file đã có trên đĩa):

```bash
python Debug/trace_pipeline.py --stage graphs --max-graphs 5 --trace-max-calls 20
python Debug/trace_pipeline.py --stage split features --force
```

Option chính: `--max-flows` (số flow giữ lại mỗi pcap, 0 = tất cả), `--max-graphs`
(số graph object tạo ra), `--packet-limit` (packet/flow, paper = 20), `--bpf "tcp port 80"`,
`--epochs`, `--device cpu|cuda|auto`, `--reset` (xoá `trace_out/` làm lại), `--force`.

Lưu ý về thứ tự: `split` ghi đè file raw, `features` không idempotent (chạy lần 2 sẽ lỗi vì
`protocol`/`expiration_id` đã bị get_dummies ăn mất). Driver ghi `trace_out/.trace_state.json`
và **từ chối** chạy lại 2 stage này trên cùng working dir (`--force` để làm bừa). Muốn làm lại
cho đúng: `--reset` rồi `--stage all`. `combine`, `graphs`, `model` chạy lại bao nhiêu lần cũng được
(`graphs` xoá `processed/*.pt` cũ trước khi build lại vì index chạy lại từ 0).

---

## 5. 3 điểm phát hiện khi dựng bộ trace này

1. **`Utility/Functions.py` từng hard-code separator Windows** (`'\\test\\'`, `split('\\')[-1]`).
   Trên Linux `split_csv` sẽ tạo file tên `raw\test\X_test.csv` và `Combining_classes` thì crash.
   Đã đổi sang `os.path.join` / `os.path.basename` (10 dòng, vẫn đúng trên Windows) trong
   `rename_files`, `Combining_classes`, `split_csv`.
2. **`Utility/Training.py` import `seaborn` mà không dùng.** Env `xmfgnn` có numpy 2.2.6 nhưng
   matplotlib 3.7.5 (build cho numpy 1.x) → `import seaborn` ném
   `ImportError: numpy.core.multiarray failed to import`, kéo theo `train.py` chết. Đã xoá dòng
   import đó. Nếu sau này cần vẽ hình thì phải nâng matplotlib (>=3.8) hoặc hạ numpy < 2.
3. **Số flow feature không khớp giữa code hiện tại và dataset đã tiền xử lý.**
   `additional_features` bản hiện tại sinh 14 cột `Rolling_*` → graph có **69** flow feature;
   còn `data/CIC_IoT2023_Processed_Data/df_class_8_train_clean.csv` (tải từ Drive của tác giả)
   có 27 cột rolling kiểu `*_Destination` / `*_SourceDestination` → **82** flow feature
   (khớp `logs/train.log`: `flow.x=(1, 82)`). Nghĩa là checkpoint trong `checkpoints/` **không**
   dùng lại được cho graph sinh từ pcap bằng code hiện tại, và ngược lại. Cần chọn: hoặc sửa
   `additional_features` sinh đủ bộ cột như dataset gốc, hoặc build lại toàn bộ graph + train lại
   từ pcap bằng code hiện tại.

---

## 6. File trong bộ này

| File | Việc |
|---|---|
| `Debug/trace_pipeline.py` | driver 6 stage, in ra từng bước + input/output từng hàm |
| `Debug/tracer.py` | tracer `sys.setprofile` (log thứ tự gọi hàm) + `Tee` (console → file log) |
| `.vscode/launch.json` | 11 config debug (F5), có ở cả `XG_NID/.vscode/` và `GNN4ID/.vscode/` |
| `.vscode/tasks.json` | 4 task chạy trace không cần debugger |
| `.vscode/settings.json` | interpreter `xmfgnn`, `PYTHONPATH`, `justMyCode: true`, cwd cho notebook |
| `.vscode/extensions.json` | danh sách extension gợi ý |

Không có file nào ở đây ghi vào `data/CIC_IoT2023_Processed_Data/` hay `checkpoints/`.
