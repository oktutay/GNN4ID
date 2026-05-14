# Báo cáo đối chiếu code GNN4ID với bài báo XG-NID

**Mục đích:** Tài liệu này tổng hợp toàn bộ kết quả đối chiếu code repository [GNN4ID](.) với bài báo gốc [XG-NID: Dual-Modality Network Intrusion Detection using a Heterogeneous Graph Neural Network and Large Language Model](XG_NID_Dual_Modality_Network_Intrusion_Detection_using_a_Heterogeneous.pdf) (Farrukh et al., Expert Systems with Applications 2025, vol 287:128089). Tài liệu liệt kê các điểm **không khớp**, **các bản vá đã áp dụng**, và **các bước cần làm** khi clone repo về để chạy lại.

**Ngày phân tích:** 2026-05-08
**Phạm vi vá:** Toàn bộ Priority A (ảnh hưởng kết quả F1) + Priority B (bổ sung explainer còn thiếu) + Priority C (cosmetic).

---

## Mục lục

1. [Tóm tắt nhanh](#1-tóm-tắt-nhanh)
2. [Cấu trúc 6 components của paper](#2-cấu-trúc-6-components-của-paper)
3. [Chi tiết từng vấn đề và cách vá](#3-chi-tiết-từng-vấn-đề-và-cách-vá)
4. [Danh sách file đã sửa / tạo mới](#4-danh-sách-file-đã-sửa--tạo-mới)
5. [Hướng dẫn chạy lại sau khi clone](#5-hướng-dẫn-chạy-lại-sau-khi-clone)
6. [Phụ lục: bảng tham chiếu paper ↔ code](#6-phụ-lục-bảng-tham-chiếu-paper--code)

---

## 1. Tóm tắt nhanh

Trước khi vá, GNN4ID **chỉ chứa 3/6 components** mà paper mô tả, và 3 components đó cũng **lệch về kiến trúc model** so với paper. Sau khi vá:

| # | Component | Trạng thái trước vá | Trạng thái sau vá |
|---|---|---|---|
| 1 | Flow & Feature Generator (Sec 3.1.1) | ✅ Khớp về cơ chế, lệch số features | ✅ Khớp paper |
| 2 | Explainable Feature Extractor (Sec 3.1.2 / Algorithm 1 / Table 1) | ⚠️ Có bug overwrite + thiếu/lệch tên features | ✅ Đủ 16 features Table 1, đặt tên khớp paper |
| 3 | Graph Generator (Sec 3.1.3 / Algorithm 2) | ✅ Khớp tốt | ✅ Khớp paper, có thêm safety drop |
| 4 | GNN Model (Sec 3.1.4 / eq 5–9) | ❌ Default dùng SAGEConv, không phải GATConv | ✅ GATConv mặc định, MLP head có ReLU |
| 5 | Integrated Gradient Explainer (Sec 3.1.5 / eq 10) | ❌ Hoàn toàn thiếu | ✅ Đã viết module mới |
| 6 | Generative Explainer / Llama-3 (Sec 3.1.6 / Algorithm 3) | ❌ Hoàn toàn thiếu | ✅ Đã viết module mới |

---

## 2. Cấu trúc 6 components của paper

Paper XG-NID đề xuất framework gồm 6 module nối tiếp nhau:

```
Raw PCAP
  │
  ▼
[1] Flow & Feature Generator    ─ NFStream, max 20 packets/flow, idle 120s
  │                                76 flow features + 14 packet features
  │                                payload mỗi packet → vector 1500 bytes
  ▼
[2] Explainable Feature Extractor ─ 16 rolling-window features (Table 1)
  │
  ▼
[3] Graph Generator              ─ HeteroGraph: flow node + packet node
  │                                contain edge (4 attrs) + link edge (1 attr - t_delta)
  ▼
[4] HGNN Model                   ─ 2 GATConv → BN → LeakyReLU
  │                                GlobalMeanPool → MLP 3 lớp → LogSoftmax
  ▼   (predicted class)
[5] Integrated Gradient Explainer ─ Tính tầm quan trọng của từng feature (eq 10)
  │
  ▼
[6] Generative Explainer (LLM)   ─ Llama-3-8B + zero-shot prompt
                                   (Algorithm 3: P_init + P_part2 + P_align)
                                   với payload-specific attacks: thêm query payload
```

---

## 3. Chi tiết từng vấn đề và cách vá

### 3.1. Component 1 — Flow & Feature Generator

#### Vấn đề
- **Số flow features lệch.** Paper Section 3.1.1 ghi *"computes 76 flow-level features and 14 packet-level features"*; nhưng [README.md](README.md) trước đây ghi `82 Features (Statistical Flow Features)` và NFStream sinh ra ~82 cột (bao gồm các meta như `id`, `vlan_id`, `tunnel_id`, `src_oui`, `dst_oui`, `ip_version`).
- Lý do: paper đã đếm số features **sau khi drop** một số cột meta (drop list xem trong [Utility/Functions.py](Utility/Functions.py) line 119–124).

#### Đã làm gì
- Cập nhật [README.md](README.md) ghi rõ `~76 Statistical Flow Features` kèm chú thích về việc drop các cột meta.
- Mức độ ảnh hưởng kết quả: **không đáng kể** — chỉ ảnh hưởng dim đầu vào của flow node, không phá kiến trúc.

---

### 3.2. Component 2 — Explainable Feature Extractor

#### Vấn đề chính
File [Utility/Additional_Features.py](Utility/Additional_Features.py) bản gốc có **3 lỗi**:

**Lỗi 1 — Overwrite biến `is_vulnerable_port`:** Cùng 1 cột `is_vulnerable_port` bị gán lại 4 lần liên tiếp với 4 ý nghĩa khác nhau:
```python
data['is_vulnerable_port'] = data['dst_port'].isin(http_ports)        # http
...
data['is_vulnerable_port'] = data['dst_port'].isin(dns_ports)         # dns dst → ghi đè http!
data['is_vulnerable_port'] = data['src_port'].isin(dns_ports)         # dns src → ghi đè
data['is_vulnerable_port'] = data['dst_port'].isin(vulnerable_ports)  # vuln → ghi đè
```
Hệ quả: tên cột rolling tính ra ổn (vì rolling tính tại thời điểm cột chưa bị ghi đè), **nhưng** giá trị cuối cùng của `is_vulnerable_port` (= `dst_port ∈ vulnerable_ports`) sẽ leak vào flow_node features dưới dạng 1 cột bool.

**Lỗi 2 — Thiếu/lệch tên features so với Table 1:** Paper liệt kê 16 features có tên cụ thể (`Rolling_UDP_Sum`, `Rolling_http_port`, `Unique_Ports_In_SourceDestination`, ...), code dùng tên khác (`Rolling_UDP_Requests_Destination`, ...). Khó so sánh với paper.

**Lỗi 3 — Thiếu biến `Rolling_vulnerable_port` per-Destination:** Code chỉ có `_SourceDestination` variant.

#### Đã làm gì
**Rewrite hoàn toàn** file [Utility/Additional_Features.py](Utility/Additional_Features.py):
- Tách helper booleans thành các tên cột riêng biệt: `_is_udp`, `_is_tcp`, `_is_icmp`, `_is_http_port`, `_is_dns_dst_port`, `_is_dns_src_port`, `_is_vuln_port`.
- Tạo đủ 16 features với **tên đúng theo Table 1**: `Rolling_UDP_Sum`, `Rolling_TCP_Sum`, `Rolling_ACK_Sum`, `Rolling_FIN_Sum`, `Rolling_RST_Sum`, `Rolling_psh_Sum`, `Rolling_SYN_Sum`, `Rolling_ICMP_Sum`, `Rolling_http_port`, `Rolling_Average_Duration`, `Rolling_DNS_Sum`, `Rolling_vulnerable_port`, `Rolling_packets_Sum`, `Rolling_bipackets_Sum`, `Unique_Ports_In_SourceDestination`, `packet_size_variation`.
- Tự drop tất cả helper booleans trước khi `to_csv()` để không leak vào flow node.

#### Bổ sung phòng vệ ở Functions.py
[Utility/Functions.py](Utility/Functions.py) `process()` thêm safety drop:
```python
for tmp_col in ('is_vulnerable_port', 'is_http_port', 'is_dns_dst_port',
                'is_dns_src_port', 'is_vuln_port', 'is_udp_request',
                'is_tcp_request', 'is_icmp_request'):
    if tmp_col in self.data.columns:
        self.data.drop(tmp_col, axis=1, inplace=True)
```
Đảm bảo dù dùng phiên bản cũ của `Additional_Features.py` cũng không leak.

---

### 3.3. Component 3 — Graph Generator

#### Vấn đề
- Paper Sec 3.1.3 mô tả `link` edges hình thành **DAG có hướng**, nhưng code [Utility/Functions.py](Utility/Functions.py) line 193 gọi `data = T.ToUndirected()(data)`, sinh thêm edge type ngược (`('packet','rev_link','packet')`, `('packet','rev_contain','flow')`).
- Đây không hẳn là bug — thực tế **bắt buộc** để message-passing chạy 2 chiều giữa flow node và packet node trong PyG. Nếu giữ nguyên DAG thì gradient không truyền được từ packet về flow.

#### Đã làm gì
- **Giữ nguyên** `T.ToUndirected()` vì cần thiết cho hoạt động của HGNN.
- Thêm safety drop helper cols trong `process()` (xem mục 3.2).

---

### 3.4. Component 4 — GNN Model ❌ SAI LỆCH NGHIÊM TRỌNG

#### Vấn đề chính
Paper Section 3.1.4 nêu rõ:
> *"The model is built upon the Graph Attention Convolution (GATConv) (Velickovic et al., 2017) approach"*

Nhưng [Utility/Model.py](Utility/Model.py) bản gốc định nghĩa **2 model**:
- `HeteroGNN` (default) → dùng `SAGEConv` ❌
- `HeteroGNN_Edge` → dùng `GATConv` ✅

Trong [GNN4ID_Model.ipynb](GNN4ID_Model.ipynb), cell khởi tạo model uncomment dòng `HeteroGNN(...)` (SAGE) và comment-out dòng `HeteroGNN_Edge(...)` (GAT). **User chạy mặc định không phải kiến trúc của paper.**

#### Vấn đề phụ
1. **MLP head thiếu activation:** Paper eq (9):
   ```
   Out = LogSoftmax(W2 · ReLU(W1 · ReLU(W0 · h_graph)))
   ```
   Code có 3 `nn.Linear` chồng lên nhau **không có ReLU ở giữa** → tương đương 1 linear layer.

2. **Hidden size hardcoded:** `SAGEConv((-1,-1), 64)` hardcoded `64` thay vì dùng `args['hidden_size']`. Crash nếu user đổi hidden_size.

3. **Activation function:** Paper eq 7 dùng ReLU; code dùng LeakyReLU (paper eq 6 cho phép cả 2 nên không sai, giữ nguyên LeakyReLU).

#### Đã làm gì
**Rewrite [Utility/Model.py](Utility/Model.py):**
- `HeteroGNN` (default) **giờ dùng `GATConv`** với `edge_dim=-1` (cho phép edge attributes — đúng paper eq 6 với $W_e$ riêng cho edge).
- MLP head có `F.relu()` giữa các Linear layer (đúng eq 9).
- `hidden_size` lấy từ `args['hidden_size']`, không hardcode.
- Input MLP head dim = `2 * hidden_size` (concat embedding flow + packet sau global_mean_pool).
- Tham số `use_edge_attr` (default True) cho phép tắt edge attr nếu cần ablation.
- `HeteroGNN_Edge` giờ là **alias** trỏ về `HeteroGNN` để backward-compatible.
- `HeteroGNN_SAGE` (mới) là class kế thừa kiến trúc cũ — dành cho **ablation**, không phải kiến trúc paper.

#### Cập nhật notebook
Cell khởi tạo model trong [GNN4ID_Model.ipynb](GNN4ID_Model.ipynb) đã được sửa để default dùng GATConv:
```python
model = HeteroGNN(data_model, args, aggr="mean").to(args['device'])
# Ablation alternatives:
# model = HeteroGNN(data_model, args, aggr="mean", use_edge_attr=False)
# from Utility.Model import HeteroGNN_SAGE
# model = HeteroGNN_SAGE(data_model, args, aggr="mean")
```

#### Cập nhật Training.py
[Utility/Training.py](Utility/Training.py) `train()/test()/test_cm()` giờ mặc định gọi model với edge_attr_dict (đúng paper):
```python
pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
```
Thêm:
- Alias `train_with_edge_Att`/`test_edge`/`test_cm_with_edge_att` (backward compat).
- `train_no_edge_attr`/`test_no_edge_attr`/`test_cm_no_edge_attr` cho ablation SAGE.
- Bổ sung precision/recall/F1 macro vào `calculate_metrics()`.

---

### 3.5. Component 5 — Integrated Gradient Explainer ❌ THIẾU HOÀN TOÀN

#### Vấn đề
Paper Section 3.1.5 mô tả dùng Integrated Gradients (Sundararajan et al. 2017) làm post-hoc local explainer:
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha$$

**Repository không có file/cell nào** implement công thức này.

#### Đã làm gì
**Tạo mới [Utility/IG_Explainer.py](Utility/IG_Explainer.py):**
- Class `IntegratedGradientExplainer`:
  - Nhận model HGNN (đã train) + device + n_steps (mặc định 50).
  - Method `.explain(data, target_class=None, flow_baseline=None, packet_baseline=None)` → trả về `IGAttribution` dataclass.
  - Tính IG riêng cho flow node features và packet node features (paper Sec 3.1.5: "feature-based local explanations").
  - Baseline mặc định = zero (đúng paper: *"If a baseline input is not provided, zero is used as the default value"*).
  - Riemann sum approximation với n_steps điểm trên đoạn [baseline, input].
- Helper `top_flow_features(attr, feature_names, top_n=5)`: trả về list `[(name, attribution, actual_value), ...]` sắp giảm dần theo `|IG|`.
- Helper `top_payload_bytes(attr, top_n=32)`: aggregate attribution các packet, sort, lấy top N bytes (chuẩn bị cho Algorithm 3 line 19–24).

Không phụ thuộc Captum (tự implement bằng `torch.autograd.grad`) — gọn hơn, dễ debug.

---

### 3.6. Component 6 — Generative Explainer (Llama-3-8B) ❌ THIẾU HOÀN TOÀN

#### Vấn đề
Paper Section 3.1.6 + Algorithm 3 mô tả:
- Llama-3-8B với zero-shot prompt
- 3 segment: $P_{\text{init}}$ + $P_{\text{part2}}$ + $P_{\text{align}}$ (cho flow query $Q_{\text{flow}}$)
- Với payload-specific attacks (WebBased, BruteForce): thêm query thứ 2 với $P_{\text{payloadPrefix}}$ + ASCII payload + $P_{\text{align}}$
- Output cuối: $G_{\text{exp}} = R_{\text{flow}} + R_{\text{payload}}$

**Repository hoàn toàn không có** — dù `requirement.txt` đã liệt kê `transformers`, `accelerate`, `bitsandbytes` chứng tỏ author định cài nhưng chưa commit.

#### Đã làm gì
**Tạo mới [Utility/LLM_Explainer.py](Utility/LLM_Explainer.py):**
- Các template **lấy verbatim từ paper Section 3.1.6**:
  - `P_INIT_TEMPLATE = "The predicted class from GNN is {predicted_class}."`
  - `P_PART2_HEADER = "The top features contributing to this prediction are:"`
  - `P_ALIGN = "Don't expect any values on your own. Explain the predicted outcome..."`
  - `P_PAYLOAD_PREFIX = "Analyze whether this payload of network flow is malicious or not. Give reason concisely."`
- Class `LLMExplainer`:
  - `.explain(attribution, feature_names, pipeline=None)` → trả về `GenerativeExplanation` dataclass.
  - Pipeline=None → dry run, chỉ trả về prompts (không cần GPU).
  - Pipeline có instance → gọi LLM tạo response.
  - Tự phát hiện payload-specific class (mặc định `WebBased`, `BruteForce`) → tự generate payload query.
- Helper `load_llama3_pipeline()`:
  - Lazy import `transformers` (không trigger download khi import module).
  - Hỗ trợ 4-bit quantization qua `bitsandbytes` để chạy được trên GPU 12 GB VRAM.

#### Notebook demo
Append 5 cells mới vào cuối [GNN4ID_Model.ipynb](GNN4ID_Model.ipynb):
1. Markdown giới thiệu phần Explainability.
2. Cell load test sample + reconstruct `FLOW_FEATURE_NAMES` từ CSV header.
3. Cell chạy IG explainer + in top 5 flow features.
4. Cell build LLM prompt (dry run, không cần GPU).
5. Cell (optional, comment-out) load Llama-3-8B và chạy thật.

---

### 3.7. Vấn đề khác

#### 3.7.1. Bug filter Benign MAC trong split_csv ⚠️
[Utility/Functions.py](Utility/Functions.py) line 531 (bản gốc):
```python
if name_check == 'Benign':
    df[(df['src_mac']!='dc:a6:32:dc:27:d5') & ...]   # ← thiếu "df ="
```
Câu lệnh này **không có gì cả** — Pandas chỉ tạo dataframe filtered tạm rồi vứt đi. Nghĩa là filter Benign theo MAC **bị bỏ qua hoàn toàn**, dẫn đến benign class bị nhiễm các flow có nguồn/đích là attacker.

**Đã sửa:**
```python
attacker_macs = {'dc:a6:32:dc:27:d5', ...}  # paper Table 3
src_is_attacker = df['src_mac'].isin(attacker_macs)
dst_is_attacker = df['dst_mac'].isin(attacker_macs)

if name_check == 'Benign':
    df = df[~src_is_attacker & ~dst_is_attacker]
else:
    df = df[src_is_attacker | dst_is_attacker]
```

#### 3.7.2. Class name lệch (`Dos` vs `DoS`, `DDos` vs `DDoS`)
- Paper Tables 2 / 4 dùng `DoS` / `DDoS`.
- Code (`label_dict`, `Combining_classes`, `rename_files` mapping) dùng `Dos` / `DDos`.

**Đã sửa:** `Combining_classes` chấp nhận **cả 2 dạng** thông qua alias merge:
```python
aliases = {'DoS': 'Dos', 'DDoS': 'DDos'}
for paper_form, code_form in aliases.items():
    if code_form in label_dict and paper_form not in label_dict:
        label_dict[paper_form] = label_dict[code_form]
```
Không break notebook hiện tại (vẫn dùng `Dos/DDos`), nhưng cũng cho phép user pass dạng paper.

#### 3.7.3. Test set size cho class hiếm
Paper Table 4 quy định:
- WebBased test = 1,090 (≈20% của 5,449 flow sau filter).
- Bruteforce test = 467 (≈20% của 2,336 flow sau filter).

Code dùng threshold `if df.shape[0] > 35000: test = 4000; else: test = 0.2 * df`. Logic này khớp tinh thần paper với class hiếm nhưng **threshold 35000** là magic number.

**Trạng thái:** Giữ nguyên — sẽ verify khi chạy thật nếu số test sample không đúng paper.

---

## 4. Danh sách file đã sửa / tạo mới

| File | Hành động | Mục đích |
|---|---|---|
| [Utility/Model.py](Utility/Model.py) | **Rewrite** | Default `HeteroGNN` dùng GATConv với edge_dim=-1; MLP head có ReLU; hidden_size động; thêm `HeteroGNN_SAGE` cho ablation. |
| [Utility/Training.py](Utility/Training.py) | **Rewrite** | `train/test/test_cm` route qua edge-attr path; bổ sung F1/Precision/Recall macro; thêm `*_no_edge_attr` cho ablation. |
| [Utility/Functions.py](Utility/Functions.py) | **3 patches** | (1) Fix bug filter Benign MAC (line 531); (2) Drop helper booleans trong `process()`; (3) `Combining_classes` chấp nhận DoS/DDoS lẫn Dos/DDos. |
| [Utility/Additional_Features.py](Utility/Additional_Features.py) | **Rewrite** | Sạch overwrite bug; tạo 16 features Table 1 với tên đúng paper; tự drop helpers. |
| [Utility/IG_Explainer.py](Utility/IG_Explainer.py) | **Tạo mới** | `IntegratedGradientExplainer` cho HeteroData (paper eq 10); helper top_flow_features & top_payload_bytes. |
| [Utility/LLM_Explainer.py](Utility/LLM_Explainer.py) | **Tạo mới** | `LLMExplainer` build prompt theo Algorithm 3; `load_llama3_pipeline()` lazy loader 4-bit. |
| [GNN4ID_Model.ipynb](GNN4ID_Model.ipynb) | **Patch + 5 cells** | Đổi cell model về GATConv default; thêm section Explainability. |
| [README.md](README.md) | Sửa | Đổi "82 Features" → mô tả đúng số features sau drop. |
| [requirement.txt](requirement.txt) | Sửa | Thêm `captum==0.7.0`, `matplotlib==3.7.5`. |
| [BAOCAO_DOI_CHIEU_XG_NID.md](BAOCAO_DOI_CHIEU_XG_NID.md) | **Tạo mới** | File báo cáo này. |

---

## 5. Hướng dẫn chạy lại sau khi clone

### 5.1. Yêu cầu môi trường
- **OS:** Linux/macOS hoặc Windows (NFStream khuyến nghị Linux/WSL trên Windows).
- **Python:** 3.8 – 3.10 (do `torch==2.3.1+cu118`).
- **GPU:** CUDA 11.8, ≥12 GB VRAM (đủ cho HGNN + Llama-3-8B 4-bit).
- **Disk:** ~100 GB free (CIC-IoT2023 raw pcap ~70 GB, processed graphs ~10 GB).

### 5.2. Cài đặt

```bash
# Clone repo
git clone <url-repo> GNN4ID
cd GNN4ID

# Tạo môi trường
conda create -n gnn4id python=3.10 -y
conda activate gnn4id

# Cài đặt dependencies
pip install -r requirement.txt
```

**Lưu ý NFStream:** Trên Windows, `nfstream` cần `libpcap` và build tools. Khuyến nghị dùng WSL2 hoặc Docker. Nếu chỉ chạy training/eval (không chạy lại từ pcap), có thể bỏ qua.

### 5.3. Tải dataset

**Cách 1 — Raw pcap (full pipeline):**
- Tải [CIC-IoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) (~70 GB, định dạng `.tar.gz`).
- Đặt vào thư mục, ví dụ `F:/CIC IoT Dataset 2023/`.
- Cập nhật `Directory` và `Out_Directory` trong [GNN4ID.ipynb](GNN4ID.ipynb).

**Cách 2 — CSV đã preprocessed (skip pipeline):**
- Tải từ [Google Drive link trong README](README.md) (~10 GB).
- Đặt vào `F:/CIC_IOT/Extracted_Flow_Features/`.
- Skip ngay đến [GNN4ID_Model.ipynb](GNN4ID_Model.ipynb).

### 5.4. Chạy pipeline đầy đủ (nếu dùng Cách 1)

**Bước 1 — Trích xuất features từ pcap:**
```bash
jupyter notebook GNN4ID.ipynb
```
- Chạy lần lượt các cell trong notebook.
- Cell `Feature_extractor_flow_packet_combined.py` sẽ chạy trên từng file pcap.
- Cell `additional_features()` sẽ thêm 16 rolling-window features vào CSV.
- **Thời gian:** vài giờ tới 1 ngày tùy GPU/CPU.

**Bước 2 — Preprocess (filter MAC + balance class):**
```bash
jupyter notebook Data_preprocessing_CIC-IoT2023.ipynb
```
- Filter benign theo attacker MAC (paper Table 3).
- Cap train = 20,000 samples/class, test = 4,000 (1,090 cho WebBased, 467 cho Bruteforce).
- Output: `df_class_8_train.csv` và `df_class_8_test.csv`.

**Bước 3 — Tạo graph objects:**
- Quay lại [GNN4ID.ipynb](GNN4ID.ipynb), chạy phần *"Transformation into Graph Data Objects"*.
- Tạo các file `.pt` trong `processed/`.

### 5.5. Train & Eval

```bash
jupyter notebook GNN4ID_Model.ipynb
```

Cell quan trọng đã được sửa default về **GATConv** (paper-faithful):
```python
model = HeteroGNN(data_model, args, aggr="mean").to(args['device'])
```

Args mặc định:
```python
args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 64,
    'epochs': 30,
    'weight_decay': 1e-5,
    'lr': 0.01,
    'attn_size': 32,
    'eps': 1.0,
}
```

Chạy `train(train_loader, model, args, args["device"])` → 30 epochs.

**Thời gian:** khoảng 2–4 giờ trên 1 GPU NVIDIA RTX 3090 / V100.

**Kết quả mong đợi (paper Table 5/6 & Fig 3):**
- F1 macro ≈ 0.97
- Precision ≈ 0.95, Recall ≈ 0.99
- Payload-specific F1 ≈ 0.94
- Flow-specific F1 ≈ 0.98

### 5.6. Chạy explainer (mới)

Sau khi model đã train xong, chạy 5 cell cuối trong [GNN4ID_Model.ipynb](GNN4ID_Model.ipynb):

**Cell 1 — Load test sample + feature names:**
```python
from Utility.IG_Explainer import IntegratedGradientExplainer, top_flow_features
from Utility.LLM_Explainer import LLMExplainer, load_llama3_pipeline

test_sample = data_Hetero[0]
# Tự động build FLOW_FEATURE_NAMES từ CSV header
```

**Cell 2 — Tính Integrated Gradient:**
```python
ig = IntegratedGradientExplainer(model, device=args['device'], n_steps=50)
attr = ig.explain(test_sample)
print('Top features:')
for name, a, v in top_flow_features(attr, FLOW_FEATURE_NAMES, top_n=5):
    print(f'  {name:40s}  attr={a:+.4f}  value={v:.4f}')
```

**Cell 3 — Build prompt (dry run):**
```python
explainer = LLMExplainer()
explanation = explainer.explain(attr, FLOW_FEATURE_NAMES, pipeline=None)
print(explanation.flow_prompt)
```

**Cell 4 — Gọi Llama-3-8B thật (optional):**
```python
# Cần huggingface-cli login + access Meta-Llama-3-8B-Instruct
pipe = load_llama3_pipeline()
explanation = explainer.explain(attr, FLOW_FEATURE_NAMES, pipeline=pipe)
print(explanation.text)
```

### 5.7. Xác minh kết quả khớp paper

Sau khi train xong, xem confusion matrix và classification report. Mong đợi:

| Class | Paper F1 | Code (kỳ vọng) |
|---|---|---|
| Benign | ~0.99 | ? |
| WebBased | ~0.95 | ? |
| Spoofing | ~0.95 | ? |
| Recon | ~0.97 | ? |
| Mirai | ~0.99 | ? |
| DoS | ~0.96 | ? |
| DDoS | ~0.99 | ? |
| BruteForce | ~0.93 | ? |
| **Macro F1** | **0.97** | ? |

**Nếu kết quả lệch nhiều với paper, các nghi ngờ tiếp:**
1. `window_size=350` trong `Additional_Features.py` — paper không công bố, mặc định 350 có thể chưa tối ưu.
2. Batch size = 64 — paper không công bố.
3. Thứ tự features trong flow node có thể khác — verify bằng cách dump tên features.
4. Random seed — paper không cố định seed.
5. Model có thể cần thêm dropout hoặc weight decay.

---

## 6. Phụ lục: bảng tham chiếu paper ↔ code

### 6.1. Paper Section 3.1.1 — Flow & Feature Generator

| Paper | Code |
|---|---|
| NFStream-based extractor | [`Utility/Feature_extractor_flow_packet_combined.py`](Utility/Feature_extractor_flow_packet_combined.py) |
| `udps=My_Custom(limit=20)` (max 20 packets/flow) | line 101 |
| `idle_timeout=120` (120s idle) | line 101 |
| 76 flow features | NFStream output - drop list trong `process()` |
| 14 packet features | `udps.payload_data, delta_time, packet_direction, ip_size, transport_size, payload_size, syn, cwr, ece, urg, ack, psh, rst, fin` |
| Payload → 1500 bytes vector | [`Utility/Functions.py:_get_packet_node_features`](Utility/Functions.py) line 246–267 |

### 6.2. Paper Section 3.1.2 / Algorithm 1 / Table 1 — Explainable Features

| Table 1 | Code (Additional_Features.py mới) |
|---|---|
| Rolling_UDP_Sum | ✅ `Rolling_UDP_Sum` |
| Rolling_TCP_Sum | ✅ `Rolling_TCP_Sum` |
| Rolling_ACK_Sum | ✅ `Rolling_ACK_Sum` |
| Rolling_FIN_Sum | ✅ `Rolling_FIN_Sum` |
| Rolling_RST_Sum | ✅ `Rolling_RST_Sum` |
| Rolling_psh_Sum | ✅ `Rolling_psh_Sum` |
| Rolling_SYN_Sum | ✅ `Rolling_SYN_Sum` |
| Rolling_ICMP_Sum | ✅ `Rolling_ICMP_Sum` |
| Rolling_http_port | ✅ `Rolling_http_port` |
| Rolling_Average_Duration | ✅ `Rolling_Average_Duration` |
| Rolling_DNS_Sum | ✅ `Rolling_DNS_Sum` |
| Rolling_vulnerable_port | ✅ `Rolling_vulnerable_port` |
| Rolling_packets_Sum | ✅ `Rolling_packets_Sum` |
| Rolling_bipackets_Sum | ✅ `Rolling_bipackets_Sum` |
| Unique_Ports_In_SourceDestination | ✅ `Unique_Ports_In_SourceDestination` |
| (16th — `packet_size_variation`) | ✅ `packet_size_variation` (giữ từ phiên bản cũ) |

### 6.3. Paper Section 3.1.3 / Algorithm 2 — Graph Generator

| Paper | Code |
|---|---|
| Flow nodes $V_f$ | [`Functions.py:_get_flow_node_features`](Utility/Functions.py) |
| Packet nodes $V_p$ ($\mathbb{R}^{1500}$) | [`Functions.py:_get_packet_node_features`](Utility/Functions.py) |
| Contain edges $E_c$ (flow→packet) | [`Functions.py:_get_contain_edge_index`](Utility/Functions.py) |
| Contain edge feats (4 attrs: direction, IP size, transport size, payload size) | [`Functions.py:_get_contain_edge_features`](Utility/Functions.py) |
| Link edges $E_l$ (packet→packet) | [`Functions.py:_get_link_edge_index`](Utility/Functions.py) |
| Link edge feats (1 attr: $t_\delta$) | [`Functions.py:_get_link_edge_features`](Utility/Functions.py) |

### 6.4. Paper Section 3.1.4 — HGNN Model (eq 5–9)

| Paper | Code (Model.py mới) |
|---|---|
| eq 5: $h^{(1)}_i = \text{ReLU}(\text{GATConv}(h^{(0)}_i, A, E))$ | `self.convs1` + `self.relus1` |
| eq 7: $h^{(2)}_i = \text{ReLU}(\text{BN}(\text{GATConv}(h^{(1)}_i, A, E)))$ | `self.convs2` + `self.bns2` + `self.relus2` |
| eq 8: $\mathbf{h}_{\text{graph}} = \text{GlobalMeanPool}(h^{(2)}_i)$ | `pyg_nn.global_mean_pool` |
| eq 9: $\text{Out} = \text{LogSoftmax}(W_2 \cdot \text{ReLU}(W_1 \cdot \text{ReLU}(W_0 \cdot \mathbf{h})))$ | `F.relu(self.graph_prediction)` x2 + `F.log_softmax` |

### 6.5. Paper Section 3.1.5 — Integrated Gradient (eq 10)

| Paper | Code (IG_Explainer.py) |
|---|---|
| $\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \partial F/\partial x_i \, d\alpha$ | `IntegratedGradientExplainer.explain()` (Riemann sum n_steps điểm) |
| Baseline mặc định = 0 | `flow_baseline = torch.zeros_like(flow_x)` |
| Apply riêng cho flow + packet features | 2 dòng `flow_grad_sum`, `packet_grad_sum` |

### 6.6. Paper Section 3.1.6 / Algorithm 3 — Generative Explainer

| Paper | Code (LLM_Explainer.py) |
|---|---|
| $P_{\text{init}}$ = "The predicted class from GNN is {C}" | `P_INIT_TEMPLATE` |
| $P_{\text{part2}}$ header | `P_PART2_HEADER` |
| $P_{\text{align}}$ | `P_ALIGN` |
| $P_{\text{payloadPrefix}}$ | `P_PAYLOAD_PREFIX` |
| Algorithm 3 line 19–24 (normalize, average, sort, hex→ASCII) | `top_payload_bytes()` |
| Llama-3 model | `load_llama3_pipeline("meta-llama/Meta-Llama-3-8B-Instruct")` |
| $G_{\text{exp}} = R_{\text{flow}} + R_{\text{payload}}$ | `GenerativeExplanation.text` property |

### 6.7. Paper Section 3.2.1 / Tables 2–4 — Dataset Preprocessing

| Paper | Code |
|---|---|
| Filter attacker MAC (Table 3, 9 MACs) | [`Functions.py:split_csv`](Utility/Functions.py) đã sửa |
| Train: 20,000 samples/class × 8 classes | [`Functions.py:Combining_classes`](Utility/Functions.py) |
| Test: 4,000 cho class lớn, 1,090 (WebBased) / 467 (Bruteforce) | `split_csv` với threshold 35000 |
| Over/under-sampling cho minority class | `duplicate_rows` + `random_pick_rows` |

---

## Liên hệ & Ghi chú

- **Paper gốc:** [arXiv:2408.16021](https://arxiv.org/abs/2408.16021)
- **Repo gốc:** https://github.com/Yasir-ali-farrukh/GNN4ID
- **Citation paper:** Farrukh et al., *Expert Systems with Applications* 2025, vol 287:128089.

**Lưu ý quan trọng:**
- Tất cả patches đã được verify bằng `py_compile` — không có lỗi syntax.
- Notebook đã được verify JSON hợp lệ.
- **Chưa chạy thực tế** với dataset CIC-IoT2023 (do môi trường máy phát triển không có GPU/dataset). User cần tự verify F1 sau khi chạy.
- Nếu F1 < 95%, debug theo thứ tự: (1) hyperparameters (window_size, batch size), (2) feature engineering (drop list, one-hot), (3) seed/random.
