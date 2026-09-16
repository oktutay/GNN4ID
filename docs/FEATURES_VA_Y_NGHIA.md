# Features và ý nghĩa trong GNN4ID

Tài liệu này tóm tắt các nhóm feature mà repo GNN4ID dùng để tạo graph cho XG-NID.
Nguồn chính của phần mô tả là:

- [Utility/Functions.py](Utility/Functions.py)
- [Utility/Additional_Features.py](Utility/Additional_Features.py)
- [Utility/Explainer.py](Utility/Explainer.py)

## 1. Tổng quan schema

- **Flow node**: 82 feature (46 thống kê NFStream + `packet_size_variation` + 28 rolling + 7 one-hot; danh sách chuẩn: `Utility/Schema.py::FLOW_FEATURE_NAMES_82`)
- **Packet node**: 1500 byte payload, cộng thêm 8 cờ TCP nếu bật `include_packetflag`
- **Contain edge**: 4 feature
- **Link edge**: 1 feature

## 2. Flow features

### 2.1 Nhóm thống kê luồng cơ bản

Các feature này mô tả hành vi lưu lượng ở mức flow:

| Feature group | Ý nghĩa |
|---|---|
| `src2dst_duration_ms`, `dst2src_duration_ms` | Thời lượng trao đổi theo từng chiều |
| `src2dst_packets`, `dst2src_packets` | Số packet theo từng chiều |
| `src2dst_bytes`, `bidirectional_bytes` | Số byte truyền đi / tổng byte |
| `bidirectional_min_ps`, `mean_ps`, `stddev_ps`, `max_ps` | Thống kê kích thước packet hai chiều |
| `src2dst_min_ps`, `mean_ps`, `stddev_ps`, `max_ps` | Thống kê kích thước packet chiều src→dst |
| `dst2src_min_ps`, `mean_ps`, `stddev_ps`, `max_ps` | Thống kê kích thước packet chiều dst→src |
| `bidirectional_min_piat_ms`, `mean_piat_ms`, `stddev_piat_ms`, `max_piat_ms` | Thống kê khoảng cách thời gian giữa packet hai chiều |
| `src2dst_min_piat_ms`, `mean_piat_ms`, `stddev_piat_ms`, `max_piat_ms` | PIAT theo chiều src→dst |
| `dst2src_min_piat_ms`, `mean_piat_ms`, `stddev_piat_ms`, `max_piat_ms` | PIAT theo chiều dst→src |

### 2.2 Nhóm cờ TCP theo chiều

Đếm số packet mang từng TCP flag:

| Feature group | Ý nghĩa |
|---|---|
| `src2dst_syn_packets`, `dst2src_syn_packets` | Số SYN packet |
| `src2dst_cwr_packets`, `dst2src_cwr_packets` | Số CWR packet |
| `src2dst_ece_packets`, `dst2src_ece_packets` | Số ECE packet |
| `src2dst_urg_packets`, `dst2src_urg_packets` | Số URG packet |
| `src2dst_ack_packets`, `dst2src_ack_packets` | Số ACK packet |
| `src2dst_psh_packets`, `dst2src_psh_packets` | Số PSH packet |
| `src2dst_rst_packets`, `dst2src_rst_packets` | Số RST packet |
| `src2dst_fin_packets`, `dst2src_fin_packets` | Số FIN packet |

### 2.3 Feature mở rộng theo rolling window (28 cột + `packet_size_variation`)

Nhóm này được tạo trong [Utility/Additional_Features.py](Utility/Additional_Features.py) — bản 16/09/2026
sinh **đúng 28 cột, đúng tên và thứ tự** như code tác giả (upstream 551d1f1) nên khớp CSV Drive/checkpoint.
Cửa sổ mặc định = **350 flow trước đó** (dòng CSV, không phải packet, không phải giây; paper chỉ nói
"rolling time window"), tính trên toàn file pcap sort theo `bidirectional_first_seen_ms`, `min_periods=1`,
reset theo từng file. Mỗi feature có 2 nhóm: `*_Destination` (theo `dst_ip`) và `*_SourceDestination`
(theo cặp `src_ip`-`dst_ip`).

| Cột (2 nhóm trừ khi ghi khác) | Nguồn | Đơn vị đếm (mặc định `count_mode='author'`) |
| --- | --- | --- |
| `Rolling_UDP_Requests_*` / `Rolling_TCP_Requests_*` / `Rolling_ICMP_Requests_*` | `protocol` == 17 / 6 / 1 | **flow** (1 mỗi dòng) |
| `Rolling_ACK_Packets_*`, `Rolling_FIN_Packets_*`, `Rolling_rst_Packets_*`, `Rolling_psh_Packets_*`, `Rolling_SYN_Packets_*` | `bidirectional_<flag>_packets` | **packet** |
| `Unique_Ports_In_SourceDestinationIP` (chỉ theo cặp) | số `dst_port` khác nhau trong cửa sổ (paper viết "source ports") | cổng |
| `Rolling_http_port_*` | `dst_port` ∈ {80, 443, 8080} | flow |
| `Rolling_Duration_Destination` / `Rolling_Duration_SourceDestination` | trung bình `bidirectional_duration_ms` | ms |
| `Rolling_DNS_request_*` / `Rolling_DNS_request_*2` | `dst_port` == 53 / `src_port` == 53 | flow |
| `Rolling_vulnerable_port` (chỉ theo cặp) | `dst_port` ∈ 13 cổng hay bị khai thác | flow |
| `Rolling_packets_destination` / `Rolling_bipackets_destination` (chỉ theo đích) | tổng `src2dst_packets` / `bidirectional_packets` | packet |
| `packet_size_variation` | std của 4 cột min/max packet size 2 chiều | byte |

`count_mode='packets'` nhân các cột đếm-flow với `bidirectional_packets` để mọi cột đều đếm packet như
Table 1 mô tả; `window_unit='time'` (vd `'60s'`) hoặc `'packets'` đổi đơn vị cửa sổ; `schema='table1'`
chỉ giữ 15 cột theo đích và đổi sang tên Table 1 (`Rolling_UDP_Sum`, ...). Với flood, cửa sổ đầy rất nhanh
nên các cột này gần như hằng số (= 350) — xem `CIC_IoT2023_EDA_v2_VI.md`.

### 2.4 Feature phân loại

| Feature | Ý nghĩa |
|---|---|
| `packet_size_variation` | Độ biến thiên kích thước packet giữa các thống kê min/max hai chiều |
| `Exp_0`, `Exp_-1` | One-hot của `expiration_id` |
| `proto_1`, `proto_2`, `proto_6`, `proto_17`, `proto_58` | One-hot của giao thức IP |

## 3. Packet node features

Mỗi packet được mã hóa bằng payload byte-wise:

| Feature | Ý nghĩa |
|---|---|
| `udps.payload_data` | Payload của packet, chuyển sang chuỗi byte hex rồi pad/cắt về 1500 byte |
| `udps.syn` ... `udps.fin` | Tùy chọn thêm 8 cờ TCP theo từng packet nếu bật `include_packetflag` |

### 3.1 `udps.payload_data` thực chất chứa gì?

Trong code của repo, giá trị này được tạo bằng:

- nếu `packet.payload_size > 0` thì lấy `packet.ip_packet[-packet.payload_size:]`
- sau đó chuyển phần byte đó sang chuỗi hex bằng `.hex()`
- nếu `packet.payload_size == 0` thì ghi chuỗi `"00"`

Điều đó có nghĩa là `udps.payload_data` là **phần payload thô của packet**, không phải toàn bộ packet.

Nói theo tầng mạng thì:

- **Không chứa IP header**
- **Không chứa TCP/UDP/ICMP header**
- **Không chứa TCP flags như SYN/ACK/FIN/RST/... trong chính chuỗi payload**
- Các flags được lưu tách riêng ở `udps.syn`, `udps.ack`, `udps.fin`, ... nếu bật `include_packetflag`

Về mặt ngữ nghĩa, `udps.payload_data` là phần dữ liệu nằm “sau” phần header của tầng vận chuyển, tức là dữ liệu ứng dụng hoặc dữ liệu thô mà packet mang theo. Ví dụ:

- với HTTP, payload có thể chứa request/response text như `GET / ...`, header HTTP, body, JSON, v.v.
- với DNS, payload có thể chứa truy vấn/response DNS
- với TLS, payload có thể là dữ liệu mã hóa nên nhìn như chuỗi byte ngẫu nhiên
- với gói rỗng hoặc gói chỉ mang control signal, payload có thể bằng 0 byte và repo sẽ ghi `"00"`

Sau đó, khi tạo graph, repo chuyển chuỗi hex này về bytes, pad/cắt về **1500 byte** để có kích thước cố định cho packet node.

## 4. Edge features

### 4.1 Contain edge

Edge nối `flow -> packet` có 4 feature:

| Feature | Ý nghĩa |
|---|---|
| `packet_direction` | Hướng packet |
| `ip_size` | Kích thước IP packet |
| `transport_size` | Kích thước transport layer |
| `payload_size` | Kích thước payload |

### 4.2 Link edge

Edge nối packet liên tiếp có 1 feature:

| Feature | Ý nghĩa |
|---|---|
| `delta_time` | Độ chênh thời gian giữa hai packet liên tiếp |

## 5. Cách hiểu nhanh theo mục đích phát hiện tấn công

- **Scan / Recon**: thường làm tăng `Unique_Ports_In_SourceDestination*`, các rolling TCP/UDP count, và các feature liên quan cổng.
- **Spoofing**: dễ để lại pattern ACK/PSH bất thường và phân bố packet theo chiều không tự nhiên.
- **DoS / DDoS**: làm tăng mạnh rolling packet count, SYN/ACK count, và giảm khoảng cách thời gian giữa packet.
- **Web / HTTP traffic**: nổi bật ở `Rolling_http_port_*`.
- **DNS abuse**: nổi bật ở `Rolling_DNS_*`.

## 6. Ghi chú thực tế trong repo

- File tạo feature mở rộng là [Utility/Additional_Features.py](Utility/Additional_Features.py).
- File ghép graph và chọn feature cho flow node là [Utility/Functions.py](Utility/Functions.py).
- Danh sách feature flow node theo đúng thứ tự schema hiện dùng nằm trong [Utility/Explainer.py](Utility/Explainer.py).

Nếu muốn, có thể mở rộng tài liệu này thành bảng 82 dòng, mỗi dòng một feature và một câu mô tả ngắn.