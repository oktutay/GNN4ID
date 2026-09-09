# Features và ý nghĩa trong GNN4ID

Tài liệu này tóm tắt các nhóm feature mà repo GNN4ID dùng để tạo graph cho XG-NID.
Nguồn chính của phần mô tả là:

- [Utility/Functions.py](Utility/Functions.py)
- [Utility/Additional_Features.py](Utility/Additional_Features.py)
- [Utility/Explainer.py](Utility/Explainer.py)

## 1. Tổng quan schema

- **Flow node**: 82 feature
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

### 2.3 Feature mở rộng theo rolling window

Nhóm này được tạo trong [Utility/Additional_Features.py](Utility/Additional_Features.py) với cửa sổ trượt `window_size=350`.

Ý nghĩa chung: đo mật độ và mẫu hành vi gần đây của destination hoặc cặp source-destination, để bắt các pattern như scan, brute force, DNS/HTTP probing, spoofing.

| Feature | Ý nghĩa |
|---|---|
| `Rolling_UDP_*` | Đếm packet UDP trong cửa sổ gần nhất |
| `Rolling_TCP_*` | Đếm packet TCP trong cửa sổ gần nhất |
| `Rolling_ICMP_*` | Đếm packet ICMP trong cửa sổ gần nhất |
| `Rolling_ACK_*` | Đếm packet có ACK flag |
| `Rolling_SYN_*` | Đếm packet có SYN flag |
| `Rolling_FIN_*` | Đếm packet có FIN flag |
| `Rolling_RST_*` | Đếm packet có RST flag |
| `Rolling_psh_*` | Đếm packet có PSH flag |
| `Rolling_http_port_*` | Tần suất truy cập cổng HTTP/HTTPS phổ biến |
| `Rolling_DNS_*` | Tần suất truy vấn DNS |
| `Rolling_vulnerable_port` | Tần suất đụng tới cổng thường bị khai thác |
| `Rolling_packets_*` | Tổng packet trong cửa sổ |
| `Rolling_bipackets_*` | Tổng packet hai chiều trong cửa sổ |
| `Rolling_Average_Duration` / `Rolling_Duration_*` | Thời lượng trung bình của flow gần đây |
| `Unique_Ports_In_SourceDestination*` | Số cổng nguồn khác nhau trong cặp source-destination, hữu ích cho phát hiện scan |

> Lưu ý: tên feature rolling trong file tạo feature và file explainer của repo có vài biến thể tên gọi, nhưng ý nghĩa ngữ nghĩa là như nhau.

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