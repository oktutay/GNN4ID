# Báo cáo Metric — Kết quả tái dựng XG-NID

Tổng hợp toàn bộ metric của 2 lần huấn luyện trên CIC-IoT2023 (25,557 test
graphs, 8 class). Tính từ confusion matrix raw (run 1 có 1 lưu ý: code gốc
gọi `confusion_matrix(y_pred, y_true)` đảo args nên ma trận in ra bị
transpose; mình đã chuyển về dạng chuẩn rows=true, cols=pred trước khi tính
precision/recall/f1).

---

## 1. Bảng tổng quát hai lần huấn luyện

| Cấu hình | Run 1 — Baseline (`train.py`) | Run 2 — Tuned (`train_tuned.py`) |
|---|---|---|
| Optimizer | Adam | AdamW |
| Learning rate | 0.01 (fixed) | 5e-3 → 1e-5 (CosineAnnealingLR) |
| Weight decay | 0 | 1e-4 |
| Batch size | 64 | 128 |
| DataLoader workers | 0 | 4 |
| Train/Val split | – (toàn 160k train) | 90/10 (144k / 16k) |
| Loss | NLLLoss uniform | NLLLoss (class weights gần uniform vì train balanced) |
| Gradient clip | – | max_norm=5.0 |
| Epochs | 30 | 50 |
| Time on RTX 3060 | ~75 phút | ~17 phút |
| **Accuracy** | **0.9520** | **0.9820** |
| Macro F1 | 0.9361 | 0.9745 |
| Weighted F1 | 0.9543 | 0.9822 |
| Δ Accuracy | (baseline) | **+3.00 pt** |

---

## 2. Per-class metric — Run 1 (Baseline 95.20%)

Test set: 25,557 graphs. Confusion matrix gốc trong log có rows=pred,
cols=true (do bug `confusion_matrix(y_pred, y_true)` trong
`Utility/Training.py:calculate_metrics`). Bảng dưới đã chuyển về chuẩn.

| Class | Support | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Benign | 4000 | 0.9759 | 0.9103 | 0.9420 |
| WebBased | 1090 | **0.6051** | 0.9560 | **0.7413** |
| Spoofing | 4000 | 0.9276 | 0.9448 | 0.9361 |
| Recon | 4000 | 0.9589 | 0.8640 | 0.9090 |
| Mirai | 4000 | 1.0000 | 0.9983 | 0.9991 |
| Dos | 4000 | 0.9995 | 0.9958 | 0.9976 |
| DDos | 4000 | 1.0000 | 0.9980 | 0.9990 |
| BruteForce | 467 | 0.9759 | 0.9529 | 0.9643 |
| **accuracy** | 25557 | | | **0.9520** |
| **macro avg** | 25557 | 0.9304 | 0.9525 | 0.9361 |
| **weighted avg** | 25557 | 0.9611 | 0.9520 | 0.9543 |

**Confusion matrix (rows = true, cols = predicted):**
```
                   Ben    Web    Spo    Rec    Mir    Dos   DDos     BF
Benign     [    3641     72    232     53      0      0      0      2 ]
WebBased   [       2   1042      8     35      0      0      0      3 ]
Spoofing   [      80     90   3779     46      0      0      0      5 ]
Recon      [       7    495     39   3456      0      2      0      1 ]
Mirai      [       0      1      6      0   3993      0      0      0 ]
Dos        [       1      0      2     14      0   3983      0      0 ]
DDos       [       0      5      3      0      0      0   3992      0 ]
BruteForce [       0     17      5      0      0      0      0    445 ]
```

**Điểm yếu rõ rệt run 1:** WebBased precision chỉ 60.51% — 495 mẫu Recon
bị model phân loại nhầm là WebBased.

---

## 3. Per-class metric — Run 2 (Tuned 98.20%)

Trích trực tiếp từ `sklearn.classification_report` ở final eval của
[train_tuned.py](GNN4ID/train_tuned.py).

| Class | Support | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Benign | 4000 | 0.9730 | 0.9720 | 0.9725 |
| WebBased | 1090 | 0.8577 | 0.9789 | 0.9143 |
| Spoofing | 4000 | 0.9719 | 0.9685 | 0.9702 |
| Recon | 4000 | 0.9894 | 0.9585 | 0.9737 |
| Mirai | 4000 | 1.0000 | 0.9992 | 0.9996 |
| Dos | 4000 | 0.9997 | 0.9968 | 0.9982 |
| DDos | 4000 | 0.9992 | 0.9992 | 0.9992 |
| BruteForce | 467 | 0.9639 | 0.9722 | 0.9680 |
| **accuracy** | 25557 | | | **0.9820** |
| **macro avg** | 25557 | 0.9694 | 0.9807 | 0.9745 |
| **weighted avg** | 25557 | 0.9828 | 0.9820 | 0.9822 |

**Confusion matrix (rows = true, cols = predicted):**
```
                   Ben    Web    Spo    Rec    Mir    Dos   DDos     BF
Benign     [    3888     12     81     14      0      0      1      4 ]
WebBased   [       7   1067      5      7      0      0      0      4 ]
Spoofing   [      80     32   3874      6      0      0      2      6 ]
Recon      [      19    125     19   3834      0      1      0      2 ]
Mirai      [       1      1      1      0   3997      0      0      0 ]
Dos        [       0      0      0     12      0   3987      0      1 ]
DDos       [       0      1      0      2      0      0   3997      0 ]
BruteForce [       1      6      6      0      0      0      0    454 ]
```

**Confusion còn lại:** Recon → WebBased (125 mẫu) — pattern semantic
(cả hai dùng HTTP-port/unique-port fan-out), được IG explainer xác nhận
đúng nguyên nhân.

---

## 4. So sánh tăng tiến giữa 2 run

| Class | Run 1 P | Run 2 P | ΔP | Run 1 R | Run 2 R | ΔR | Run 1 F1 | Run 2 F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 0.9759 | 0.9730 | -0.0029 | 0.9103 | 0.9720 | +0.0617 | 0.9420 | 0.9725 | **+0.0305** |
| WebBased | 0.6051 | 0.8577 | **+0.2526** | 0.9560 | 0.9789 | +0.0229 | 0.7413 | 0.9143 | **+0.1730** |
| Spoofing | 0.9276 | 0.9719 | +0.0443 | 0.9448 | 0.9685 | +0.0237 | 0.9361 | 0.9702 | +0.0341 |
| Recon | 0.9589 | 0.9894 | +0.0305 | 0.8640 | 0.9585 | +0.0945 | 0.9090 | 0.9737 | **+0.0647** |
| Mirai | 1.0000 | 1.0000 | 0.0000 | 0.9983 | 0.9992 | +0.0009 | 0.9991 | 0.9996 | +0.0005 |
| Dos | 0.9995 | 0.9997 | +0.0002 | 0.9958 | 0.9968 | +0.0010 | 0.9976 | 0.9982 | +0.0006 |
| DDos | 1.0000 | 0.9992 | -0.0008 | 0.9980 | 0.9992 | +0.0012 | 0.9990 | 0.9992 | +0.0002 |
| BruteForce | 0.9759 | 0.9639 | -0.0120 | 0.9529 | 0.9722 | +0.0193 | 0.9643 | 0.9680 | +0.0037 |
| **Macro** | 0.9304 | 0.9694 | +0.0390 | 0.9525 | 0.9807 | +0.0282 | 0.9361 | 0.9745 | **+0.0384** |
| **Weighted** | 0.9611 | 0.9828 | +0.0217 | 0.9520 | 0.9820 | +0.0300 | 0.9543 | 0.9822 | **+0.0279** |

**Phân tích:**
- Lợi ích lớn nhất rơi vào **WebBased** (F1 +17.3 pt) và **Recon** (F1
  +6.5 pt) — đúng hai class chia sẻ confusion ở run 1. Cosine LR + val
  selection cho phép model học được boundary tốt hơn cho hai class này.
- Mirai/Dos/DDos đã saturate gần 99.9% ở run 1, nên gain nhỏ.
- BruteForce precision giảm nhẹ (-0.012) vì run 2 phân loại tổng quát
  hơn 1 ít, nhưng recall tăng đủ bù → F1 tăng.

---

## 5. So sánh reproduction vs paper

Paper báo cáo accuracy ~99% trên CIC-IoT2023 nhưng **không công bố table
per-class P/R/F1 đầy đủ** trong abstract/summary mình có. Bảng dưới là so
sánh các điểm có thể verify.

| Metric | Paper (claim) | Reproduction (tuned) | Gap |
|---|---|---|---|
| Overall accuracy | ~0.99 | 0.9820 | ~−0.008 |
| Số class | 8 (Benign + 7 family) | 8 | 0 |
| Architecture | 2× GATConv + edge attr + GMP + FC | giống | 0 |
| Flow node features | 76 (theo paper) | 82 (theo toolkit GNN4ID) | +6 (dummy expansion) |
| Packet node features | 1500 payload bytes | 1500 | 0 |
| Test set size | (paper không nói rõ trong tóm tắt) | 25,557 | – |

---

## 6. Metric áp dụng cho explainer (IG + LLM)

Hai khối explainer là **post-hoc**, không có ground-truth label cho
attribution → metric chỉ ở mức smoke-test qualitative:

| Aspect | Verify trên |
|---|---|
| IG attribution sign + magnitude | Sample idx 22000 (true=Spoofing, pred=Spoofing). IG nêu Rolling_ACK + Rolling_psh — đúng signature ARP/DNS spoofing. |
| IG misclassification analysis | Sample idx 20000 (true=Recon, pred=WebBased). IG nêu Unique_Ports + Rolling_UDP_Requests — đúng signature scan, xác nhận model "nghe" được tín hiệu Recon nhưng vẫn vote sai → đó là 1 trong 125 confusion Recon→WebBased. |
| Prompt builder (Algorithm-3) | Output prompt structure đầy đủ 4 phần (header, flow top-k, payload top-k, ASCII render) + 3 task instructions. Test với PreviewBackend OK. |
| LLM end-to-end | HFCausalBackend wired sẵn cho Qwen2.5-1.5B-Instruct (không test live trong session để tiết kiệm GPU mem). |

---

## 7. Lưu ý quan trọng về run 1 confusion matrix

Trong [Utility/Training.py:calculate_metrics](GNN4ID/Utility/Training.py)
của repo gốc:
```python
print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
```
Tham số bị **đảo thứ tự** so với chuẩn sklearn (`confusion_matrix(y_true,
y_pred)`). Hệ quả: ma trận in trong log run 1 có rows = predicted,
cols = true. Mọi số trong bảng run 1 ở mục 2 đã được chuẩn hóa về dạng
rows=true, cols=pred trước khi tính P/R/F1.

Run 2 dùng custom `evaluate()` trong [train_tuned.py](GNN4ID/train_tuned.py)
gọi `confusion_matrix(labels, preds)` đúng thứ tự nên không bị vấn đề
này.

---

## 8. Files chứa metric

| File | Nội dung |
|---|---|
| `GNN4ID/logs/train.log` | Log run 1 đầy đủ |
| `GNN4ID/logs/train_tuned.log` | Log run 2, có classification_report + CM |
| `GNN4ID/checkpoints/xgnid_hgnn_edge.pth` | Checkpoint run 1 (95.20%) |
| `GNN4ID/checkpoints/xgnid_hgnn_edge_tuned.pth` | Checkpoint run 2 (98.20%) |
| `BANG_KET_QUA_METRIC.md` | File này |
| `BAO_CAO_CHI_TIET.md` | Báo cáo tổng — bao gồm metric + audit + repro commands |
| `GNN4ID/REPRODUCTION_REPORT.md` | English version |
