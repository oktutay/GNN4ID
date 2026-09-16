# XG-NID Reproduction Report

Reproduction of *"XG-NID: Dual-Modality Network Intrusion Detection using a
Heterogeneous Graph Neural Network and Large Language Model"* (Farrukh et al.,
ESWA 2025) using the authors' GNN4ID open-source toolkit + completion of the
two missing explainer blocks.

## TL;DR

| Block | Paper | Code | Status |
|---|---|---|---|
| 1. Flow & Feature Generator | NFStream, idle_timeout=120s, 20 packets/flow | `Utility/Feature_extractor_flow_packet_combined.py` | reused as-is |
| 2. Explainable Feature Extractor | rolling-window temporal features | `Utility/Additional_Features.py` | reused as-is |
| 3. Graph Generator | flow + packet hetero graph, contain/link edges | `Utility/Functions.py:NIDSDataset` | reused as-is |
| 4. HGNN | 2× GATConv + edge attrs, BN+LeakyReLU, global mean pool, FC head | `Utility/Model.py:HeteroGNN_Edge` | reused, tuned training added |
| 5. Integrated Gradients explainer | Captum IG over HGNN inputs | `Utility/Explainer.py` (new) | reproduced |
| 6. Generative Explainer (LLM) | Algorithm-3 prompt, Llama-3-8B | `Utility/Generative_Explainer.py` (new) | reproduced (LLM pluggable; default preview / Qwen-1.5B) |

Final tuned test accuracy: **98.20%** on 25,557 CIC-IoT2023 graphs (8 classes).

## Paper-faithful HGNN (block 4) — verified

- **Heterogeneous nodes**: `flow` (82-dim flow stats incl. dummies), `packet`
  (1500-dim payload bytes). Paper said 76 flow features; toolkit yields 82
  due to expiration + protocol dummy expansion. Treated as a paper-vs-toolkit
  divergence, not corrected.
- **Edges**: contain (flow→packet, 4-dim attr: packet_direction, ip_size,
  transport_size, payload_size) and link (packet→packet, 1-dim attr:
  delta_time). Symmetric variants added by `T.ToUndirected()`.
- **Architecture**: 2× `GATConv((-1,-1), 64, edge_dim=-1, add_self_loops=False)`
  per edge type via `HeteroConv` → BatchNorm1d(64) → LeakyReLU → 2nd GAT layer
  → BN → LeakyReLU → global_mean_pool per node type → concat([flow,packet]) →
  Linear 128→64→16→8 → log_softmax.
- 431,704 trainable params.

## Block 5 — Integrated Gradients

`Utility/Explainer.py:explain_graph_ig` wraps the HGNN forward into a closure
over `(flow_x, packet_x)` and calls Captum's `IntegratedGradients`. Returns
attributions for both node types + ranked top-k flow feature names and top-k
payload byte indices. Default 32 IG steps, zero baseline.

Verified on two test samples:
- Sample idx 20000 (true=Recon, predicted=WebBased): IG surfaced
  `Unique_Ports_In_SourceDestinationIP=222`, `Rolling_UDP_Requests_*=222` as
  the dominant drivers — semantically correct (port-fan-out is the canonical
  recon/scan signature; the misclassification toward WebBased is the
  documented confusion in the test confusion matrix).
- Sample idx 22000 (true=Spoofing, predicted=Spoofing, log-prob -0.012): IG
  surfaced `Rolling_ACK_Packets`, `Rolling_psh_Packets` — consistent with
  MITM ARP/DNS spoofing's high-volume relayed-packet pattern.

## Block 6 — Generative Explainer

`Utility/Generative_Explainer.py:build_prompt` constructs an
Algorithm-3-style structured prompt with four sections:
1. Predicted class + log-prob.
2. Top-k flow features (name = value, IG attribution sign+magnitude).
3. Top-k payload bytes (index, hex, IG attribution).
4. ASCII-rendered first-packet payload (printable bytes only, 200-char cap).

Followed by the Algorithm-3 task instructions: explain decision citing the
listed features only; identify any known attack signature; recommend two
concrete mitigations.

LLM backend is pluggable:
- `PreviewBackend` — returns the prompt unchanged. Default; no weights, no GPU.
- `HFCausalBackend` — HuggingFace causal LM via `apply_chat_template`. Default
  model `Qwen/Qwen2.5-1.5B-Instruct` (no gated access). Paper used
  `Llama-3-8B-Instruct`; replace `--hf-model meta-llama/Meta-Llama-3-8B-Instruct`
  + `--load-4bit` if you have HF access. The bitsandbytes path is wired
  through `BitsAndBytesConfig(load_in_4bit=True)`.

## Training, tuning, and final result

Two training runs, both on RTX 3060 12GB / CUDA 12.1.

### Run 1 — paper-default (`train.py`)
| | |
|---|---|
| Optimizer | Adam, lr=0.01 fixed (no scheduler — author's `train_with_edge_Att`) |
| Batch | 64 |
| Epochs | 30 |
| Loss | NLLLoss (uniform) |
| Train acc / loss (last) | 95.87% / 0.108 |
| **Test accuracy** | **95.20%** |
| Time | ~75 min |

Per-epoch loss bounced (0.16 → 0.14 → 0.16) — fixed lr too high to settle.

### Run 2 — tuned (`train_tuned.py`)
| | |
|---|---|
| Optimizer | AdamW, lr=5e-3, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR(T_max=50, eta_min=1e-5) |
| Batch | 128 (`num_workers=4`, `pin_memory`) |
| Train/val split | 90/10 (144k / 16k) |
| Epochs | 50 |
| Loss | NLLLoss with class weights from train (effectively uniform — train is balanced 20k/class) |
| Grad clip | 5.0 |
| Best-val-acc checkpoint | epoch 50, val 98.67% |
| **Test accuracy** | **98.20%** |
| Time | ~17 min |

Speed-up vs run 1 (75 min → 17 min) is mostly batch=128 + 4 dataloader workers
on the same GPU; cosine schedule produced smoother convergence (no bounce).

#### Per-class test results (tuned)
```
              precision  recall  f1     support
Benign        0.9730     0.9720  0.9725  4000
WebBased      0.8577     0.9789  0.9143  1090
Spoofing      0.9719     0.9685  0.9702  4000
Recon         0.9894     0.9585  0.9737  4000
Mirai         1.0000     0.9992  0.9996  4000
Dos           0.9997     0.9968  0.9982  4000
DDos          0.9992     0.9992  0.9992  4000
BruteForce    0.9639     0.9722  0.9680   467
accuracy                         0.9820  25557
macro avg     0.9694     0.9807  0.9745  25557
weighted avg  0.9828     0.9820  0.9822  25557
```

#### Tuned confusion matrix
```
                  pred →
true ↓  Ben  Web  Spo  Rec  Mir  Dos DDos   BF
Benign[3888   12   81   14    0    0    1    4]
Web   [   7 1067    5    7    0    0    0    4]
Spoof [  80   32 3874    6    0    0    2    6]
Recon [  19  125   19 3834    0    1    0    2]
Mirai [   1    1    1    0 3997    0    0    0]
Dos   [   0    0    0   12    0 3987    0    1]
DDos  [   0    1    0    2    0    0 3997    0]
BF    [   1    6    6    0    0    0    0  454]
```

Largest residual confusion: **Recon → WebBased (125)** — both classes share
HTTP-port / unique-port-fanout signals; the IG attribution on misclassified
samples reflects this overlap rather than a model bug.

### Gap to paper
Paper reports ~99% accuracy on CIC-IoT2023. We obtained 98.20%. Remaining
~0.8 pt gap is plausibly explained by:
- Different random seed / data ordering during the author's
  oversampling-balanced training set construction.
- Possibly different hyperparameter choices the paper does not list (the
  paper does not publish a tuned-hyperparameter table for the HGNN).
- Single-run result; paper may average over seeds.

No paper-vs-code architectural divergence remaining at this point.

## Reproducibility commands

```bash
# 0. environment (pre-existing on this host)
conda activate xmfgnn
pip install gdown captum  # already done

# 1. data — preprocessed CSVs from the authors' Google Drive
mkdir -p data/CIC_IoT2023_Processed_Data
gdown --folder \
  "https://drive.google.com/drive/folders/1FiZh87vvCZF3gX1Fnj9iTB4j74u-nuR6" \
  -O data/CIC_IoT2023_Processed_Data

# 2. materialize 160k+25.5k graph objects (~12 GB, ~16 min)
python build_graphs.py data/CIC_IoT2023_Processed_Data

# 3a. paper-default training (30 ep, baseline)
python train.py --data-root data/CIC_IoT2023_Processed_Data \
  --skip-processing --model edge --epochs 30 --batch-size 64 --lr 0.01 \
  --save-path checkpoints/xgnid_hgnn_edge.pth

# 3b. tuned training (50 ep, recommended)
python train_tuned.py --data-root data/CIC_IoT2023_Processed_Data \
  --epochs 50 --batch-size 128 --lr 5e-3 --weight-decay 1e-4 \
  --save-path checkpoints/xgnid_hgnn_edge_tuned.pth

# 4. explain a test sample (preview prompt; no LLM weights)
python explain.py --data-root data/CIC_IoT2023_Processed_Data \
  --checkpoint checkpoints/xgnid_hgnn_edge_tuned.pth \
  --idx 22000 --n-steps 32 --backend preview

# 4b. with a real LLM (downloads ~3 GB)
python explain.py --data-root data/CIC_IoT2023_Processed_Data \
  --checkpoint checkpoints/xgnid_hgnn_edge_tuned.pth \
  --idx 22000 --backend hf --hf-model Qwen/Qwen2.5-1.5B-Instruct
```

## Files added in this reproduction

| File | Purpose |
|---|---|
| `train.py` | Paper-default training driver (CLI). |
| `train_tuned.py` | Tuned training (val split, AdamW, cosine, class weights). |
| `build_graphs.py` | One-shot CSV → graph object materialization. |
| `Utility/Explainer.py` | Block 5 — Captum Integrated Gradients on the HGNN. |
| `Utility/Generative_Explainer.py` | Block 6 — Algorithm-3 prompt + LLM backend. |
| `explain.py` | End-to-end explainability CLI (IG + LLM). |
| `REPRODUCTION_REPORT.md` | This file. |
| `checkpoints/xgnid_hgnn_edge.pth` | Baseline 30-epoch checkpoint (95.20%). |
| `checkpoints/xgnid_hgnn_edge_tuned.pth` | Tuned 50-epoch checkpoint (98.20%). |

## Known divergences (faithfulness map)

- Flow feature count: paper 76 vs toolkit 82 (dummy-encoded expiration_id +
  protocol). Toolkit count is what the public NIDSDataset emits; we kept it.
- Generative-explainer LLM: paper uses Llama-3-8B-Instruct; default backend
  here is Qwen2.5-1.5B-Instruct (open weights, no gated access). The Llama
  model is wired but not exercised end-to-end on this host.
- Tuned training (run 2) is an *engineering improvement*, not a paper-stated
  recipe. The original `Utility/Training.py:train_with_edge_Att` (no
  scheduler, no val split) is also retained as run 1 / `train.py` for direct
  paper reproduction.
