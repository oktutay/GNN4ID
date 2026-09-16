# Utility/ — module map

| Module | Role |
| --- | --- |
| `Schema.py` | Single source of truth for the column schema: the 82 flow features / 97-column class-8 header in the authors' order, the 29 identifier columns dropped before the graph, NFStream L7 columns, the 34 CIC-IoT2023 folder names -> class mapping, attacker MACs, port lists. |
| `Feature_extractor_flow_packet_combined.py` | Flow and Feature Generator: NFStream + `My_Custom` plugin (20 packets per flow, idle timeout 120 s, per-packet payload/flag/size lists). `extract_pcap()` + CLI. |
| `Additional_Features.py` | Explainable Feature Extractor: the authors' 28 rolling-window columns (window = 350 previous flows per destination by default; `window_unit='time'|'packets'`, `count_mode`, `schema` options). Run on the whole time-ordered raw file, before any split. |
| `Functions.py` | `NIDSDataset` (CSV row -> HeteroData graph), CIC-IoT2023 file-name resolver, `mac_filter`, `split_csv` (temporal split + caps), `Combining_classes` (dedup, proportional caps, train-only oversampling), `build_class8_csvs` (29-column drop + header check), dedup / sampling helpers. |
| `Model.py` | HGNN (2x GATConv with edge attributes, mean pooling, MLP head) and the SAGE ablation. |
| `Training.py` | train / test loops, confusion matrix, class-weighted NLL. |
| `IG_Explainer.py`, `LLM_Explainer.py`, `Explainer.py`, `Generative_Explainer.py` | Integrated-Gradients explainer and the Llama-3 generative explainer (paper Sec. 3.1.5-3.1.6). |

Pipeline entry point: `../run_preprocessing.py` (extract -> features -> split -> combine -> class8 -> graphs); tests in `../tests/`.
