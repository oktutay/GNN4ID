# <img src="https://github.com/user-attachments/assets/1f788a19-dffa-4a93-b780-c767c03de45d" width="120" valign="middle" alt="Scapy" />&nbsp; GNN4ID

<p align="justify">GNN4ID is a tool designed to transform raw packet capture files (PCAP) of network traffic into structured graph-based datasets. This tool uniquely integrates both flow and packet-level information, providing a comprehensive view of network activity. Developed to facilitate research in Graph Neural Networks (GNNs) for Network Intrusion Detection Systems (NIDS), GNN4ID empowers users to seamlessly extract flow-level information along with its respective packet-level information, which is ultimately combined to form a graph-based dataset. The developed graph-based dataset utilizes both flow-level and packet-level information. </p>

## Usage
<p align="justify">
GNN4ID can be utilized to extract and create graph objects from any network traffic data. The primary requirement is that the network traffic is available in its raw pcap format and that the pcap files or individual packets are appropriately labeled. If the provided pcap files are not labeled but their flow-level information is labeled, you can use the <a href="https://github.com/Yasir-ali-farrukh/Payload-Byte">Payload-Byte</a> Tool to label the raw packets with respect to their corresponding flow.

<p align="justify">
The input files required for GNN4ID are labeled pcap files or labeled packet information. Alternatively, you can use the provided flow information along with their respective packet-level information. For ease of usage, we have provided three notebook files that can be used to extract and generate graph data objects:


1. [`GNN4ID.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID.ipynb): This notebook provides instructions on how to handle pcap files for flow and packet-level information extraction and how to transform the extracted flow and packet-level information into graph data objects. You can either create the graph data objects directly or use the extracted combined information of flow and packet for your own use case.
2. [`Data_preprocessing_CIC-IoT2023.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/Data_preprocessing_CIC-IoT2023.ipynb): This notebook details the preprocessing of the CIC-IoT2023 dataset, specifically addressing the issue of class imbalance and providing the dataset sample size for our example.
3. [`GNN4ID_Model.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID_Model.ipynb): This notebook provides details on training and testing a GNN model. We have incorporated two different models, one with edge attributes and another without edge attributes.
</p>



To start, you can begin with [`GNN4ID.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID.ipynb), as it provides comprehensive details on information extraction from pcap files and the creation of graph models. For ease of computation, you should follow the steps in [`Data_preprocessing_CIC-IoT2023.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/Data_preprocessing_CIC-IoT2023.ipynb) before creating the data objects (as highlighted in the notebook)/Or download the processed CIC-IoT2023 dataset directly from the [Link](https://drive.google.com/drive/folders/1FiZh87vvCZF3gX1Fnj9iTB4j74u-nuR6?usp=drive_link). Lastly, [`GNN4ID_Model.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID_Model.ipynb) provides a baseline on how to develop a model, train it, and test it.

## Graph Data Modeling
<p align="justify">
To model network traffic data into a graph data structure, we have utilized the relationships between packets and how combined packets are used to create flow information. By leveraging this connection, we can transform network traffic into graph objects. We adopted heterogeneous graph modeling as it allows us to model two distinct types of nodes with their own attributes. Specifically, we used flow information as one node type and packet information as another. Consequently, our graph comprises flow nodes connected with their respective packets.

<p align="justify">
For our experimentation and real-time detection, we set a limit on the maximum number of packets in a flow to 20. This means that if a flow contains 20 packets, the flow is terminated, and a new flow is computed. This approach ensures real-time detection, preventing flows from containing a large number of packets, which could lead to several minutes of delay.

<p align="justify">
As we have two types of nodes, we also have two different types of edges: link edges and contain edges. The link edges connect packet nodes to packet nodes, while the contain edges connect flow nodes to packet nodes. The attributes of each node and edge are as follows:

1. **Flow Node**: 82 features = 46 NFStream statistics + `packet_size_variation` + 28 rolling-window explainable features (paper Table 1, two groupings: per destination and per src-dst pair) + 7 one-hot (`Exp_*`, `proto_*`). The 29 identifier / redundant NFStream columns are dropped before the graph; the exact list and order live in `Utility/Schema.py` (`FLOW_FEATURE_NAMES_82`).
2. **Packet Node**: 1500 Features (Payload Data Byte-wise)
3. **Contain Edge**: 4 Features (packet direction + IP / transport / payload sizes)
4. **Link Edge**: 1 Feature (Time delta between each consecutive packet)

A pictorial representation of the graph object is provided below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd55c744-7227-4b49-a970-2729bd8c0de6" width="400" height="300">
</p>

## Citation 
 If you are using our tool, kindly cite our paper  [paper](https://arxiv.org/abs/2408.16021) which outlines the details of the graph modeling and processing. 


 ```yaml
@article{GNN4ID,
  title={XG-NID: Dual-modality network intrusion detection using a heterogeneous graph neural network and large language model},
  author={Farrukh, Yasir Ali and Wali, Syed and Khan, Irfan and Bastian, Nathaniel D},
  journal={Expert Systems with Applications},
  volume={287},
  pages={128089},
  year={2025},
  publisher={Elsevier}
}
```




## Preprocessing v2 (leak-free, author-compatible) — `run_preprocessing.py`

One command replaces the notebook sequence and produces the same `df_class_8_{train,test}.csv`
layout (97 columns, byte-compatible with the published Google-Drive CSVs and the checkpoints):

```bash
python run_preprocessing.py --pcap-dir /path/to/CIC_IoT2023/PCAP --out-dir /path/to/out            # authors' setup
python run_preprocessing.py --pcap-dir ... --out-dir ... --no-oversample                               # class-weight variant
python run_preprocessing.py --pcap-dir ... --out-dir ... --window 60s --window-unit time               # time window (Algorithm 1)
python run_preprocessing.py --csv-dir /path/to/raw_csv --out-dir ... --stages features,split,combine,class8
python Debug/verify_preprocessing.py --out-dir /path/to/out --drive-train df_class_8_train.csv          # 9 groups of assertions
```

Stage order and the leakage argument:

```
extract   pcap -> raw/<Class>-<Short>_<n>.csv     NFStream, 20 packets/flow, idle 120 s
features  raw  -> features/<stem>.csv             28 rolling columns + packet_size_variation + one-hots,
                                                  computed on the WHOLE time-ordered file (label-free,
                                                  backward-looking -> a later temporal split cannot leak;
                                                  computing them after undersampling, as the June-2026
                                                  revision did, gives train and test different units)
split     features -> split/{train,test}/         attacker-MAC filter, latest 20 % = test pool (cap 4000),
                                                  earlier 80 % = train pool (feature dedup, cap 20000/file)
combine   split -> combined/{train,test}/         per class: dedup, proportional per-sub-attack caps,
                                                  GLOBAL test-vs-train removal, train-only oversampling to
                                                  20000 (paper Table 4); class_weights.json (pre-oversample)
class8    combined -> df_class_8_{train,test}.csv drop the 29 identifier columns, assert the 97-col header
graphs    (optional) NIDSDataset -> processed/    same as build_graphs.py
```

Options: `--window-unit flows|time|packets` (default `flows`, 350 = the authors' code; the paper only says
"rolling time window"), `--count-mode author|packets|flows` (the authors count UDP/TCP/ICMP/HTTP/DNS as
flows but flags as packets), `--schema author82|table1`, `--rolling-scope all|filtered`, `--test-pick
random|tail`, `--n-dissections N` (adds NFStream L7 columns such as SNI/JA3, dropped before the graph
unless `--keep-l7`), `--include-packetflag` (1508-d packet nodes). File names are resolved with a
case-insensitive longest-prefix match on the 34 CIC-IoT2023 folder names (`Utility/Schema.py::CIC_SUBTYPES`),
which also fixes the `DDoS-SlowLoris` / `BrowserHijacking` / `BenignTraffic*` naming bugs of the original
`name_mapping`. Every stage is re-runnable, nothing is overwritten, and `manifest.json` records the counts.
`make_preprocess_zip.py` packages the code (no data) for another machine. Details: `BAOCAO_PREPROCESS_V2.md`.
