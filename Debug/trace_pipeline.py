#!/usr/bin/env python3
"""Step-by-step driver for the whole XG-NID pipeline on a couple of PCAP files.

Purpose: *study / debug*, not training. It runs the real project functions, in
the real order, on a small input (the two PCAPs in data/Debug and Trace/), so
you can either

  * set breakpoints in VS Code and step through them (F5 -> "XG-NID: trace ..."), or
  * run it plainly and read the produced call-trace log
    (Debug and Trace/trace_out/logs/trace_*.trace).

The six stages mirror GNN4ID.ipynb + Data_preprocessing_CIC-IoT2023.ipynb:

  1 extract   PCAP  -> raw CSV        NFStreamer + My_Custom (on_init / on_update)
  2 split     raw CSV -> train/test   Utility.Functions.split_csv       (MAC filter,
                                      dedup, temporal 80/20)
  3 features  per split, in place     Utility.Additional_Features.additional_features
                                      (16 rolling features + one-hot)
  4 combine   per class -> 1 file     Utility.Functions.Combining_classes + the
                                      notebook's concat / drop-biased-columns cells
  5 graphs    CSV -> HeteroData .pt   Utility.Functions.NIDSDataset.process
  6 model     .pt -> train/eval       Utility.Model.HeteroGNN_Edge +
                                      Utility.Training.train / test_cm

Nothing here writes into the real dataset directory, and the PCAP files are
never deleted (GNN4ID.ipynb does `os.remove(pcap)` after extraction - we don't).

Examples
--------
    # everything, small and fast
    python Debug/trace_pipeline.py --stage all --max-flows 400 --max-graphs 120

    # only the graph construction, on what is already on disk
    python Debug/trace_pipeline.py --stage graphs --max-graphs 20

    # start over
    python Debug/trace_pipeline.py --stage all --reset
"""

import argparse
import glob
import json
import inspect
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                 # .../XG_NID/GNN4ID
REPO = os.path.dirname(PROJECT)                 # .../XG_NID
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from Debug.tracer import CallTracer, Tee, summarize  # noqa: E402

DEFAULT_PCAP_DIR = os.path.join(REPO, "data", "Debug and Trace")

# GNN4ID.ipynb, cell "name_mapping": pcap base name -> "<BroadClass>-<SubClass>".
# split_csv()/Combining_classes() read the broad class from the part before '-'.
NAME_MAPPING = {
    'Benign': 'Benign-Benign',
    'DDoS-ACK_Fragmentation': 'DDos-AckFrg', 'DDoS-UDP_Flood': 'DDos-UDPFlood',
    'DDos-SlowLoris': 'DDos-SlowLoris', 'DDoS-ICMP_Flood': 'DDos-ICMPFlood',
    'DDoS-RSTFINFlood': 'DDos-RSTFIN', 'DDoS-PSHACK_Flood': 'DDos-PSHACK',
    'DDoS-HTTP_Flood': 'DDos-HTTPFlood', 'DDoS-UDP_Fragmentation': 'DDos-UDPFrg',
    'DDoS-ICMP_Fragmentation': 'DDos-ICMPFrg', 'DDoS-TCP_Flood': 'DDos-TCPFlood',
    'DDoS-SYN_Flood': 'DDos-SYNFlood', 'DDoS-SynonymousIP_Flood': 'DDos-SynonymousIPFlood',
    'DoS-TCP_Flood': 'Dos-TCPFlood', 'DoS-HTTP_Flood': 'Dos-HTTPFlood',
    'DoS-SYN_Flood': 'Dos-SYNFlood', 'DoS-UDP_Flood': 'Dos-UDPFlood',
    'Recon-PingSweep': 'Recon-PingSweep', 'Recon-OSScan': 'Recon-OSScan',
    'VulnerabilityScan': 'Recon-VulScan', 'Recon-PortScan': 'Recon-PortScan',
    'Recon-HostDiscovery': 'Recon-HostDisc', 'SqlInjection': 'WebBased-SqlInject',
    'CommandInjection': 'WebBased-CmmdInject', 'Backdoor_Malware': 'WebBased-BckdoorMalware',
    'Uploading_Attack': 'WebBased-UploadAttack', 'XSS': 'WebBased-XSS',
    'BrowserHijacking': 'Webbased-BrwserHijack', 'DictionaryBruteForce': 'BruteForce-Dictionary',
    'MITM-ArpSpoofing': 'Spoofing-ARP', 'DNS_Spoofing': 'Spoofing-DNS',
    'Mirai-greip_flood': 'Mirai-GREIP', 'Mirai-greeth_flood': 'Mirai-Greeth',
    'Mirai-udpplain': 'Mirai-UDPPlain',
}

LABEL_DICT = {'Benign': 0, 'WebBased': 1, 'Spoofing': 2, 'Recon': 3,
              'Mirai': 4, 'Dos': 5, 'DDos': 6, 'BruteForce': 7}

# Data_preprocessing_CIC-IoT2023.ipynb, cells 12/14: identifiers + columns that
# leak the label or duplicate other features. Everything left must be numeric,
# because NIDSDataset feeds the whole row to np.asarray(dtype=float).
DROP_COLUMNS = [
    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'ip_version',
    'bidirectional_bytes', 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
    'bidirectional_duration_ms', 'bidirectional_packets',
    'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
    'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
    'id', 'src_mac', 'src_oui', 'dst_mac', 'dst_oui', 'vlan_id', 'tunnel_id',
    'bidirectional_syn_packets', 'bidirectional_cwr_packets', 'bidirectional_ece_packets',
    'bidirectional_urg_packets', 'bidirectional_ack_packets', 'bidirectional_psh_packets',
    'bidirectional_rst_packets', 'bidirectional_fin_packets',
]

_T0 = time.time()
_TRACER = None          # set in main(); step() annotates the trace log through it


# ---------------------------------------------------------------------------
# tiny logging helpers
# ---------------------------------------------------------------------------
def log(msg=""):
    print("      %s" % msg, flush=True)


def step(title):
    print("\n" + "=" * 100, flush=True)
    print("[%6.1fs] %s" % (time.time() - _T0, title), flush=True)
    print("=" * 100, flush=True)
    if _TRACER is not None:
        _TRACER.note(title)


def substep(title):
    print("\n  --- %s" % title, flush=True)


def where(fn):
    """Print where a function actually lives, so you know what you are stepping into."""
    try:
        f = inspect.getsourcefile(fn)
        _, line = inspect.getsourcelines(fn)
        log("calling %s()  ->  %s:%d" % (fn.__name__, os.path.relpath(f, REPO), line))
    except Exception:
        log("calling %s()" % getattr(fn, "__name__", fn))


def rows(path):
    import pandas as pd
    try:
        return pd.read_csv(path, low_memory=False).shape
    except Exception as exc:
        return "unreadable (%s)" % exc


# ---------------------------------------------------------------------------
# paths of the working directory
# ---------------------------------------------------------------------------
class Paths:
    def __init__(self, out_dir):
        self.work = out_dir
        self.raw = os.path.join(out_dir, "raw")          # NIDSDataset's raw_dir too
        self.raw_test = os.path.join(self.raw, "test")
        self.raw_train = os.path.join(self.raw, "train")
        self.processed = os.path.join(out_dir, "processed")
        self.logs = os.path.join(out_dir, "logs")
        self.train_csv = os.path.join(out_dir, "df_class_8_train.csv")
        self.test_csv = os.path.join(out_dir, "df_class_8_test.csv")
        self.train_csv_dbg = os.path.join(out_dir, "df_class_8_train_dbg.csv")
        self.test_csv_dbg = os.path.join(out_dir, "df_class_8_test_dbg.csv")

    def mk(self):
        for d in (self.work, self.raw, self.logs):
            os.makedirs(d, exist_ok=True)

    @property
    def state_file(self):
        return os.path.join(self.work, ".trace_state.json")


def _state(p):
    try:
        with open(p.state_file) as fh:
            return json.load(fh)
    except Exception:
        return {}


def state_done(p, stage):
    return bool(_state(p).get(stage))


def state_mark(p, stage, done=True):
    d = _state(p)
    d[stage] = done
    with open(p.state_file, "w") as fh:
        json.dump(d, fh, indent=1)


def guard(cfg, p, stage, why):
    """Stages 2 and 3 rewrite their input in place, so running them twice on the
    same working dir corrupts it. Refuse unless asked twice."""
    if state_done(p, stage) and not cfg.force:
        log("stage '%s' already ran in this working dir (see .trace_state.json)." % stage)
        log(why)
        log("use --force to run it anyway, or --reset to start again from the pcaps.")
        return True
    return False


# ---------------------------------------------------------------------------
# stage 1: PCAP -> flow+packet CSV
# ---------------------------------------------------------------------------
def _flows_in_process(streamer):
    """Run nfstream's own metering workflow in THIS process.

    NFStreamer normally forks a meter process per core, which means the
    My_Custom.on_init / on_update callbacks run in a child process: breakpoints
    and the tracer would not see them. nfstream.meter.meter_workflow is a plain
    function, so we can call it here with the streamer's own attributes and read
    the flows out of a plain queue instead.
    """
    import queue
    import threading
    from nfstream.meter import meter_workflow
    from nfstream.utils import NFEvent

    expected = ['source', 'snaplen', 'decode_tunnels', 'bpf_filter', 'promisc',
                'n_roots', 'root_idx', 'mode', 'idle_timeout', 'active_timeout',
                'accounting_mode', 'udps', 'n_dissections', 'statistics', 'splt',
                'channel', 'tracker', 'lock', 'group_id', 'system_visibility_mode']
    got = list(inspect.signature(meter_workflow).parameters)
    if got != expected:
        raise RuntimeError("nfstream.meter.meter_workflow signature changed: %s" % got)

    class _Val:
        def __init__(self):
            self.value = 0

    channel = queue.Queue()
    lock = threading.Lock()
    lock.acquire()                      # meter_workflow releases it when root_idx == n_roots-1
    tracker = [_Val(), _Val(), _Val()]

    meter_workflow(streamer.source, streamer.snapshot_length, streamer.decode_tunnels,
                   streamer.bpf_filter, streamer.promiscuous_mode, 1, 0, streamer._mode,
                   streamer.idle_timeout * 1000, streamer.active_timeout * 1000,
                   streamer.accounting_mode, streamer.udps, streamer.n_dissections,
                   streamer.statistical_analysis, streamer.splt_analysis,
                   channel, tracker, lock, os.getpid(), streamer.system_visibility_mode)

    idx = 0
    while True:
        recv = channel.get()
        if recv is None:
            break
        if recv.id == NFEvent.ERROR:
            raise RuntimeError("nfstream engine error: %s" % recv.message)
        recv.id = idx
        idx += 1
        yield recv


def stage_extract(cfg, p):
    step("STAGE 1/6  extract: PCAP -> flow CSV  (NFStreamer + My_Custom plugin)")
    import pandas as pd
    from nfstream import NFStreamer
    from Utility.Feature_extractor_flow_packet_combined import My_Custom

    log("real pipeline runs Utility/Feature_extractor_flow_packet_combined.py as a")
    log("subprocess (GNN4ID.ipynb cell 11). Here it is imported instead, so you can")
    log("breakpoint My_Custom.on_init / My_Custom.on_update.")
    log("extract-mode=%s  packet-limit=%s  max-flows=%s  bpf=%s"
        % (cfg.extract_mode, cfg.packet_limit, cfg.max_flows or "all", cfg.bpf or "-"))

    pcaps = sorted(glob.glob(os.path.join(cfg.pcap_dir, "*.pcap")) +
                   glob.glob(os.path.join(cfg.pcap_dir, "*.pcapng")))
    if not pcaps:
        sys.exit("no pcap in %s" % cfg.pcap_dir)

    for pcap in pcaps:
        base = os.path.basename(pcap).rsplit(".", 1)[0]
        mapped = NAME_MAPPING.get(base, base)
        out_csv = os.path.join(p.raw, mapped + ".csv")
        substep("%s  ->  %s.csv   (class = %s)"
                % (os.path.basename(pcap), mapped, mapped.split("-")[0]))
        log("rename_files() in Utility/Functions.py does this renaming on disk;")
        log("here we only apply the same NAME_MAPPING to the output csv name.")

        if os.path.exists(out_csv) and not cfg.force:
            log("exists, skipping (use --force / --reset to redo): %s" % out_csv)
            continue

        streamer = NFStreamer(source=pcap, accounting_mode=1, idle_timeout=120,
                             statistical_analysis=True, n_dissections=0,
                             bpf_filter=cfg.bpf or None,
                             udps=My_Custom(limit=cfg.packet_limit))
        t0 = time.time()
        if cfg.extract_mode == "inproc":
            try:
                it = _flows_in_process(streamer)
            except Exception as exc:
                log("in-process metering unavailable (%s) -> falling back to NFStreamer" % exc)
                it = iter(streamer)
        else:
            it = iter(streamer)

        records = []
        for flow in it:
            records.append(dict(zip(flow.keys(), flow.values())))
            if cfg.max_flows and len(records) >= cfg.max_flows:
                log("reached --max-flows=%d, stopping the metering early" % cfg.max_flows)
                break
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        log("%d flows, %d columns in %.1fs -> %s"
            % (df.shape[0], df.shape[1], time.time() - t0, out_csv))
        state_mark(p, "split", False)      # fresh raw csv -> split/features may run again
        state_mark(p, "features", False)
        if df.shape[0]:
            f0 = df.iloc[0]
            log("flow[0]: %s:%s -> %s:%s proto=%s packets=%s"
                % (f0['src_ip'], f0['src_port'], f0['dst_ip'], f0['dst_port'],
                   f0['protocol'], f0['bidirectional_packets']))
            log("flow[0] per-packet lists (My_Custom): payload_data=%s delta_time=%s"
                % (summarize(f0['udps.payload_data']), summarize(f0['udps.delta_time'])))
            log("  first payload hex[:60] = %s" % str(f0['udps.payload_data'][0])[:60])


# ---------------------------------------------------------------------------
# stage 2: split_csv
# ---------------------------------------------------------------------------
def stage_split(cfg, p):
    step("STAGE 2/6  split: attacker-MAC filter + dedup + temporal 80/20  (split_csv)")
    from Utility.Functions import split_csv
    where(split_csv)
    log("split_csv OVERWRITES its input with the train part and writes the test")
    log("part to raw/test/<name>_test.csv. It must run on the RAW csv, before")
    log("additional_features(), or the rolling features leak across the split.")

    if guard(cfg, p, "split", "re-running it would split an already-split (and already "
                              "enriched) train file."):
        return

    csvs = sorted(glob.glob(os.path.join(p.raw, "*.csv")))
    if not csvs:
        sys.exit("no csv in %s - run --stage extract first" % p.raw)

    for csv_path in csvs:
        name = os.path.basename(csv_path).rsplit(".", 1)[0]
        test_out = os.path.join(p.raw_test, name + "_test.csv")
        substep("split_csv(%s)" % os.path.basename(csv_path))
        if os.path.exists(test_out) and not cfg.force:
            log("already split (%s exists); skipping so the train part is not" % os.path.basename(test_out))
            log("split a second time. Re-run with --reset to start clean.")
            continue
        log("before: shape=%s" % (rows(csv_path),))
        split_csv(csv_path, test_sample=cfg.test_sample,
                  Number_in_individaul_class=cfg.train_cap)
        log("after : train part %s -> shape=%s" % (os.path.basename(csv_path), rows(csv_path)))
        log("        test  part %s -> shape=%s" % (os.path.relpath(test_out, p.raw), rows(test_out)))
    state_mark(p, "split")


# ---------------------------------------------------------------------------
# stage 3: additional_features
# ---------------------------------------------------------------------------
def stage_features(cfg, p):
    step("STAGE 3/6  features: 16 rolling temporal features per split  (additional_features)")
    import pandas as pd
    from Utility.Additional_Features import additional_features
    where(additional_features)
    log("run once per split (train and test separately) - paper Table 1 / Algorithm 1")

    if guard(cfg, p, "features", "it is not idempotent: the second pass fails because "
                                 "get_dummies already consumed protocol/expiration_id."):
        return

    targets = sorted(glob.glob(os.path.join(p.raw, "*.csv"))) + \
        sorted(glob.glob(os.path.join(p.raw_test, "*_test.csv")))
    if not targets:
        sys.exit("nothing to enrich - run --stage extract/split first")

    for f in targets:
        substep("additional_features(%s)" % os.path.relpath(f, p.raw))
        before = pd.read_csv(f, low_memory=False)
        if 'Rolling_UDP_Sum' in before.columns and not cfg.force:
            log("already enriched (Rolling_UDP_Sum present); skipping - it is not")
            log("idempotent, the second run would fail on the one-hot columns.")
            continue
        additional_features(f, window_size=cfg.window_size)
        after = pd.read_csv(f, low_memory=False)
        added = [c for c in after.columns if c not in before.columns]
        removed = [c for c in before.columns if c not in after.columns]
        log("shape %s -> %s" % (before.shape, after.shape))
        log("added  (%d): %s" % (len(added), added))
        log("removed(%d): %s   (consumed by get_dummies / helper cols)" % (len(removed), removed))
    state_mark(p, "features")


# ---------------------------------------------------------------------------
# stage 4: Combining_classes + the notebook's final concat / drop cells
# ---------------------------------------------------------------------------
def stage_combine(cfg, p):
    step("STAGE 4/6  combine: per-class files -> one train + one test csv")
    import pandas as pd
    from Utility.Functions import Combining_classes
    where(Combining_classes)

    csvs = sorted(glob.glob(os.path.join(p.raw, "*.csv")))
    if not csvs:
        sys.exit("no csv in %s - run the earlier stages first" % p.raw)
    classes = sorted({os.path.basename(c).split("-")[0] for c in csvs})
    log("classes found from file names: %s" % classes)
    log("labels: %s" % {c: LABEL_DICT.get(c, "?") for c in classes})

    missing = [c for c in classes if c not in LABEL_DICT]
    if missing:
        sys.exit("class %s not in LABEL_DICT - rename the pcap per NAME_MAPPING" % missing)

    substep("Combining_classes(): dedup, cap per class, add the Label column")
    Combining_classes(p.raw + os.sep, classes,
                      Number_in_individaul_class=cfg.train_cap,
                      Number_of_test_samples=cfg.test_sample,
                      label_dict=dict(LABEL_DICT))
    for d in (p.raw_train, p.raw_test):
        log("%s: %s" % (os.path.relpath(d, p.work),
                        [os.path.basename(f) for f in sorted(glob.glob(os.path.join(d, "*")))]))

    # Data_preprocessing_CIC-IoT2023.ipynb cells 11-14, minus the os.remove() calls
    # so the intermediates stay around for a second look.
    for sub, out_csv in ((p.raw_train, p.train_csv), (p.raw_test, p.test_csv)):
        substep("concat %s/* -> %s (notebook cells 11-14)"
                % (os.path.relpath(sub, p.work), os.path.basename(out_csv)))
        files = sorted(glob.glob(os.path.join(sub, "*.csv")))
        if not files:
            log("nothing in %s, skipping" % sub)
            continue
        parts = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            log("  %-34s shape=%s label=%s" % (os.path.basename(f), df.shape,
                                               sorted(df['Label'].unique()) if 'Label' in df else "-"))
            parts.append(df)
        final_df = pd.concat(parts, ignore_index=True)
        present = [c for c in DROP_COLUMNS if c in final_df.columns]
        absent = [c for c in DROP_COLUMNS if c not in final_df.columns]
        final_df.drop(present, axis=1, inplace=True)
        log("dropped %d biased/identifier columns%s"
            % (len(present), (", %d were already gone: %s" % (len(absent), absent)) if absent else ""))
        obj_cols = [c for c in final_df.columns
                    if final_df[c].dtype == object and not c.startswith('udps.')]
        if obj_cols:
            log("WARNING non-numeric columns left, NIDSDataset will fail on them: %s" % obj_cols)
        final_df.to_csv(out_csv, index=False)
        log("wrote %s  shape=%s  label counts=%s"
            % (out_csv, final_df.shape, final_df['Label'].value_counts().to_dict()))


# ---------------------------------------------------------------------------
# stage 5: CSV -> HeteroData graph objects
# ---------------------------------------------------------------------------
def _truncate(src, dst, n):
    import pandas as pd
    df = pd.read_csv(src, low_memory=False)
    if n and df.shape[0] > n:
        df = df.groupby('Label', group_keys=False).head(max(1, n // max(1, df['Label'].nunique())))
        df.to_csv(dst, index=False)
        log("%s: %d rows -> %s (%d rows, --max-graphs)"
            % (os.path.basename(src), pd.read_csv(src, low_memory=False).shape[0],
               os.path.basename(dst), df.shape[0]))
        return dst
    return src


def stage_graphs(cfg, p):
    step("STAGE 5/6  graphs: each flow row -> one heterogeneous graph  (NIDSDataset.process)")
    from Utility.Functions import NIDSDataset
    where(NIDSDataset.process)
    log("flow node   = the numeric flow features of the row")
    log("packet node = 1500 payload bytes per packet (<=--packet-limit packets)")
    log("contain edge= flow -> every packet, attrs [direction, ip_size, transport_size, payload_size]")
    log("link edge   = packet i -> packet i+1, attr [delta_time]; then T.ToUndirected()")

    if not os.path.exists(p.train_csv):
        sys.exit("%s missing - run --stage combine first" % p.train_csv)

    train_csv = _truncate(p.train_csv, p.train_csv_dbg, cfg.max_graphs)
    test_csv = _truncate(p.test_csv, p.test_csv_dbg, max(1, cfg.max_graphs // 4)) \
        if os.path.exists(p.test_csv) else None

    if os.path.isdir(p.processed):
        old = glob.glob(os.path.join(p.processed, "*.pt"))
        if old:
            log("clearing %d old .pt files in %s (indices restart at 0)" % (len(old), p.processed))
            for f in old:
                os.remove(f)

    substep("NIDSDataset(train, single_file=True, test=False)")
    train_ds = NIDSDataset(root=p.work, label_dict=LABEL_DICT, filename=[train_csv],
                           skip_processing=False, test=False, single_file=True)
    log("train graphs: %d" % len(train_ds))

    if test_csv:
        substep("NIDSDataset(test, single_file=True, test=True)")
        test_ds = NIDSDataset(root=p.work, label_dict=LABEL_DICT, filename=[test_csv],
                              skip_processing=False, test=True, single_file=True)
        log("test graphs : %d" % len(test_ds))

    substep("what one graph object looks like")
    g = train_ds[0]
    log("node types  : %s" % g.node_types)
    log("edge types  : %s" % g.edge_types)
    log("flow.x      : %s" % (tuple(g['flow'].x.shape),))
    log("packet.x    : %s" % (tuple(g['packet'].x.shape),))
    for et in g.edge_types:
        store = g[et]
        log("%-28s edge_index=%s edge_attr=%s"
            % (str(et), tuple(store.edge_index.shape),
               tuple(store.edge_attr.shape) if 'edge_attr' in store else None))
    log("y (label)   : %s" % g.y.tolist())


# ---------------------------------------------------------------------------
# stage 6: model forward / train / eval
# ---------------------------------------------------------------------------
def stage_model(cfg, p):
    step("STAGE 6/6  model: HeteroGNN_Edge forward + %d training epoch(s) + eval" % cfg.epochs)
    import torch
    from torch_geometric.loader import DataLoader
    from Utility.Functions import NIDSDataset
    from Utility.Model import HeteroGNN_Edge
    from Utility.Training import train_with_edge_Att, test_cm_with_edge_att
    where(HeteroGNN_Edge.forward)

    if not glob.glob(os.path.join(p.processed, "data_*.pt")):
        sys.exit("no graph objects in %s - run --stage graphs first" % p.processed)

    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    log("device=%s torch=%s" % (device, torch.__version__))

    substep("load the .pt graphs back (skip_processing=True)")
    train_ds = NIDSDataset(root=p.work, label_dict=LABEL_DICT, filename=[],
                           skip_processing=True, test=False, single_file=True)
    log("train graphs=%d" % len(train_ds))
    test_ds = None
    if glob.glob(os.path.join(p.processed, "data_test_*.pt")):
        test_ds = NIDSDataset(root=p.work, label_dict=LABEL_DICT, filename=[],
                              skip_processing=True, test=True, single_file=True)
        log("test graphs =%d" % len(test_ds))

    sample = train_ds[0].to(device)
    model_args = {"device": device, "hidden_size": cfg.hidden_size, "epochs": cfg.epochs,
                  "weight_decay": 1e-5, "lr": cfg.lr, "attn_size": 32, "eps": 1.0}

    substep("build HeteroGNN_Edge (2x GATConv over %s)" % (sample.edge_types,))
    model = HeteroGNN_Edge(sample, model_args, aggr="mean").to(device)

    substep("one forward pass (lazy GATConv shapes get materialised here)")
    init_loader = DataLoader(train_ds, batch_size=min(2, len(train_ds)), shuffle=False)
    batch = next(iter(init_loader)).to(device)
    log("batch: %s" % {k: tuple(v.shape) for k, v in batch.x_dict.items()})
    with torch.no_grad():
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
    log("model output (log-softmax) shape=%s" % (tuple(out.shape),))
    log("params=%d" % sum(x.numel() for x in model.parameters()))

    substep("train_with_edge_Att() for %d epoch(s), batch-size=%d" % (cfg.epochs, cfg.batch_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    train_with_edge_Att(train_loader, model, model_args, device, class_weight=None)

    if test_ds is not None and len(test_ds):
        substep("test_cm_with_edge_att()")
        acc, preds, labels = test_cm_with_edge_att(DataLoader(test_ds, batch_size=1, shuffle=False),
                                                  model, device)
        log("accuracy=%.4f on %d test graphs (only the classes of these pcaps are present,"
            % (acc, len(test_ds)))
        log("so the numbers are meaningless - this stage is here to be stepped through)")


STAGES = [("extract", stage_extract), ("split", stage_split), ("features", stage_features),
          ("combine", stage_combine), ("graphs", stage_graphs), ("model", stage_model)]
STAGE_NAMES = [n for n, _ in STAGES]


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["all"], choices=["all"] + STAGE_NAMES,
                    help="stages to run, in pipeline order (default: all)")
    ap.add_argument("--pcap-dir", default=DEFAULT_PCAP_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="working dir (default: <pcap-dir>/trace_out)")
    ap.add_argument("--reset", action="store_true", help="delete the working dir first")
    ap.add_argument("--force", action="store_true", help="redo stages even if outputs exist")

    ap.add_argument("--extract-mode", choices=["inproc", "streamer"], default="inproc",
                    help="inproc = run nfstream's meter in this process so breakpoints "
                         "hit My_Custom.on_init/on_update (default)")
    ap.add_argument("--max-flows", type=int, default=3000,
                    help="cap flows kept per pcap (0 = all; the whole pcap is metered "
                         "either way, this only limits what is written)")
    ap.add_argument("--packet-limit", type=int, default=20,
                    help="My_Custom(limit=): packets kept per flow (paper: 20)")
    ap.add_argument("--bpf", default=None, help="optional BPF filter, e.g. 'tcp port 80'")

    ap.add_argument("--test-sample", type=int, default=4000)
    ap.add_argument("--train-cap", type=int, default=20000)
    ap.add_argument("--window-size", type=int, default=350)

    ap.add_argument("--max-graphs", type=int, default=200,
                    help="cap rows turned into graph objects (0 = all)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])

    ap.add_argument("--trace", choices=["on", "off"], default="on",
                    help="write the call trace log (auto-off under the debugger)")
    ap.add_argument("--trace-deps", action="store_true",
                    help="also trace inside nfstream / torch_geometric")
    ap.add_argument("--trace-max-calls", type=int, default=3,
                    help="log this many calls per function, then only count")
    ap.add_argument("--trace-echo", action="store_true", help="also print the trace to stdout")
    ap.add_argument("--trace-returns", choices=["on", "off"], default="on")

    ap.add_argument("--break-at", default="off",
                    help="auto-break (không cần tự đặt breakpoint): 'off' | 'all' = mọi hàm "
                         "chính của các stage đang chạy | danh sách tên hàm cách nhau bởi dấu "
                         "phẩy, vd 'on_init,process,forward'")
    ap.add_argument("--break-max-hits", type=int, default=1,
                    help="mỗi hàm chỉ auto-break bao nhiêu lần đầu (mặc định 1)")
    ap.add_argument("--break-pause", choices=["off", "input", "pdb"], default="off",
                    help="khi KHÔNG có debugger: 'input' = dừng chờ bấm Enter mới chạy tiếp")
    return ap.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)
    cfg.out_dir = cfg.out_dir or os.path.join(cfg.pcap_dir, "trace_out")
    p = Paths(cfg.out_dir)
    if cfg.reset and os.path.isdir(p.work):
        shutil.rmtree(p.work)
    p.mk()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_log = os.path.join(p.logs, "run_%s.log" % stamp)
    trace_log = os.path.join(p.logs, "trace_%s.trace" % stamp)
    fh = open(run_log, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)

    wanted = STAGE_NAMES if "all" in cfg.stage else [n for n in STAGE_NAMES if n in cfg.stage]

    print("#" * 100)
    print("# XG-NID pipeline trace")
    print("#   pcap dir : %s" % cfg.pcap_dir)
    print("#   work dir : %s" % p.work)
    print("#   stages   : %s" % ", ".join(wanted))
    print("#   run log  : %s" % run_log)
    print("#   python   : %s" % sys.executable)
    print("#" * 100)

    if cfg.break_at != "off":
        from Debug.autobreak import install
        names = None if cfg.break_at == "all" else \
            [x.strip() for x in cfg.break_at.split(",") if x.strip()]
        installed, break_mode = install(stages=wanted, names=names,
                                        max_hits=cfg.break_max_hits,
                                        pause=cfg.break_pause,
                                        log=lambda m: print(m, flush=True))
        print("#   auto-break [%s]: %d hàm, tối đa %d lần/hàm"
              % (break_mode, len(installed), cfg.break_max_hits))
        for spec in installed:
            print("#      %s" % spec)
        if break_mode == "pydevd":
            print("#   -> debugger tự dừng BÊN TRONG từng hàm, F5 để sang hàm kế tiếp")
        elif break_mode == "wrapper":
            print("#   -> không có debugger: chỉ IN thứ tự hàm. Bấm F5 trong VS Code "
                  "(config có --break-at) để dừng thật, hoặc XGNID_BREAK_PDB=1 dùng pdb")

    tracer = None
    tfh = None
    if cfg.trace == "on":
        include = [PROJECT]
        if cfg.trace_deps:
            import site
            for pkg in ("nfstream", "torch_geometric"):
                for sp in site.getsitepackages():
                    cand = os.path.join(sp, pkg)
                    if os.path.isdir(cand):
                        include.append(cand)
        tfh = open(trace_log, "w", buffering=1)
        tracer = CallTracer(include_dirs=include, out=tfh, echo=cfg.trace_echo,
                            max_calls_per_func=cfg.trace_max_calls,
                            log_returns=cfg.trace_returns == "on")
        print("#   trace    : %s" % trace_log)
        tracer.start()
        globals()["_TRACER"] = tracer

    try:
        for name, fn in STAGES:
            if name in wanted:
                fn(cfg, p)
    finally:
        if tracer is not None:
            tracer.stop()
            summary = tracer.summary()
            tfh.write(summary + "\n")
            tfh.close()
            print(summary)
            print("\n[trace] full call log: %s" % trace_log)
        print("\n[done] %.1fs  run log: %s" % (time.time() - _T0, run_log))
        fh.close()


if __name__ == "__main__":
    main()
