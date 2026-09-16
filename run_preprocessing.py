#!/usr/bin/env python3
"""End-to-end, leak-free, author-compatible CIC-IoT2023 preprocessing for GNN4ID / XG-NID.

    pcap -> raw CSV -> rolling features -> temporal split -> per-class combine
         -> df_class_8_{train,test}.csv  (-> graphs)

Stage order and why it is leak-free (see Utility/Additional_Features.py docstring):

  1 extract   NFStream, 20 packets/flow, idle 120 s           -> <out>/raw/<Class>-<Short>_<n>.csv
  2 features  rolling features on the WHOLE time-ordered file  -> <out>/features/<stem>.csv
              (label-free, backward-looking; identical semantics for train and test)
  3 split     attacker-MAC filter, latest 20% = test pool,     -> <out>/split/{train,test}/<stem>_*.csv
              caps per file, feature-level dedup of the train pool
  4 combine   per class: dedup, proportional per-sub-attack    -> <out>/combined/{train,test}/<Class>_*.csv
              caps, GLOBAL test-vs-train removal, train-only         <out>/class_weights.json
              oversampling (paper Table 4)
  5 class8    drop 29 identifier columns, assert 97-col header -> <out>/df_class_8_{train,test}.csv
  6 graphs    (optional) NIDSDataset -> <out>/processed/*.pt

Nothing is written in place; every stage skips outputs that already exist
(use --force to redo, --reset to wipe <out>).  A manifest.json with per-file /
per-class counts is kept up to date after each stage.

Examples
--------
  # authors' configuration (350-flow windows, 82 flow features, oversample train to 20k)
  python run_preprocessing.py --pcap-dir /data/CIC_IoT2023/PCAP --out-dir /data/xgnid_pp

  # class-weight variant instead of oversampling; time window of 60 s
  python run_preprocessing.py --pcap-dir ... --out-dir ... --no-oversample --window 60s --window-unit time

  # start from CSVs already produced by the extractor
  python run_preprocessing.py --csv-dir /data/raw_csv --out-dir /data/xgnid_pp --stages features,split,combine,class8
"""
import argparse
import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from Utility.Functions import (  # noqa: E402
    resolve_class_subtype, canonical_stem, assert_all_resolvable, split_csv,
    Combining_classes, build_class8_csvs, mac_filter, NIDSDataset)
from Utility.Additional_Features import additional_features  # noqa: E402
from Utility.Schema import DEFAULT_LABELS, LABEL_ALIASES  # noqa: E402

STAGES = ["extract", "features", "split", "combine", "class8", "graphs"]
EXTRACTOR = os.path.join(HERE, "Utility", "Feature_extractor_flow_packet_combined.py")
log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
class Paths:
    def __init__(self, out_dir):
        self.out = out_dir
        self.raw = os.path.join(out_dir, "raw")
        self.filtered = os.path.join(out_dir, "filtered")
        self.features = os.path.join(out_dir, "features")
        self.split = os.path.join(out_dir, "split")
        self.combined = os.path.join(out_dir, "combined")
        self.logs = os.path.join(out_dir, "logs")
        self.processed = os.path.join(out_dir, "processed")
        self.manifest = os.path.join(out_dir, "manifest.json")
        self.train_csv = os.path.join(out_dir, "df_class_8_train.csv")
        self.test_csv = os.path.join(out_dir, "df_class_8_test.csv")


def load_manifest(p):
    try:
        with open(p.manifest) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_manifest(p, m):
    with open(p.manifest, "w") as fh:
        json.dump(m, fh, indent=1, default=str)


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("pandas", "numpy", "nfstream", "torch", "torch_geometric"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _wanted_class(path, classes):
    if not classes:
        return True
    return resolve_class_subtype(path).class_code in classes


def _list_csvs(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*.csv")))


# ---------------------------------------------------------------------------
# stage 1: extract
# ---------------------------------------------------------------------------
def _extract_one(pcap, raw_dir, stem, args):
    cmd = [sys.executable, EXTRACTOR, pcap, raw_dir, "--n-dissections", str(args.n_dissections),
           "--limit", str(args.packet_limit), "--idle-timeout", str(args.idle_timeout),
           "--out-name", stem]
    if args.bpf:
        cmd += ["--bpf", args.bpf]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("extractor failed on %s:\n%s" % (pcap, res.stderr[-2000:]))
    return time.time() - t0


def stage_extract(args, p, manifest):
    pcaps = sorted(glob.glob(os.path.join(args.pcap_dir, "**", "*.pcap*"), recursive=True))
    pcaps = [f for f in pcaps if os.path.isfile(f)]
    if not pcaps:
        sys.exit("no pcap under %s" % args.pcap_dir)
    assert_all_resolvable(pcaps)
    pcaps = [f for f in pcaps if _wanted_class(f, args.classes)]
    os.makedirs(p.raw, exist_ok=True)
    man = manifest.setdefault("extract", {})
    log.info("[extract] %d pcap(s) -> %s", len(pcaps), p.raw)
    for pcap in pcaps:
        stem = canonical_stem(pcap)
        out_csv = os.path.join(p.raw, stem + ".csv")
        if os.path.exists(out_csv) and not args.force:
            log.info("[extract] exists, skip: %s", os.path.basename(out_csv))
            continue
        log.info("[extract] %s -> %s.csv", os.path.basename(pcap), stem)
        secs = _extract_one(pcap, p.raw, stem, args)
        import pandas as pd
        n_rows = sum(1 for _ in open(out_csv)) - 1
        n_cols = len(pd.read_csv(out_csv, nrows=1).columns)
        man[stem] = {"pcap": os.path.basename(pcap), "n_flows": n_rows, "n_cols": n_cols,
                     "seconds": round(secs, 1)}
        log.info("[extract]   %d flows x %d cols in %.1fs", n_rows, n_cols, secs)
        if args.delete_pcap:
            os.remove(pcap)
        save_manifest(p, manifest)


# ---------------------------------------------------------------------------
# stage 2: features
# ---------------------------------------------------------------------------
def _features_one(src, dst, args):
    import pandas as pd
    if args.rolling_scope == "filtered":
        df = pd.read_csv(src, low_memory=False)
        info = resolve_class_subtype(src)
        df = mac_filter(df, is_benign=(info.class_code == "Benign"))
        fsrc = os.path.join(os.path.dirname(dst), "..", "filtered", os.path.basename(src))
        os.makedirs(os.path.dirname(os.path.abspath(fsrc)), exist_ok=True)
        df.to_csv(fsrc, index=False)
        src = fsrc
    out = additional_features(src, window=args.window, window_unit=args.window_unit,
                              count_mode=args.count_mode, schema=args.schema, out_file=dst,
                              include_current=not args.no_include_current, force=args.force)
    if not out:
        raise RuntimeError("additional_features failed on %s" % src)
    df = pd.read_csv(out, nrows=1)
    n_rows = sum(1 for _ in open(out)) - 1
    return {"n_rows": n_rows, "n_cols": len(df.columns), "window": args.window,
            "window_unit": args.window_unit, "count_mode": args.count_mode, "schema": args.schema,
            "rolling_scope": args.rolling_scope}


def stage_features(args, p, manifest):
    raw_dir = args.csv_dir or p.raw
    csvs = [f for f in _list_csvs(raw_dir) if _wanted_class(f, args.classes)]
    if not csvs:
        sys.exit("no raw csv in %s (run the extract stage or pass --csv-dir)" % raw_dir)
    assert_all_resolvable(csvs)
    os.makedirs(p.features, exist_ok=True)
    man = manifest.setdefault("features", {})
    jobs = []
    for src in csvs:
        stem = canonical_stem(src)
        dst = os.path.join(p.features, stem + ".csv")
        if os.path.exists(dst) and not args.force:
            log.info("[features] exists, skip: %s", os.path.basename(dst))
            continue
        jobs.append((src, dst, stem))
    log.info("[features] %d file(s), window=%s %s, count_mode=%s, schema=%s, scope=%s",
             len(jobs), args.window, args.window_unit, args.count_mode, args.schema, args.rolling_scope)
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_features_one, s, d, args): stem for s, d, stem in jobs}
            for fut, stem in futs.items():
                man[stem] = fut.result()
                log.info("[features] %s: %s", stem, man[stem])
                save_manifest(p, manifest)
    else:
        for s, d, stem in jobs:
            man[stem] = _features_one(s, d, args)
            log.info("[features] %s: %s", stem, man[stem])
            save_manifest(p, manifest)


# ---------------------------------------------------------------------------
# stage 3: split
# ---------------------------------------------------------------------------
def _split_one(src, args, split_dir):
    return split_csv(src, out_dir=split_dir, test_frac=args.test_frac, test_cap=args.test_cap,
                     train_cap=args.train_pool_cap_per_file, temporal=not args.random_split,
                     test_pick=args.test_pick, dedup="feature" if not args.no_dedup else "none",
                     mac_filter_on=(args.rolling_scope != "filtered"), seed=args.seed)


def stage_split(args, p, manifest):
    csvs = [f for f in _list_csvs(p.features) if _wanted_class(f, args.classes)]
    if not csvs:
        sys.exit("no feature csv in %s (run the features stage first)" % p.features)
    man = manifest.setdefault("split", {})
    jobs = []
    for src in csvs:
        stem = os.path.basename(src).rsplit(".", 1)[0]
        test_out = os.path.join(p.split, "test", stem + "_test.csv")
        if os.path.exists(test_out) and not args.force:
            log.info("[split] exists, skip: %s", os.path.basename(test_out))
            continue
        jobs.append((src, stem))
    log.info("[split] %d file(s): %s split, test_frac=%.2f, test_cap=%d, train_pool_cap_per_file=%d",
             len(jobs), "random" if args.random_split else "temporal", args.test_frac,
             args.test_cap, args.train_pool_cap_per_file)
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_split_one, s, args, p.split): stem for s, stem in jobs}
            for fut, stem in futs.items():
                info = fut.result()
                man[stem] = {k: v for k, v in info.items() if not k.endswith("_path")}
                log.info("[split] %s: %s", stem, man[stem])
                save_manifest(p, manifest)
    else:
        for s, stem in jobs:
            info = _split_one(s, args, p.split)
            man[stem] = {k: v for k, v in info.items() if not k.endswith("_path")}
            log.info("[split] %s: %s", stem, man[stem])
            save_manifest(p, manifest)


# ---------------------------------------------------------------------------
# stage 4: combine
# ---------------------------------------------------------------------------
def stage_combine(args, p, manifest):
    if not glob.glob(os.path.join(p.split, "train", "*_train.csv")):
        sys.exit("no split files in %s (run the split stage first)" % p.split)
    if os.path.isdir(p.combined) and (args.force or args.reset):
        shutil.rmtree(p.combined)
    log.info("[combine] train_cap=%d test_cap=%d oversample=%s classes=%s",
             args.train_cap, args.test_cap, not args.no_oversample, args.classes or "all")
    rep = Combining_classes(p.split, classes_list=args.classes or None,
                            Number_in_individaul_class=args.train_cap,
                            Number_of_test_samples=args.test_cap, label_dict=dict(DEFAULT_LABELS),
                            oversample=not args.no_oversample, dedup=not args.no_dedup,
                            dedup_test_within=args.dedup_test_within, seed=args.seed,
                            out_dir=p.combined)
    manifest["combine"] = rep
    cw = os.path.join(p.combined, "class_weights.json")
    if os.path.exists(cw):
        shutil.copy(cw, os.path.join(p.out, "class_weights.json"))
    save_manifest(p, manifest)


# ---------------------------------------------------------------------------
# stage 5: class8
# ---------------------------------------------------------------------------
def stage_class8(args, p, manifest):
    strict = (args.schema == "author82") and not args.keep_l7
    res = build_class8_csvs(p.combined, p.out, keep_l7=args.keep_l7, strict_header=strict)
    if not strict:
        log.warning("[class8] header check relaxed (schema=%s keep_l7=%s): the flow node "
                    "dimension differs from the authors' 82, Drive checkpoints do not apply",
                    args.schema, args.keep_l7)
    manifest["class8"] = {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in res.items()}
    # cross checks
    checks = {}
    try:
        import pandas as pd
        from Utility.Functions import row_keys, feature_view_columns
        tr = pd.read_csv(p.train_csv, low_memory=False)
        te = pd.read_csv(p.test_csv, low_memory=False)
        cols = [c for c in feature_view_columns(tr) if c in te.columns]
        tk = set(row_keys(tr, cols).tolist())
        ek = row_keys(te, cols)
        checks["test_keys_in_train"] = int(sum(k in tk for k in ek.tolist()))
        comb = manifest.get("combine", {})
        checks["train_rows_match_combine"] = (len(tr) == sum(v["train"]["oversampled"] for v in comb.values())) if comb else None
        checks["test_rows_match_combine"] = (len(te) == sum(v["test"]["capped"] for v in comb.values())) if comb else None
    except Exception as exc:  # pragma: no cover
        checks["error"] = str(exc)
    manifest["checks"] = checks
    log.info("[class8] checks: %s", checks)
    save_manifest(p, manifest)


# ---------------------------------------------------------------------------
# stage 6: graphs
# ---------------------------------------------------------------------------
def stage_graphs(args, p, manifest):
    if not os.path.exists(p.train_csv):
        sys.exit("%s missing (run the class8 stage first)" % p.train_csv)
    if os.path.isdir(p.processed) and args.force:
        for f in glob.glob(os.path.join(p.processed, "*.pt")):
            os.remove(f)
    labels = dict(DEFAULT_LABELS)
    log.info("[graphs] building train graphs (include_packetflag=%s)", args.include_packetflag)
    tr = NIDSDataset(root=p.out, label_dict=labels, filename=[p.train_csv], skip_processing=False,
                     test=False, single_file=True, include_packetflag=args.include_packetflag)
    te = NIDSDataset(root=p.out, label_dict=labels, filename=[p.test_csv], skip_processing=False,
                     test=True, single_file=True, include_packetflag=args.include_packetflag)
    g = tr[0]
    manifest["graphs"] = {"train": len(tr), "test": len(te),
                          "flow_x": list(g["flow"].x.shape), "packet_x": list(g["packet"].x.shape)}
    log.info("[graphs] %s", manifest["graphs"])
    save_manifest(p, manifest)


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap-dir", help="directory with CIC-IoT2023 pcaps (searched recursively)")
    src.add_argument("--csv-dir", help="directory with raw NFStream CSVs (skips the extract stage)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stages", default="extract,features,split,combine,class8",
                    help="comma list from %s (default: all but graphs)" % STAGES)
    ap.add_argument("--classes", default=None,
                    help="comma list of class codes to process, e.g. Benign,DDos,WebBased")
    # rolling features
    ap.add_argument("--window", default="350", help="350 (flows/packets) or e.g. 60s (time)")
    ap.add_argument("--window-unit", default="flows", choices=["flows", "time", "packets"])
    ap.add_argument("--count-mode", default="author", choices=["author", "packets", "flows"])
    ap.add_argument("--schema", default="author82", choices=["author82", "table1"])
    ap.add_argument("--rolling-scope", default="all", choices=["all", "filtered"],
                    help="all = windows over every flow of the pcap (authors); "
                         "filtered = attacker-MAC filter before the windows")
    ap.add_argument("--no-include-current", action="store_true",
                    help="exclude the current flow from its own window (XMF-GNN style)")
    # split / combine
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--test-cap", type=int, default=4000)
    ap.add_argument("--train-cap", type=int, default=20000)
    ap.add_argument("--train-pool-cap-per-file", type=int, default=20000,
                    help="per-file train cap applied at split time (0 = none)")
    ap.add_argument("--test-pick", default="random", choices=["random", "tail"])
    ap.add_argument("--random-split", action="store_true",
                    help="row-level random split instead of temporal (to MEASURE the leak)")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--dedup-test-within", action="store_true")
    ap.add_argument("--no-oversample", action="store_true",
                    help="keep the true unique train counts; use class_weights.json in train.py")
    ap.add_argument("--seed", type=int, default=42)
    # extractor
    ap.add_argument("--n-dissections", type=int, default=0)
    ap.add_argument("--packet-limit", type=int, default=20)
    ap.add_argument("--idle-timeout", type=int, default=120)
    ap.add_argument("--bpf", default=None)
    ap.add_argument("--keep-l7", action="store_true", help="keep NFStream L7 columns in class-8 CSVs")
    ap.add_argument("--delete-pcap", action="store_true")
    ap.add_argument("--include-packetflag", action="store_true", help="graphs stage: 1508-d packet nodes")
    # misc
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reset", action="store_true", help="delete <out-dir> first")
    args = ap.parse_args(argv)
    if args.window_unit != "time":
        try:
            args.window = int(args.window)
        except ValueError:
            ap.error("--window must be an int for --window-unit %s" % args.window_unit)
    if args.classes:
        args.classes = [LABEL_ALIASES.get(c.strip(), c.strip()) for c in args.classes.split(",") if c.strip()]
        bad = [c for c in args.classes if c not in DEFAULT_LABELS]
        if bad:
            ap.error("unknown classes %s (known: %s)" % (bad, sorted(DEFAULT_LABELS)))
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in args.stages if s not in STAGES]
    if bad:
        ap.error("unknown stages %s" % bad)
    if args.csv_dir and "extract" in args.stages:
        args.stages = [s for s in args.stages if s != "extract"]
    return args


def main(argv=None):
    args = parse_args(argv)
    p = Paths(os.path.abspath(args.out_dir))
    if args.reset and os.path.isdir(p.out):
        shutil.rmtree(p.out)
    os.makedirs(p.logs, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(p.logs, "preprocess_%s.log" % stamp))])
    manifest = load_manifest(p)
    manifest["config"] = {**{k: v for k, v in vars(args).items()},
                          "git_sha": _git_sha(), "versions": _versions(), "started": stamp}
    save_manifest(p, manifest)
    log.info("out=%s stages=%s", p.out, args.stages)
    runners = {"extract": stage_extract, "features": stage_features, "split": stage_split,
               "combine": stage_combine, "class8": stage_class8, "graphs": stage_graphs}
    t0 = time.time()
    for st in STAGES:
        if st in args.stages:
            log.info("==== stage %s ====", st)
            runners[st](args, p, manifest)
    manifest["finished_seconds"] = round(time.time() - t0, 1)
    save_manifest(p, manifest)
    log.info("done in %.1fs; manifest: %s", time.time() - t0, p.manifest)


if __name__ == "__main__":
    main()
