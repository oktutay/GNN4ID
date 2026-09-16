#!/usr/bin/env python3
"""End-to-end assertions on a run_preprocessing.py output directory.

    python Debug/verify_preprocessing.py --out-dir <out> [--drive-train df_class_8_train.csv]

Checks (each prints PASS/FAIL; exit code 1 if any FAIL):
 1 class-8 CSVs have the authors' 97-column header (== the Drive file when given)
 2 no non-numeric column outside udps.* in the class-8 CSVs
 3 no test feature row equals a train feature row (global)
 4 rolling invariance: split files carry exactly the values of features/<stem>.csv (join on id)
 5 author equivalence: the authors' original Additional_Features (551d1f1) on raw/<stem>.csv
   reproduces features/<stem>.csv (default flows/author82 config only)
 6 temporal boundary per file: max(train time) < min(test time)
 7 oversampling touched train only; row counts match the manifest
 8 manifest arithmetic
 9 resolver on hard CIC names
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "tests"))

from Utility.Functions import row_keys, feature_view_columns, resolve_class_subtype  # noqa: E402
from Utility.Schema import CLASS8_HEADER_97, ROLLING_28, TIME_COL, ONEHOT_7  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("%s  %s %s" % ("PASS" if ok else "FAIL", name, ("- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drive-train", default=None)
    ap.add_argument("--skip-author", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    man = json.load(open(os.path.join(out, "manifest.json")))
    cfg = man.get("config", {})
    train = pd.read_csv(os.path.join(out, "df_class_8_train.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(out, "df_class_8_test.csv"), low_memory=False)

    # 1 header
    hdr_tr = [c.strip() for c in train.columns]
    hdr_te = [c.strip() for c in test.columns]
    strict = cfg.get("schema", "author82") == "author82" and not cfg.get("keep_l7", False)
    if strict:
        check("1a header train == 97-col author layout", hdr_tr == CLASS8_HEADER_97,
              "n=%d" % len(hdr_tr))
        check("1b header test == 97-col author layout", hdr_te == CLASS8_HEADER_97)
        if args.drive_train:
            drive = [c.strip() for c in pd.read_csv(args.drive_train, nrows=0).columns]
            check("1c header == Drive CSV header", hdr_tr == drive,
                  "diff=%s" % [(a, b) for a, b in zip(hdr_tr, drive) if a != b][:5])
    else:
        check("1  header (relaxed: schema=%s keep_l7=%s)" % (cfg.get("schema"), cfg.get("keep_l7")),
              hdr_tr == hdr_te, "n=%d" % len(hdr_tr))

    # 2 dtypes
    bad = [c for c in train.columns if train[c].dtype == object and not c.startswith("udps.")]
    check("2  no non-numeric flow columns", not bad, str(bad))

    # 3 test disjoint from train
    cols = [c for c in feature_view_columns(train) if c in test.columns]
    tk = set(row_keys(train, cols).tolist())
    ek = row_keys(test, cols)
    n_leak = int(sum(k in tk for k in ek.tolist()))
    check("3  no test feature row in train", n_leak == 0, "%d leaked" % n_leak)

    # 4 rolling invariance + 6 temporal boundary
    feats = sorted(glob.glob(os.path.join(out, "features", "*.csv")))
    inv_ok, tb_ok, details = True, True, []
    for f in feats:
        stem = os.path.basename(f)[:-4]
        ff = pd.read_csv(f, low_memory=False).set_index("id")
        for part in ("train", "test"):
            sp = os.path.join(out, "split", part, "%s_%s.csv" % (stem, part))
            if not os.path.exists(sp):
                continue
            s = pd.read_csv(sp, low_memory=False).set_index("id")
            cols_r = [c for c in ROLLING_28 + ["packet_size_variation"] if c in s.columns]
            j = ff.loc[s.index, cols_r]
            same = np.allclose(j.to_numpy(float), s[cols_r].to_numpy(float), equal_nan=True)
            inv_ok &= same
            if not same:
                details.append("%s/%s" % (stem, part))
        tr_p = os.path.join(out, "split", "train", stem + "_train.csv")
        te_p = os.path.join(out, "split", "test", stem + "_test.csv")
        if os.path.exists(tr_p) and os.path.exists(te_p) and not cfg.get("random_split"):
            a = pd.read_csv(tr_p, usecols=[TIME_COL])
            b = pd.read_csv(te_p, usecols=[TIME_COL])
            if len(a) and len(b):
                tb_ok &= a[TIME_COL].max() < b[TIME_COL].min()
    check("4  split rows carry the file-level rolling values", inv_ok, ", ".join(details))
    check("6  temporal boundary per file (max train t < min test t)", tb_ok)

    # 5 author equivalence
    default_cfg = (cfg.get("window_unit") == "flows" and int(cfg.get("window", 350)) == 350
                   and cfg.get("count_mode") == "author" and cfg.get("schema") == "author82"
                   and cfg.get("rolling_scope") == "all")
    if default_cfg and not args.skip_author:
        import _author_additional_features_551d1f1 as author
        eq_ok, det = True, []
        for f in feats:
            stem = os.path.basename(f)[:-4]
            raw = os.path.join(out, "raw", stem + ".csv")
            if not os.path.exists(raw):
                continue
            tmp = tempfile.mkdtemp()
            try:
                p = os.path.join(tmp, "x.csv")
                shutil.copy(raw, p)
                author.additional_features(p)
                a = pd.read_csv(p, low_memory=False).set_index("id").sort_index()
                b = pd.read_csv(f, low_memory=False).set_index("id").sort_index()
                same_cols = list(a.columns) == list(b.columns)
                n_bad = 0
                for c in ROLLING_28 + ["packet_size_variation"] + ONEHOT_7:
                    x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                    n_bad += int((~np.isclose(x, y, equal_nan=True)).sum())
                eq_ok &= same_cols and n_bad == 0
                det.append("%s: cols_equal=%s mismatches=%d" % (stem, same_cols, n_bad))
            finally:
                shutil.rmtree(tmp)
        check("5  authors' 551d1f1 code reproduces features/", eq_ok, "; ".join(det))
    else:
        print("SKIP  5  author equivalence (non-default config or --skip-author)")

    # 7 / 8 manifest arithmetic
    comb = man.get("combine", {})
    if comb:
        n_tr = sum(v["train"]["oversampled"] for v in comb.values())
        n_te = sum(v["test"]["capped"] for v in comb.values())
        check("7a class-8 train rows == sum(combine.train.oversampled)", len(train) == n_tr,
              "%d vs %d" % (len(train), n_tr))
        check("7b class-8 test rows == sum(combine.test.capped)", len(test) == n_te,
              "%d vs %d" % (len(test), n_te))
        over = not cfg.get("no_oversample", False)
        tc = cfg.get("train_cap", 20000)
        ok = True
        for cls, v in comb.items():
            t = v["train"]
            ok &= t["unique"] <= t["pool"] and t["capped"] <= t["unique"]
            ok &= (t["oversampled"] == max(t["capped"], tc)) if over else (t["oversampled"] == t["capped"])
            te_ = v["test"]
            ok &= te_["capped"] <= min(te_["pool"] - te_["leaked_removed"], cfg.get("test_cap", 4000))
        check("7c oversample only raised train to the target; test never grew", ok)
        # unique train rows in class-8 == sum(capped)
        n_unique = int((~pd.Series(row_keys(train, cols)).duplicated()).sum())
        n_capped = sum(v["train"]["capped"] for v in comb.values())
        check("7d unique class-8 train rows == sum(combine.train.capped)", n_unique == n_capped,
              "%d vs %d" % (n_unique, n_capped))
    sp = man.get("split", {})
    if sp:
        ok = all(v["n_after_mac"] == v["n_test_pool"] + v["n_train_pool"] for v in sp.values())
        check("8  split: n_after_mac == test_pool + train_pool", ok)

    # 9 resolver
    names = {"DDoS-SlowLoris1.pcap": "DDos", "DDos-SlowLoris": "DDos", "BrowserHijacking.pcap": "WebBased",
             "Webbased-BrwserHijack_0.csv": "WebBased", "BenignTraffic3.pcap": "Benign",
             "Benign_Final": "Benign", "DDoS-ICMP_Fragmentation2": "DDos", "DDoS-ICMP_Flood10": "DDos",
             "DoS-UDP_Flood": "Dos", "WebBased-XSS_0_test.csv": "WebBased"}
    ok = all(resolve_class_subtype(n).class_code == c for n, c in names.items())
    check("9  resolver on hard CIC names", ok)

    print("\n%d check(s) failed" % len(FAILS) if FAILS else "\nALL CHECKS PASSED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
