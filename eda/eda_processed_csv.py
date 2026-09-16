#!/usr/bin/env python3
"""EDA on the authors' published class-8 CSVs (df_class_8_train.csv / df_class_8_test.csv):
per-class flow shape, protocol mix, payload states, class-name leakage inside payloads,
rolling-window saturation, duplicates, shortcut oracles, feature/label mutual information.

    python eda/eda_processed_csv.py --root ../data/CIC_IoT2023_Processed_Data --out eda/tables

Writes CSV tables into --out; every table is referenced from CIC_IoT2023_EDA_v2_VI.md.
"""
import argparse
import collections
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from Utility.Functions import row_keys, feature_view_columns  # noqa: E402
from Utility.Schema import ROLLING_28, FLOW_FEATURE_NAMES_82  # noqa: E402

LAB = {0: "Benign", 1: "WebBased", 2: "Spoofing", 3: "Recon", 4: "Mirai", 5: "Dos", 6: "DDos", 7: "BruteForce"}
KW = [b"dvwa", b"vulnerabilities", b"sqli", b"xss", b"exec", b"upload", b"backdoor", b"brute", b"hijack",
      b"Host: 192.168", b"192.168.", b"GET /", b"POST /", b"HTTP/1", b"User-Agent", b"python-requests",
      b"nmap", b"Nmap", b"SSH-", b"<script", b"union", b"select ", b"cmd=", b"ip=", b"admin", b"login"]


def parse_list(s):
    return [h.strip().strip("'") for h in s.strip("[]").split(",")]


def payload_state(b):
    if len(b) == 0:
        return "empty"
    if len(b) >= 3 and b[1] == 3 and b[2] in (0, 1, 2, 3, 4):
        if b[0] in (0x14, 0x15, 0x16):
            return "tls_control"
        if b[0] == 0x17:
            return "tls_appdata"
    printable = sum(32 <= c < 127 or c in (9, 10, 13) for c in b) / len(b)
    return "printable" if printable > 0.9 else "binary"


def analyse(df, split, out):
    df = df.copy()
    df["npk"] = df["udps.packet_direction"].str.count(",") + 1
    df["has_payload"] = df["udps.payload_size"].map(lambda s: sum(int(x) > 0 for x in parse_list(s)))
    g = df.groupby("Label")
    ov = pd.DataFrame({
        "class": [LAB[k] for k in g.size().index], "flows": g.size().values,
        "pk_median": g["npk"].median().values, "pk_mean": g["npk"].mean().round(2).values,
        "pk_eq_20": g["npk"].apply(lambda s: (s == 20).mean()).round(3).values,
        "pk_le_2": g["npk"].apply(lambda s: (s <= 2).mean()).round(3).values,
        "exp_minus1(cut@20)": g["Exp_-1"].mean().round(3).values,
        "tcp": g["proto_6"].mean().round(3).values, "udp": g["proto_17"].mean().round(3).values,
        "icmp": g["proto_1"].mean().round(3).values,
        "payload_pk_per_flow": g["has_payload"].mean().round(2).values,
        "flows_with_payload": g["has_payload"].apply(lambda s: (s > 0).mean()).round(3).values,
        "src2dst_max_ps_median": g["src2dst_max_ps"].median().values,
        "bidir_mean_piat_ms_median": g["bidirectional_mean_piat_ms"].median().round(1).values,
    }, index=g.size().index)
    ov.insert(0, "split", split)
    ov.to_csv(os.path.join(out, f"class_overview_{split}.csv"), index=False)

    # rolling saturation
    rows = []
    for k, grp in g:
        for c in ROLLING_28:
            v = grp[c]
            rows.append({"split": split, "class": LAB[k], "column": c, "share_eq_350": round((v == 350).mean(), 3),
                         "share_eq_0": round((v == 0).mean(), 3), "median": v.median(), "nunique": v.nunique()})
    pd.DataFrame(rows).to_csv(os.path.join(out, f"rolling_saturation_{split}.csv"), index=False)

    # payload states + keywords + paths
    st_rows, kw_rows, paths, hosts = [], [], collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for k, grp in g:
        st = collections.Counter(); kwc = collections.Counter(); kwf = collections.Counter(); n_pk = 0
        flows_plain = 0
        for s in grp["udps.payload_data"]:
            hit_flow = set(); plain = False
            for h in parse_list(s):
                n_pk += 1
                if h == "00" or len(h) < 2:
                    st["empty"] += 1; continue
                try:
                    b = bytes.fromhex(h)
                except ValueError:
                    st["badhex"] += 1; continue
                state = payload_state(b); st[state] += 1
                if state == "printable":
                    plain = True
                    m = re.match(rb"(GET|POST|PUT|HEAD|OPTIONS) ([^ \r\n]{1,80})", b)
                    if m:
                        paths[k][(m.group(1) + b" " + m.group(2)).decode(errors="ignore")[:70]] += 1
                    mh = re.search(rb"Host: ([^\r\n]{1,60})", b)
                    if mh:
                        hosts[k][mh.group(1).decode(errors="ignore")] += 1
                for kw in KW:
                    if kw in b:
                        kwc[kw.decode()] += 1; hit_flow.add(kw.decode())
            for kw in hit_flow:
                kwf[kw] += 1
            flows_plain += plain
        tot = max(1, n_pk)
        st_rows.append({"split": split, "class": LAB[k], "flows": len(grp), "packets": n_pk,
                        **{f"{s}_pk": st.get(s, 0) for s in ("empty", "tls_control", "tls_appdata", "printable", "binary")},
                        **{f"{s}_share": round(st.get(s, 0) / tot, 3) for s in ("empty", "tls_control", "tls_appdata", "printable", "binary")},
                        "flows_with_printable_share": round(flows_plain / len(grp), 3)})
        for kw in KW:
            kw_rows.append({"split": split, "class": LAB[k], "keyword": kw.decode(), "packets": kwc.get(kw.decode(), 0),
                            "flows": kwf.get(kw.decode(), 0), "flow_share": round(kwf.get(kw.decode(), 0) / len(grp), 4)})
    pd.DataFrame(st_rows).to_csv(os.path.join(out, f"payload_states_{split}.csv"), index=False)
    pd.DataFrame(kw_rows).to_csv(os.path.join(out, f"payload_keywords_{split}.csv"), index=False)
    prow = [{"split": split, "class": LAB[k], "request_line": p, "packets": n}
            for k in paths for p, n in paths[k].most_common(15)]
    pd.DataFrame(prow).to_csv(os.path.join(out, f"payload_paths_{split}.csv"), index=False)
    hrow = [{"split": split, "class": LAB[k], "host": h, "packets": n} for k in hosts for h, n in hosts[k].most_common(8)]
    pd.DataFrame(hrow).to_csv(os.path.join(out, f"payload_hosts_{split}.csv"), index=False)

    # shortcut oracles (train-fit lookups evaluated on the same split; cross-split done in main)
    return df


def oracle(train, test, out):
    def keys(df, spec):
        if spec == "proto":
            return train_key(df, ["proto_1", "proto_6", "proto_17", "proto_2", "proto_58"])
        if spec == "proto+pk20+pk2":
            return train_key(df, ["proto_1", "proto_6", "proto_17"]) + "|" + (df["npk"] == 20).astype(str) + "|" + (df["npk"] <= 2).astype(str)
        if spec == "proto+npk":
            return train_key(df, ["proto_1", "proto_6", "proto_17"]) + "|" + df["npk"].astype(str)
        if spec == "proto+npk+exp":
            return train_key(df, ["proto_1", "proto_6", "proto_17"]) + "|" + df["npk"].astype(str) + "|" + df["Exp_-1"].astype(str)
        if spec == "maxps_bucket":
            return pd.cut(df["src2dst_max_ps"], [-1, 0, 40, 60, 100, 200, 500, 1000, 1e9]).astype(str)
        if spec == "proto+npk+maxps":
            return train_key(df, ["proto_1", "proto_6", "proto_17"]) + "|" + df["npk"].astype(str) + "|" + pd.cut(df["src2dst_max_ps"], [-1, 0, 40, 60, 100, 200, 500, 1000, 1e9]).astype(str)
        raise KeyError(spec)

    def train_key(df, cols):
        return df[cols].astype(int).astype(str).agg("".join, axis=1)

    rows = []
    for spec in ["proto", "proto+pk20+pk2", "proto+npk", "proto+npk+exp", "maxps_bucket", "proto+npk+maxps"]:
        ktr, kte = keys(train, spec), keys(test, spec)
        table = pd.crosstab(ktr, train["Label"])
        pred = table.idxmax(axis=1)
        purity_train = table.max(axis=1).sum() / len(train)
        acc_test = (kte.map(pred).fillna(-1).astype(int) == test["Label"]).mean()
        per_class = {LAB[c]: round(((kte.map(pred).fillna(-1).astype(int) == c) & (test["Label"] == c)).sum() / max(1, (test["Label"] == c).sum()), 3) for c in LAB}
        rows.append({"lookup": spec, "n_keys": len(table), "train_purity": round(purity_train, 3),
                     "test_accuracy": round(acc_test, 3), **{f"recall_{k}": v for k, v in per_class.items()}})
    pd.DataFrame(rows).to_csv(os.path.join(out, "shortcut_oracle.csv"), index=False)


def dedup_stats(train, test, out):
    cols = [c for c in feature_view_columns(train) if c in test.columns]
    tk, ek = row_keys(train, cols), row_keys(test, cols)
    tset = set(tk.tolist())
    rows = []
    for c in sorted(LAB):
        mtr, mte = (train["Label"] == c).values, (test["Label"] == c).values
        tr_dup = pd.Series(tk[mtr]).duplicated().mean() if mtr.any() else np.nan
        te_dup = pd.Series(ek[mte]).duplicated().mean() if mte.any() else np.nan
        leaked = np.mean([k in tset for k in ek[mte].tolist()]) if mte.any() else np.nan
        rows.append({"class": LAB[c], "train_rows": int(mtr.sum()), "train_unique": int(pd.Series(tk[mtr]).nunique()),
                     "train_dup_share": round(tr_dup, 3), "test_rows": int(mte.sum()),
                     "test_unique": int(pd.Series(ek[mte]).nunique()), "test_dup_share": round(te_dup, 3),
                     "test_in_train_share": round(leaked, 3)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "duplicates.csv"), index=False)


def feature_ami(train, out, n_bins=20):
    from sklearn.metrics import adjusted_mutual_info_score
    rows = []
    y = train["Label"].values
    for c in FLOW_FEATURE_NAMES_82:
        x = train[c].values.astype(float)
        nun = int(pd.Series(x).nunique())
        if nun <= 1:
            rows.append({"feature": c, "ami_8class": 0.0, "nunique": nun}); continue
        if nun <= n_bins:                      # low-cardinality (one-hot, small counts): use the values themselves
            xb = pd.factorize(x)[0].astype(float)
        else:
            try:
                xb = pd.qcut(x, n_bins, duplicates="drop", labels=False)
            except ValueError:
                xb = pd.cut(x, n_bins, labels=False)
            if pd.Series(xb).nunique() <= 1:   # quantiles collapsed (heavy zero mass): rank-bin the non-zero part
                xb = pd.cut(pd.Series(x).rank(method="dense"), n_bins, labels=False).to_numpy(float)
        rows.append({"feature": c, "ami_8class": round(adjusted_mutual_info_score(y, np.nan_to_num(np.asarray(xb, dtype=float), nan=-1)), 4),
                     "nunique": nun})
    pd.DataFrame(rows).sort_values("ami_8class", ascending=False).to_csv(os.path.join(out, "feature_ami.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(HERE), "..", "data", "CIC_IoT2023_Processed_Data"))
    ap.add_argument("--out", default=os.path.join(HERE, "tables"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train = pd.read_csv(os.path.join(args.root, "df_class_8_train.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(args.root, "df_class_8_test.csv"), low_memory=False)
    train.columns = [c.strip() for c in train.columns]; test.columns = [c.strip() for c in test.columns]
    print("train", train.shape, "test", test.shape)
    train = analyse(train, "train", args.out)
    test = analyse(test, "test", args.out)
    oracle(train, test, args.out)
    dedup_stats(train, test, args.out)
    feature_ami(train, args.out)
    print("tables written to", args.out)


if __name__ == "__main__":
    main()
