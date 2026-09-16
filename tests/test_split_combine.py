"""End-to-end unit test of split_csv -> Combining_classes -> build_class8_csvs on
synthetic per-pcap CSVs (two classes, three sub-attacks, injected duplicates)."""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from Utility.Functions import (  # noqa: E402
    split_csv, Combining_classes, build_class8_csvs, row_keys, feature_view_columns,
    proportional_cap, oversample_train, _allocate, dedup_across_splits)
from Utility.Additional_Features import compute_rolling_features  # noqa: E402
from Utility.Schema import (  # noqa: E402
    CLASS8_HEADER_97, NFSTREAM_NUMERIC_46, UDPS_14, IDENTIFIER_DROP_29, TIME_COL,
    CIC_IOT2023_ATTACKER_MACS)

ATTACKER = sorted(CIC_IOT2023_ATTACKER_MACS)[0]


def make_raw(n, seed, attacker_share=0.6, dup_every=0):
    """A synthetic NFStream-like raw CSV with the 77 base columns + 14 udps lists."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 3_600_000, size=n))
    df = pd.DataFrame({"id": np.arange(n)})
    df["expiration_id"] = rng.choice([0, -1], size=n)
    df["src_ip"] = rng.choice(["10.0.0.%d" % i for i in range(4)], size=n)
    df["src_mac"] = np.where(rng.random(n) < attacker_share, ATTACKER, "00:11:22:33:44:55")
    df["src_oui"] = "x"
    df["src_port"] = rng.integers(1024, 65535, size=n)
    df["dst_ip"] = rng.choice(["192.168.1.%d" % i for i in range(3)], size=n)
    df["dst_mac"] = "66:77:88:99:aa:bb"
    df["dst_oui"] = "y"
    df["dst_port"] = rng.choice([80, 443, 53, 22], size=n)
    df["protocol"] = rng.choice([1, 6, 17], size=n)
    df["ip_version"] = 4
    df["vlan_id"] = 0
    df["tunnel_id"] = 0
    df[TIME_COL] = t
    df["bidirectional_last_seen_ms"] = t + 10
    df["bidirectional_duration_ms"] = 10
    df["bidirectional_packets"] = rng.integers(1, 21, size=n)
    df["bidirectional_bytes"] = rng.integers(60, 3000, size=n)
    for d in ("src2dst", "dst2src"):
        df[d + "_first_seen_ms"] = t
        df[d + "_last_seen_ms"] = t + 5
    for c in NFSTREAM_NUMERIC_46:
        df[c] = rng.integers(0, 50, size=n)
    for c in ["bidirectional_syn_packets", "bidirectional_cwr_packets", "bidirectional_ece_packets",
              "bidirectional_urg_packets", "bidirectional_ack_packets", "bidirectional_psh_packets",
              "bidirectional_rst_packets", "bidirectional_fin_packets"]:
        df[c] = rng.integers(0, 5, size=n)
    for c in UDPS_14:
        df[c] = ["['00', '00']"] * n
    if dup_every:   # inject exact feature duplicates (different id / timestamp)
        src_rows = df.iloc[::dup_every].copy()
        src_rows["id"] = np.arange(n, n + len(src_rows))
        src_rows[TIME_COL] = src_rows[TIME_COL] + 1_000_000   # later -> lands in the test pool
        df = pd.concat([df, src_rows], ignore_index=True)
    return df


class TestSplitCombine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.feat_dir = os.path.join(self.tmp, "features")
        os.makedirs(self.feat_dir)
        specs = [("WebBased-XSS_0.csv", 600, 1, 0), ("WebBased-SqlInject_0.csv", 300, 2, 0),
                 ("BruteForce-Dictionary_0.csv", 500, 3, 25), ("BenignTraffic_1.csv", 400, 4, 0)]
        for name, n, seed, dup in specs:
            raw = make_raw(n, seed)
            feat = compute_rolling_features(raw, window=50)
            if dup:
                # exact FEATURE-level duplicates (rolling values included) placed later in
                # time so they land in the test pool -> must be removed as leaked
                copies = feat.iloc[::dup].copy()
                copies["id"] = np.arange(len(feat), len(feat) + len(copies))
                copies[TIME_COL] = copies[TIME_COL] + 1_000_000
                feat = pd.concat([feat, copies], ignore_index=True)
            feat.to_csv(os.path.join(self.feat_dir, name), index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pipeline(self):
        split_dir = os.path.join(self.tmp, "split")
        infos = []
        for f in sorted(os.listdir(self.feat_dir)):
            infos.append(split_csv(os.path.join(self.feat_dir, f), out_dir=split_dir,
                                   test_cap=60, train_cap=250))
        for info in infos:
            tr = pd.read_csv(info["train_path"])
            te = pd.read_csv(info["test_path"])
            # strictly temporal boundary per file
            self.assertLess(tr[TIME_COL].max(), te[TIME_COL].min(), info["stem"])
            self.assertLessEqual(len(te), 60)
            self.assertLessEqual(len(tr), 250)
            self.assertEqual(info["n_after_mac"], info["n_test_pool"] + info["n_train_pool"])
            if info["class"] == "Benign":
                self.assertFalse((tr["src_mac"] == ATTACKER).any())
            else:
                self.assertTrue((tr["src_mac"] == ATTACKER).all())

        combined = os.path.join(self.tmp, "combined")
        rep = Combining_classes(split_dir, Number_in_individaul_class=400, Number_of_test_samples=50,
                                oversample=True, out_dir=combined)
        self.assertEqual(sorted(rep), ["Benign", "BruteForce", "WebBased"])
        # test never duplicated, capped, and disjoint from train at the feature level
        all_train_keys = set()
        for cls in rep:
            tr = pd.read_csv(os.path.join(combined, "train", cls + "_train.csv"))
            self.assertEqual(len(tr), 400)                          # oversampled to target
            self.assertEqual(rep[cls]["train"]["oversampled"], 400)
            self.assertLessEqual(rep[cls]["train"]["capped"], 400)
            all_train_keys |= set(row_keys(tr).tolist())
        for cls in rep:
            te = pd.read_csv(os.path.join(combined, "test", cls + "_test.csv"))
            self.assertLessEqual(len(te), 50)
            self.assertEqual(len(te), rep[cls]["test"]["capped"])
            ek = set(row_keys(te).tolist())
            self.assertEqual(len(ek & all_train_keys), 0, cls)
        # the injected duplicates were caught (BruteForce had dup rows in the test pool)
        self.assertGreater(rep["BruteForce"]["test"]["leaked_removed"], 0)
        # class weights are computed BEFORE oversampling
        with open(os.path.join(combined, "class_weights.json")) as fh:
            w = json.load(fh)
        self.assertEqual(len(w), 3)
        self.assertNotEqual(len(set(round(v, 6) for v in w.values())), 1)
        # sub-attack proportionality on WebBased train: two subtypes present
        self.assertEqual(sorted(rep["WebBased"]["subtypes"]), ["SqlInject", "XSS"])

        out = build_class8_csvs(combined, self.tmp)
        self.assertTrue(out["train"]["header_ok"] and out["test"]["header_ok"])
        train = pd.read_csv(os.path.join(self.tmp, "df_class_8_train.csv"))
        self.assertEqual([c.strip() for c in train.columns], CLASS8_HEADER_97)
        self.assertEqual(len(train), 1200)
        self.assertEqual(sorted(train["Label"].unique()), [0, 1, 7])

    def test_no_oversample_and_class_weights(self):
        split_dir = os.path.join(self.tmp, "split")
        for f in sorted(os.listdir(self.feat_dir)):
            split_csv(os.path.join(self.feat_dir, f), out_dir=split_dir, test_cap=60, train_cap=250)
        combined = os.path.join(self.tmp, "combined")
        rep = Combining_classes(split_dir, Number_in_individaul_class=400, Number_of_test_samples=50,
                                oversample=False, out_dir=combined)
        for cls in rep:
            self.assertEqual(rep[cls]["train"]["oversampled"], rep[cls]["train"]["capped"])

    def test_helpers(self):
        self.assertEqual(sum(_allocate({"a": 100, "b": 50, "c": 3}, 20).values()), 20)
        self.assertEqual(_allocate({"a": 5, "b": 5}, 20), {"a": 5, "b": 5})
        df = pd.DataFrame({"x": np.arange(100), "__subtype": ["a"] * 80 + ["b"] * 20})
        capped = proportional_cap(df, 10, rng=0)
        self.assertEqual(len(capped), 10)
        self.assertEqual(capped["__subtype"].value_counts().to_dict(), {"a": 8, "b": 2})
        over = oversample_train(df, 150, rng=0)
        self.assertEqual(len(over), 150)
        self.assertEqual(set(df["x"]) <= set(over["x"]), True)
        tr = pd.DataFrame({"f": [1, 1, 2, 3], "g": ["a", "a", "b", "c"]})
        te = pd.DataFrame({"f": [1, 4, 4], "g": ["a", "d", "d"]})
        tu, tc, st = dedup_across_splits(tr, te, feat_cols=["f", "g"], dedup_test_within=True)
        self.assertEqual(len(tu), 3)
        self.assertEqual(len(tc), 1)
        self.assertEqual(st["test_leaked_removed"], 1)


if __name__ == "__main__":
    unittest.main()
