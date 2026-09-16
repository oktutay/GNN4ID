"""Unit tests for Utility.Additional_Features (window units, count modes, schema)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from Utility.Additional_Features import (  # noqa: E402
    compute_rolling_features, additional_features, ROLLING_SPEC, _window_bounds, _group_order)
from Utility.Schema import ROLLING_28, TABLE1_16, ONEHOT_7, TIME_COL  # noqa: E402


def synthetic(n=2000, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 600_000, size=n))          # 10 minutes of ms timestamps
    if ties:
        t[rng.choice(n, size=40, replace=False)] = t[rng.choice(n, size=40, replace=False)]
    else:
        t = np.sort(rng.choice(600_000, size=n, replace=False))
    t = np.sort(t)
    src = rng.choice(["10.0.0.%d" % i for i in range(5)], size=n)
    dst = rng.choice(["192.168.1.%d" % i for i in range(3)], size=n)
    proto = rng.choice([1, 6, 17], size=n, p=[0.1, 0.6, 0.3])
    pk = rng.integers(1, 21, size=n)
    df = pd.DataFrame({
        "id": np.arange(n), TIME_COL: t, "src_ip": src, "dst_ip": dst,
        "src_port": rng.integers(1024, 65535, size=n), "dst_port": rng.choice([80, 443, 53, 22, 8888], size=n),
        "protocol": proto, "expiration_id": rng.choice([0, -1], size=n),
        "bidirectional_packets": pk, "src2dst_packets": np.maximum(1, pk // 2),
        "bidirectional_duration_ms": rng.integers(0, 5000, size=n),
        "bidirectional_ack_packets": rng.integers(0, 10, size=n),
        "bidirectional_fin_packets": rng.integers(0, 2, size=n),
        "bidirectional_rst_packets": rng.integers(0, 2, size=n),
        "bidirectional_psh_packets": rng.integers(0, 5, size=n),
        "bidirectional_syn_packets": rng.integers(0, 2, size=n),
        "src2dst_min_ps": rng.integers(40, 100, size=n), "src2dst_max_ps": rng.integers(100, 1500, size=n),
        "dst2src_min_ps": rng.integers(40, 100, size=n), "dst2src_max_ps": rng.integers(100, 1500, size=n),
    })
    return df


class TestRollingUnits(unittest.TestCase):

    def setUp(self):
        self.df = synthetic()

    def test_flows_matches_authors_code(self):
        import _author_additional_features_551d1f1 as author
        import tempfile
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "x.csv")
        self.df.to_csv(p, index=False)
        author.additional_features(p, window_size=50)
        a = pd.read_csv(p).set_index("id").sort_index()
        b = compute_rolling_features(self.df, window=50).set_index("id").sort_index()
        self.assertEqual(list(a.columns), list(b.columns))
        t = self.df.set_index("id")[TIME_COL]
        for col, grp, _, how in ROLLING_SPEC:
            x, y = a[col].to_numpy(), b[col].to_numpy()
            ok = np.isclose(x, y) if how == "mean" else (x == y)
            # the authors' unstable sort may order tied timestamps differently
            bad = np.flatnonzero(~ok)
            tied = t.duplicated(keep=False).to_numpy()
            self.assertTrue(np.all(tied[bad]), "%s differs on untied rows: %s" % (col, bad[:5]))
        for c in ONEHOT_7:
            self.assertTrue((a[c] == b[c]).all(), c)

    def test_time_matches_pandas_offset_rolling(self):
        # tie-free data: pandas' offset rolling treats duplicate timestamps oddly
        # (rows tied with the window end can get an empty history); our engine
        # counts every earlier row in (t-W, t], which test_time_ties checks.
        df = synthetic(ties=False)
        b = compute_rolling_features(df, window="60s", window_unit="time", onehot=False)
        d = df.sort_values(TIME_COL).reset_index(drop=True)
        d["_t"] = pd.to_datetime(d[TIME_COL], unit="ms")
        d["is_udp"] = (d["protocol"] == 17).astype(float)
        ref = d.groupby("dst_ip", sort=False).rolling("60s", on="_t", min_periods=1)["is_udp"].sum()
        ref = ref.reset_index(level=0, drop=True).sort_index()
        got = b.set_index("id").loc[d["id"], "Rolling_UDP_Requests_Destination"].to_numpy()
        self.assertTrue(np.array_equal(ref.to_numpy(), got))
        ref_m = d.groupby("dst_ip", sort=False).rolling("60s", on="_t", min_periods=1)["bidirectional_duration_ms"].mean()
        ref_m = ref_m.reset_index(level=0, drop=True).sort_index()
        got_m = b.set_index("id").loc[d["id"], "Rolling_Duration_Destination"].to_numpy()
        self.assertTrue(np.allclose(ref_m.to_numpy(), got_m))

    def test_time_ties(self):
        b = compute_rolling_features(self.df, window="60s", window_unit="time", onehot=False)
        d = b.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)   # b is already time-sorted
        d["is_udp"] = (d["protocol"] == 17).astype(float)
        rng = np.random.default_rng(1)
        for i in rng.choice(len(d), size=60, replace=False):
            sub = d.iloc[: i + 1]
            sub = sub[(sub["dst_ip"] == d.loc[i, "dst_ip"]) & (sub[TIME_COL] > d.loc[i, TIME_COL] - 60000)]
            self.assertEqual(sub["is_udp"].sum(), d.loc[i, "Rolling_UDP_Requests_Destination"])

    def test_packets_window_is_minimal_suffix(self):
        N = 100
        gid = pd.factorize(self.df.sort_values(TIME_COL, kind="mergesort")["dst_ip"])[0]
        d = self.df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
        order, gstart, first = _group_order(gid)
        pk = d["bidirectional_packets"].to_numpy(float)[order]
        t = d[TIME_COL].to_numpy(np.int64)[order]
        start, end = _window_bounds(gstart, first, t, pk, "packets", N, True)
        for i in range(len(pk)):
            s = pk[start[i]:end[i]].sum()
            if start[i] == gstart[i]:
                self.assertTrue(s <= N or start[i] == i or pk[start[i] + 1:end[i]].sum() < N)
            else:
                self.assertGreaterEqual(s, N)
                self.assertLess(pk[start[i] + 1:end[i]].sum(), N)  # dropping the oldest breaks it
        b = compute_rolling_features(self.df, window=N, window_unit="packets", onehot=False)
        self.assertEqual(len(b), len(self.df))

    def test_count_mode_packets_weights_by_packets(self):
        a = compute_rolling_features(self.df, window=30, count_mode="author", onehot=False)
        b = compute_rolling_features(self.df, window=30, count_mode="packets", onehot=False)
        # ACK column is a packet count in both modes
        self.assertTrue(np.array_equal(a["Rolling_ACK_Packets_Destination"], b["Rolling_ACK_Packets_Destination"]))
        # UDP column counts flows in author mode, packets in packets mode
        self.assertTrue((b["Rolling_UDP_Requests_Destination"] >= a["Rolling_UDP_Requests_Destination"]).all())
        self.assertGreater(b["Rolling_UDP_Requests_Destination"].sum(), a["Rolling_UDP_Requests_Destination"].sum())

    def test_schema_order_and_table1(self):
        b = compute_rolling_features(self.df, window=30)
        extra = [c for c in b.columns if c not in self.df.columns]
        self.assertEqual(extra, ["packet_size_variation"] + ROLLING_28 + ONEHOT_7)
        t1 = compute_rolling_features(self.df, window=30, schema="table1")
        extra1 = [c for c in t1.columns if c not in self.df.columns and c not in ONEHOT_7]
        self.assertEqual(sorted(extra1), sorted(TABLE1_16))

    def test_include_current_false_zero_on_group_heads(self):
        b = compute_rolling_features(self.df, window=30, include_current=False, onehot=False)
        d = b.sort_values(TIME_COL, kind="mergesort")
        heads = ~d.duplicated(subset=["dst_ip"], keep="first")
        self.assertTrue((d.loc[heads, "Rolling_bipackets_destination"] == 0).all())

    def test_file_wrapper_guard_and_alias(self):
        import tempfile
        import warnings
        tmp = tempfile.mkdtemp()
        raw = os.path.join(tmp, "raw.csv")
        synthetic(ties=False).to_csv(raw, index=False)   # tie-free: re-sorting is order-independent
        out = additional_features(raw, window_size=30, out_file=os.path.join(tmp, "f", "feat.csv"))
        self.assertTrue(os.path.exists(out))
        self.assertEqual(pd.read_csv(raw).shape, self.df.shape)   # raw untouched
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out2 = additional_features(out, window=30, out_file=os.path.join(tmp, "f", "again.csv"))
            self.assertTrue(any("already" in str(x.message) for x in w))
        self.assertEqual(pd.read_csv(out).shape, pd.read_csv(out2).shape)
        # force recomputes and gives the same result
        out3 = additional_features(out, window=30, out_file=os.path.join(tmp, "f", "force.csv"), force=True)
        x, y = pd.read_csv(out), pd.read_csv(out3)
        self.assertEqual(list(x.columns), list(y.columns))
        self.assertTrue(np.allclose(x["Rolling_TCP_Requests_Destination"], y["Rolling_TCP_Requests_Destination"]))

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            compute_rolling_features(self.df, window="60s", window_unit="flows")
        with self.assertRaises(ValueError):
            compute_rolling_features(self.df, window=10, window_unit="nope")


if __name__ == "__main__":
    unittest.main()
