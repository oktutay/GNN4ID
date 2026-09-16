#!/usr/bin/env python3
"""What a 350-flow rolling window means in seconds, and what the June-2026 order
(rolling AFTER undersampling / per split) did to the features.  Runs on raw NFStream
CSVs (e.g. <out>/raw/*.csv from run_preprocessing.py).

    python eda/eda_rolling_window.py --raw "../data/Debug and Trace/pp_out/raw" --out eda/tables
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from Utility.Additional_Features import compute_rolling_features  # noqa: E402
from Utility.Schema import TIME_COL  # noqa: E402

COLS = ["Rolling_TCP_Requests_Destination", "Rolling_packets_destination",
        "Rolling_Duration_Destination", "Unique_Ports_In_SourceDestinationIP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "tables"))
    ap.add_argument("--window", type=int, default=350)
    ap.add_argument("--thin", type=float, default=0.2, help="fraction kept when simulating the old order")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    span_rows, shift_rows = [], []
    for f in sorted(glob.glob(os.path.join(args.raw, "*.csv"))):
        stem = os.path.basename(f)[:-4]
        raw = pd.read_csv(f, low_memory=False).sort_values(TIME_COL).reset_index(drop=True)
        t = raw[TIME_COL].to_numpy(np.int64)
        dur_s = (t[-1] - t[0]) / 1000
        # seconds spanned by `window` consecutive flows, per destination
        spans = []
        for _, grp in raw.groupby("dst_ip"):
            tt = grp[TIME_COL].to_numpy(np.int64)
            if len(tt) > args.window:
                spans.extend(((tt[args.window:] - tt[:-args.window]) / 1000).tolist())
        spans = np.array(spans) if spans else np.array([np.nan])
        n_dst = raw["dst_ip"].nunique()
        top_dst_share = raw["dst_ip"].value_counts(normalize=True).iloc[0]
        span_rows.append({"file": stem, "flows": len(raw), "capture_seconds": round(dur_s, 1),
                          "flows_per_second": round(len(raw) / max(dur_s, 1e-3), 2), "n_dst_ip": n_dst,
                          "top_dst_share": round(top_dst_share, 3),
                          "dst_groups_reaching_window": int(sum(1 for _, g in raw.groupby("dst_ip") if len(g) > args.window)),
                          "window_span_s_p10": round(np.nanpercentile(spans, 10), 1),
                          "window_span_s_median": round(np.nanmedian(spans), 1),
                          "window_span_s_p90": round(np.nanpercentile(spans, 90), 1)})
        # dense (correct) vs thinned (old order) on the SAME rows
        dense = compute_rolling_features(raw, window=args.window, onehot=False).set_index("id")
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(raw), size=max(args.window + 1, int(len(raw) * args.thin)), replace=False))
        thin = compute_rolling_features(raw.iloc[keep], window=args.window, onehot=False).set_index("id")
        ids = thin.index
        for c in COLS:
            d, s = dense.loc[ids, c].to_numpy(float), thin[c].to_numpy(float)
            rel = np.abs(d - s) / np.maximum(np.abs(d), 1e-9)
            shift_rows.append({"file": stem, "column": c, "rows_compared": len(ids),
                               "share_different": round(float(np.mean(~np.isclose(d, s))), 3),
                               "median_rel_diff": round(float(np.median(rel)), 3),
                               "dense_share_eq_window": round(float(np.mean(d == args.window)), 3),
                               "thin_share_eq_window": round(float(np.mean(s == args.window)), 3),
                               "dense_median": float(np.median(d)), "thin_median": float(np.median(s))})
        # cold start: rolling on the last 20% alone vs on the whole file
        n_pool = int(round(len(raw) * 0.2)); tail = raw.iloc[len(raw) - n_pool:]
        alone = compute_rolling_features(tail, window=args.window, onehot=False).set_index("id")
        for c in COLS:
            d, s = dense.loc[alone.index, c].to_numpy(float), alone[c].to_numpy(float)
            shift_rows.append({"file": stem, "column": c + " (test slice alone vs whole file)", "rows_compared": len(alone),
                               "share_different": round(float(np.mean(~np.isclose(d, s))), 3),
                               "median_rel_diff": round(float(np.median(np.abs(d - s) / np.maximum(np.abs(d), 1e-9))), 3),
                               "dense_share_eq_window": round(float(np.mean(d == args.window)), 3),
                               "thin_share_eq_window": round(float(np.mean(s == args.window)), 3),
                               "dense_median": float(np.median(d)), "thin_median": float(np.median(s))})
    pd.DataFrame(span_rows).to_csv(os.path.join(args.out, "rolling_window_span.csv"), index=False)
    pd.DataFrame(shift_rows).to_csv(os.path.join(args.out, "rolling_order_shift.csv"), index=False)
    print(pd.DataFrame(span_rows).to_string()); print(pd.DataFrame(shift_rows).to_string())


if __name__ == "__main__":
    main()
