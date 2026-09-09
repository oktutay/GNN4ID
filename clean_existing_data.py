"""Salvage the already-combined CIC-IoT2023 CSVs by removing the two defects we
can still fix without the (now-missing) raw per-subtype files:

  Defect A -- naive duplicate oversampling (Functions.duplicate_rows). The train
              set was inflated to 20k/class by copying rows verbatim, so several
              classes are mostly exact duplicates (BruteForce ~91%, WebBased
              ~78%). We collapse the train set back to its UNIQUE flows.

  Defect B -- train/test content leakage. Because the split was a random row
              split over a dataset full of byte-identical flows, ~13% of test
              rows (up to ~46% for DoS) are byte-identical to a train row. Those
              rows cannot be evaluated fairly, so we drop them from the test set.

What this script CANNOT fix (needs the raw NFStream CSVs that are no longer in
the repo -- see report):
  * group/temporal split (src_ip/dst_ip/src_mac/dst_mac/timestamp were dropped),
  * recomputing the rolling-window features per split (they are already baked in).

For class imbalance we DO NOT re-duplicate. We emit balanced class weights to be
consumed by the training loss instead (train.py --class-weights).

Outputs (next to the inputs):
  df_class_8_train_clean.csv, df_class_8_test_clean.csv, class_weights.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "Benign", 1: "WebBased", 2: "Spoofing", 3: "Recon",
    4: "Mirai", 5: "Dos", 6: "DDos", 7: "BruteForce",
}


def _row_keys(df, feat_cols):
    """Return a 1-D array of hashable per-row keys over the feature columns.

    Rows are stringified so list-valued packet columns (stored as text) and
    numeric columns hash consistently and cheaply.
    """
    return (
        df[feat_cols].astype(str).agg("\x1f".join, axis=1).values
    )


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/tutay/Tutay/Tutay_Sec/XG_NID/data/CIC_IoT2023_Processed_Data"
    train_path = os.path.join(root, "df_class_8_train.csv")
    test_path = os.path.join(root, "df_class_8_test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feat_cols = [c for c in train.columns if c != "Label" and c in test.columns]

    print("=" * 70)
    print("BEFORE")
    print("=" * 70)
    print(f"train rows={len(train)}  test rows={len(test)}")
    tr_dup = int(train.duplicated().sum())
    print(f"train exact-duplicate rows: {tr_dup} ({tr_dup/len(train)*100:.1f}%)")

    # ---- Defect A: collapse naive oversampling -> unique train flows --------
    train_unique = train.drop_duplicates().reset_index(drop=True)

    # ---- Defect B: drop test rows that are byte-identical to a train row ----
    train_keys = set(_row_keys(train_unique, feat_cols))
    test_keys = _row_keys(test, feat_cols)
    leaked_mask = np.fromiter((k in train_keys for k in test_keys),
                              dtype=bool, count=len(test))
    test_clean = test.loc[~leaked_mask].reset_index(drop=True)

    # ---- Per-class report --------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-CLASS (label: train_before -> train_unique | "
          "test_before -> test_clean [leaked removed])")
    print("=" * 70)
    for lab in sorted(LABEL_NAMES):
        tb = int((train["Label"] == lab).sum())
        tu = int((train_unique["Label"] == lab).sum())
        eb = int((test["Label"] == lab).sum())
        leaked = int(leaked_mask[(test["Label"] == lab).values].sum())
        ec = eb - leaked
        print(f"  {lab} {LABEL_NAMES[lab]:<11} "
              f"train {tb:>6} -> {tu:>6} ({tu/tb*100:5.1f}% unique) | "
              f"test {eb:>5} -> {ec:>5}  (leaked -{leaked})")

    # ---- Balanced class weights (sklearn 'balanced' formula) ---------------
    counts = train_unique["Label"].value_counts().sort_index()
    n_classes = len(counts)
    total = counts.sum()
    weights = {int(c): float(total / (n_classes * n)) for c, n in counts.items()}

    print("\n" + "=" * 70)
    print("AFTER")
    print("=" * 70)
    print(f"train rows={len(train_unique)}  test rows={len(test_clean)}")
    print(f"leaked test rows removed: {int(leaked_mask.sum())} "
          f"({leaked_mask.mean()*100:.2f}% of original test)")
    print(f"class weights (balanced): "
          f"{ {k: round(v,3) for k,v in weights.items()} }")

    # ---- Write outputs -----------------------------------------------------
    out_train = os.path.join(root, "df_class_8_train_clean.csv")
    out_test = os.path.join(root, "df_class_8_test_clean.csv")
    out_w = os.path.join(root, "class_weights.json")
    train_unique.to_csv(out_train, index=False)
    test_clean.to_csv(out_test, index=False)
    with open(out_w, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\n[write] {out_train}")
    print(f"[write] {out_test}")
    print(f"[write] {out_w}")


if __name__ == "__main__":
    main()
