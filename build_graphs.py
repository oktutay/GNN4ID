"""Materialize train + test graph objects from preprocessed CIC-IoT2023 CSVs.

Run once before training. After this, train.py uses --skip-processing to reuse
the .pt files in <data-root>/processed/.
"""

import os
import sys

from Utility.Functions import NIDSDataset


LABEL_DICT = {
    "Benign": 0, "WebBased": 1, "Spoofing": 2, "Recon": 3,
    "Mirai": 4, "Dos": 5, "DDos": 6, "BruteForce": 7,
}


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?",
                    default="/home/tutay/Tutay/Tutay_Sec/XG_NID/data/CIC_IoT2023_Processed_Data",
                    help="data root holding df_class_8_{train,test}[_clean].csv; graphs go to <root>/processed/")
    ap.add_argument("--include-packetflag", action="store_true",
                    help="prepend the 8 TCP flags to each packet node (1508-d instead of 1500-d; "
                         "checkpoints trained on 1500-d packet nodes will not load)")
    return ap.parse_args()


def main():
    args = parse_args()
    root = args.root
    # Prefer the de-leaked / de-duplicated CSVs produced by clean_existing_data.py
    # when they exist; fall back to the original (leaky) combined CSVs otherwise.
    train_clean = os.path.join(root, "df_class_8_train_clean.csv")
    test_clean = os.path.join(root, "df_class_8_test_clean.csv")
    if os.path.exists(train_clean) and os.path.exists(test_clean):
        train_csv, test_csv = train_clean, test_clean
        print("[build] using CLEAN (de-leaked) CSVs")
    else:
        train_csv = os.path.join(root, "df_class_8_train.csv")
        test_csv = os.path.join(root, "df_class_8_test.csv")
        print("[build] WARNING: using original (leaky) CSVs; run clean_existing_data.py first")

    print(f"[build] root={root}")
    print(f"[build] processing TRAIN ({train_csv})")
    train_ds = NIDSDataset(
        root=root, label_dict=LABEL_DICT, filename=[train_csv],
        skip_processing=False, test=False, single_file=True,
        include_packetflag=args.include_packetflag,
    )
    print(f"[build] train graphs created: {len(train_ds)}")

    print(f"[build] processing TEST ({test_csv})")
    test_ds = NIDSDataset(
        root=root, label_dict=LABEL_DICT, filename=[test_csv],
        skip_processing=False, test=True, single_file=True,
        include_packetflag=args.include_packetflag,
    )
    print(f"[build] test graphs created: {len(test_ds)}")


if __name__ == "__main__":
    main()
