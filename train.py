"""Training driver for the XG-NID HGNN (paper-faithful: GATConv + edge attrs).

Replaces the Windows-path notebook flow with a runnable Linux script.
Uses Utility.NIDSDataset to load processed graph objects, HeteroGNN_Edge as
the 2x GATConv heterogeneous model with edge attributes, and trains with
train_with_edge_Att from Utility.Training.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from Utility.Functions import NIDSDataset
from Utility.Model import HeteroGNN_Edge, HeteroGNN
from Utility.Training import (
    train_with_edge_Att,
    test_cm_with_edge_att,
    train,
    test_cm,
)


LABEL_DICT = {
    "Benign": 0,
    "WebBased": 1,
    "Spoofing": 2,
    "Recon": 3,
    "Mirai": 4,
    "Dos": 5,
    "DDos": 6,
    "BruteForce": 7,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="Root with raw/ (CSV) and processed/ (graph .pt). "
                        "Equivalent to the F:/CIC_IOT/Extracted_Flow_Features/ in the notebook.")
    p.add_argument("--train-csv-glob", default=None,
                   help="Glob for the train CSV (only needed when graphs are not yet processed).")
    p.add_argument("--test-csv-glob", default=None,
                   help="Glob for the test CSV.")
    p.add_argument("--skip-processing", action="store_true",
                   help="Reuse already-processed graph objects in <data-root>/processed/.")
    p.add_argument("--model", choices=["edge", "no_edge"], default="edge",
                   help="edge = HeteroGNN_Edge (GATConv + edge attrs, paper-faithful). "
                        "no_edge = HeteroGNN (SAGEConv).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", default="checkpoints/xgnid_hgnn.pth")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load --save-path and run test_cm_with_edge_att.")
    return p.parse_args()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__}")

    train_files = sorted(glob.glob(args.train_csv_glob)) if args.train_csv_glob else []
    test_files = sorted(glob.glob(args.test_csv_glob)) if args.test_csv_glob else []
    if not args.skip_processing and not train_files:
        sys.exit("Need --train-csv-glob unless --skip-processing is set.")

    print("[data] building train dataset")
    train_ds = NIDSDataset(
        root=args.data_root,
        label_dict=LABEL_DICT,
        filename=train_files,
        skip_processing=args.skip_processing,
        test=False,
        single_file=True,
    )
    print(f"[data] train graphs={len(train_ds)}")

    test_ds = None
    if test_files or args.skip_processing:
        print("[data] building test dataset")
        test_ds = NIDSDataset(
            root=args.data_root,
            label_dict=LABEL_DICT,
            filename=test_files,
            skip_processing=args.skip_processing,
            test=True,
            single_file=True,
        )
        print(f"[data] test graphs={len(test_ds)}")

    sample = train_ds[0].to(device)
    print(f"[data] sample metadata: nodes={sample.node_types} edges={sample.edge_types}")
    print(f"[data] flow.x={tuple(sample['flow'].x.shape)} packet.x={tuple(sample['packet'].x.shape)}")

    model_args = {
        "device": device,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "weight_decay": 1e-5,
        "lr": args.lr,
        "attn_size": 32,
        "eps": 1.0,
    }

    if args.model == "edge":
        model = HeteroGNN_Edge(sample, model_args, aggr="mean").to(device)
        train_fn = train_with_edge_Att
        test_fn = test_cm_with_edge_att
    else:
        model = HeteroGNN(sample, model_args, aggr="mean").to(device)
        train_fn = train
        test_fn = test_cm

    init_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    init_batch = next(iter(init_loader)).to(device)
    with torch.no_grad():
        if args.model == "edge":
            model(init_batch.x_dict, init_batch.edge_index_dict, init_batch.edge_attr_dict, init_batch)
        else:
            model(init_batch.x_dict, init_batch.edge_index_dict, init_batch)
    print(f"[model] {model.__class__.__name__} params={sum(p.numel() for p in model.parameters()):,}")

    if not args.eval_only:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        train_fn(train_loader, model, model_args, device)
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(model, args.save_path)
        print(f"[save] {args.save_path}")
    else:
        model = torch.load(args.save_path, map_location=device)
        model.eval()

    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        acc, preds, labels = test_fn(test_loader, model, device)
        print(f"[test] accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
