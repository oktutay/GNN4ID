"""Tuned training driver for XG-NID HGNN.

Improvements over train.py:
- 90/10 train/val split (so LR scheduler can step on a held-out signal)
- AdamW + weight_decay=1e-4
- CosineAnnealingLR scheduler
- Larger batch (128)
- Class-weighted NLL loss (test set has minority classes)
- Best-val-acc checkpoint + final test eval
- Gradient clipping (max_norm=5.0)
"""

import argparse
import glob
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from tqdm import tqdm

from Utility.Functions import NIDSDataset
from Utility.Model import HeteroGNN_Edge


LABEL_DICT = {
    "Benign": 0, "WebBased": 1, "Spoofing": 2, "Recon": 3,
    "Mirai": 4, "Dos": 5, "DDos": 6, "BruteForce": 7,
}
CLASS_NAMES = list(LABEL_DICT.keys())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", default="checkpoints/xgnid_hgnn_edge_tuned.pth")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    preds_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
        loss = F.nll_loss(out, batch.y, weight=class_weights)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
        total_loss += loss.item() * batch.num_graphs
        preds_all.append(pred.cpu().numpy())
        labels_all.append(batch.y.cpu().numpy())
    return (correct / total,
            total_loss / total,
            np.concatenate(preds_all),
            np.concatenate(labels_all))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__}")

    print("[data] loading train + test dataset")
    train_full = NIDSDataset(
        root=args.data_root, label_dict=LABEL_DICT,
        filename=[os.path.join(args.data_root, "df_class_8_train.csv")],
        skip_processing=True, test=False, single_file=True,
    )
    test_ds = NIDSDataset(
        root=args.data_root, label_dict=LABEL_DICT,
        filename=[os.path.join(args.data_root, "df_class_8_test.csv")],
        skip_processing=True, test=True, single_file=True,
    )
    print(f"[data] train_full={len(train_full)}  test={len(test_ds)}")

    n = len(train_full)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(train_full, val_idx.tolist())
    print(f"[split] train={len(train_ds)} val={len(val_ds)}")

    print("[data] computing class weights from train labels…")
    label_counts = Counter()
    for i in tqdm(train_idx[:5000], desc="sample"):
        label_counts[int(train_full[int(i)].y)] += 1
    print(f"[data] sampled label dist={dict(sorted(label_counts.items()))}")
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (len(label_counts) * max(1, label_counts.get(c, 1)))
         for c in range(len(LABEL_DICT))],
        dtype=torch.float32, device=device,
    )
    weights = weights / weights.mean()
    print(f"[data] class weights (normalized) = {weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    sample = train_full[0].to(device)
    model_args = {"hidden_size": args.hidden_size, "eps": 1.0,
                  "attn_size": 32, "lr": args.lr, "epochs": args.epochs,
                  "weight_decay": args.weight_decay}
    model = HeteroGNN_Edge(sample, model_args, aggr="mean").to(device)
    init_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        model(init_batch.x_dict, init_batch.edge_index_dict,
              init_batch.edge_attr_dict, init_batch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] HeteroGNN_Edge params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_correct = 0
        run_n = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict,
                        batch.edge_attr_dict, batch)
            loss = F.nll_loss(out, batch.y, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            run_loss += loss.item() * batch.num_graphs
            run_correct += (out.argmax(dim=1) == batch.y).sum().item()
            run_n += batch.num_graphs

        train_loss = run_loss / run_n
        train_acc = run_correct / run_n
        val_acc, val_loss, _, _ = evaluate(model, val_loader, device, weights)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            marker = " *best*"
            torch.save(model.state_dict(), args.save_path)
        print(f"Epoch {epoch:2d}: tr_acc={train_acc:.4f} tr_loss={train_loss:.4f}"
              f" val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
              f" lr={cur_lr:.5f} dt={dt:.1f}s{marker}", flush=True)

    print(f"[best] epoch={best_epoch} val_acc={best_val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    print("[test] evaluating best model on test set…")
    test_acc, test_loss, preds, labels = evaluate(model, test_loader, device, None)
    print(f"[test] accuracy={test_acc:.4f} loss={test_loss:.4f}")
    print("\n[test] classification report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    print("\n[test] confusion matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)


if __name__ == "__main__":
    main()
