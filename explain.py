"""End-to-end explainability CLI: pick a graph, run IG + generative explainer.

Examples:
  # Preview prompt only (no LLM weights needed):
  python explain.py --data-root data/CIC_IoT2023_Processed_Data \
      --checkpoint checkpoints/xgnid_hgnn_edge_tuned.pth --idx 100

  # Use a local HF model (downloads weights on first run):
  python explain.py ... --backend hf --hf-model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import os
import sys

import numpy as np
import torch

from Utility.Functions import NIDSDataset
from Utility.Model import HeteroGNN_Edge
from Utility.Explainer import explain_graph_ig
from Utility.Generative_Explainer import (
    explain_generative, PreviewBackend, HFCausalBackend, CLASS_NAMES_8,
)


LABEL_DICT = {
    "Benign": 0, "WebBased": 1, "Spoofing": 2, "Recon": 3,
    "Mirai": 4, "Dos": 5, "DDos": 6, "BruteForce": 7,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--idx", type=int, default=0,
                   help="Index into the test dataset.")
    p.add_argument("--n-steps", type=int, default=32, help="IG interpolation steps.")
    p.add_argument("--top-k-flow", type=int, default=10)
    p.add_argument("--top-k-payload", type=int, default=10)
    p.add_argument("--backend", choices=["preview", "hf"], default="preview")
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--load-4bit", action="store_true")
    return p.parse_args()


def load_model(checkpoint_path, sample, device):
    args_init = {"hidden_size": 64, "eps": 1.0, "attn_size": 32,
                 "lr": 0.005, "epochs": 1, "weight_decay": 1e-4}
    model = HeteroGNN_Edge(sample, args_init, aggr="mean").to(device)
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict):
        # Lazy params need a forward pass before loading state_dict. Use a
        # batch >1 because BN1d in train mode rejects batch=1; eval mode would
        # also work but switching back-and-forth is fragile.
        from torch_geometric.loader import DataLoader
        dummy = next(iter(DataLoader(
            [sample.cpu(), sample.cpu()], batch_size=2))).to(device)
        with torch.no_grad():
            model(dummy.x_dict, dummy.edge_index_dict,
                  dummy.edge_attr_dict, dummy)
        model.load_state_dict(obj)
    else:
        model = obj
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = NIDSDataset(
        root=args.data_root, label_dict=LABEL_DICT,
        filename=[os.path.join(args.data_root, "df_class_8_test.csv")],
        skip_processing=True, test=True, single_file=True,
    )
    if args.idx >= len(test_ds):
        sys.exit(f"--idx {args.idx} out of range (test size={len(test_ds)})")
    data = test_ds[args.idx]
    true_label = int(data.y)
    print(f"[graph #{args.idx}] true class = {CLASS_NAMES_8[true_label]} ({true_label})")

    sample = test_ds[0].to(device)
    model = load_model(args.checkpoint, sample, device)

    print(f"[ig] running Integrated Gradients (n_steps={args.n_steps})")
    expl = explain_graph_ig(
        model, data, device,
        target=None,
        n_steps=args.n_steps,
        top_k_flow=args.top_k_flow,
        top_k_payload=args.top_k_payload,
    )
    print(f"[ig] predicted = {CLASS_NAMES_8[expl.pred_class]} ({expl.pred_class})  "
          f"log-prob={expl.pred_logprob:.3f}")
    print("[ig] top flow features:")
    for name, attr in expl.top_flow:
        print(f"     {name:50s}  attr={attr:+.3e}")
    print("[ig] top payload bytes:")
    for idx, attr in expl.top_payload_bytes:
        print(f"     byte[{idx:4d}]  attr={attr:+.3e}")

    flow_x_np = data["flow"].x.cpu().numpy()
    packet_x_np = data["packet"].x.cpu().numpy()

    if args.backend == "hf":
        print(f"[llm] loading HF model {args.hf_model} (4bit={args.load_4bit})")
        backend = HFCausalBackend(model_id=args.hf_model, load_in_4bit=args.load_4bit)
    else:
        backend = PreviewBackend()

    gen = explain_generative(expl, flow_x_np, packet_x_np, backend=backend)
    print("\n========== PROMPT ==========")
    print(gen.prompt)
    print("\n========== RESPONSE ==========")
    print(gen.response)


if __name__ == "__main__":
    main()
