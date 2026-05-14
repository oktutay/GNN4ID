"""
Integrated Gradient Explainer for the HGNN classifier, paper Section 3.1.5.

Implements eq. 10:

    IG_i(x) = (x_i - x'_i) * integral_{alpha=0..1} dF/dx_i (x' + alpha*(x-x')) d_alpha

We approximate the integral with a Riemann sum over `n_steps` interpolation
points and apply the rule independently to flow-node features and packet-node
features. Edge attributes are not differentiated (the GATConv path doesn't
backprop through them in a way that maps cleanly onto the paper's eq. 10) --
this matches what XG-NID reports as "feature-based local explanations".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData


@dataclass
class IGAttribution:
    """Container for a single-sample IG result.

    Attributes are returned as numpy arrays so they can be sorted/printed/
    fed into the LLM Explainer (Algorithm 3) without further conversion.
    """
    predicted_class: int
    predicted_logprob: float
    flow_attr: np.ndarray         # shape: (num_flow_features,)
    packet_attr: np.ndarray       # shape: (num_packets, 1500) for payload bytes
    flow_values: np.ndarray       # actual flow feature values (for the prompt)
    packet_values: np.ndarray     # actual payload bytes per packet


class IntegratedGradientExplainer:
    """Computes Integrated Gradient attributions for a single graph at a time.

    The HGNN expects a `Batch` (PyG bundles even single graphs into a batch
    of size 1). We mimic that here so the model's forward signature is
    untouched.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda",
                 n_steps: int = 50):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, data: HeteroData,
                target_class: Optional[int] = None,
                flow_baseline: Optional[torch.Tensor] = None,
                packet_baseline: Optional[torch.Tensor] = None) -> IGAttribution:
        """Compute IG attributions for the prediction on a single HeteroData.

        Args:
            data: A single PyG HeteroData object (same shape as those produced
                by NIDSDataset).
            target_class: If None, IG is computed wrt the predicted class.
            flow_baseline / packet_baseline: Reference inputs x'. Default = zeros
                (paper Sec 3.1.5: "If a baseline input is not provided, zero is
                used as the default value").
        """
        batch = Batch.from_data_list([data]).to(self.device)

        flow_x = batch['flow'].x.detach().clone()
        packet_x = batch['packet'].x.detach().clone()

        if flow_baseline is None:
            flow_baseline = torch.zeros_like(flow_x)
        else:
            flow_baseline = flow_baseline.to(self.device)
        if packet_baseline is None:
            packet_baseline = torch.zeros_like(packet_x)
        else:
            packet_baseline = packet_baseline.to(self.device)

        # Forward once at the actual input to get the predicted class.
        with torch.no_grad():
            logits = self._forward(batch, flow_x, packet_x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            target_logprob = float(logits[0, target_class].item())

        # Riemann-sum approximation of integral_{alpha=0..1} dF/dx d_alpha.
        # We accumulate the gradient of F[target_class] wrt the interpolated
        # flow_x and packet_x at n_steps positions along [baseline, input].
        flow_grad_sum = torch.zeros_like(flow_x)
        packet_grad_sum = torch.zeros_like(packet_x)

        for k in range(1, self.n_steps + 1):
            alpha = k / self.n_steps
            f_interp = flow_baseline + alpha * (flow_x - flow_baseline)
            p_interp = packet_baseline + alpha * (packet_x - packet_baseline)
            f_interp.requires_grad_(True)
            p_interp.requires_grad_(True)

            logits = self._forward(batch, f_interp, p_interp)
            score = logits[0, target_class]

            grads = torch.autograd.grad(
                score, [f_interp, p_interp],
                retain_graph=False, create_graph=False, allow_unused=True
            )
            g_f = grads[0] if grads[0] is not None else torch.zeros_like(flow_x)
            g_p = grads[1] if grads[1] is not None else torch.zeros_like(packet_x)
            flow_grad_sum = flow_grad_sum + g_f.detach()
            packet_grad_sum = packet_grad_sum + g_p.detach()

        avg_flow_grad = flow_grad_sum / self.n_steps
        avg_packet_grad = packet_grad_sum / self.n_steps

        flow_ig = (flow_x - flow_baseline) * avg_flow_grad
        packet_ig = (packet_x - packet_baseline) * avg_packet_grad

        return IGAttribution(
            predicted_class=target_class,
            predicted_logprob=target_logprob,
            flow_attr=flow_ig.detach().cpu().numpy().squeeze(0),
            packet_attr=packet_ig.detach().cpu().numpy(),
            flow_values=flow_x.detach().cpu().numpy().squeeze(0),
            packet_values=packet_x.detach().cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Internal: rebuild a batch with patched x tensors and call the model.
    # ------------------------------------------------------------------
    def _forward(self, original_batch: Batch,
                 flow_x: torch.Tensor, packet_x: torch.Tensor) -> torch.Tensor:
        x_dict = {'flow': flow_x, 'packet': packet_x}
        edge_index_dict = original_batch.edge_index_dict
        # The paper's HGNN uses edge attributes; legacy SAGE model does not.
        if hasattr(self.model, 'use_edge_attr') and not self.model.use_edge_attr:
            return self.model(x_dict, edge_index_dict, original_batch)
        # Default path: model expects edge_attr_dict.
        try:
            return self.model(x_dict, edge_index_dict,
                              original_batch.edge_attr_dict, original_batch)
        except TypeError:
            # Fallback to the no-edge-attr signature if the model is the legacy
            # SAGE-based ablation.
            return self.model(x_dict, edge_index_dict, original_batch)


# ---------------------------------------------------------------------------
# Convenience: rank features by attribution and return human-readable lists.
# ---------------------------------------------------------------------------

def top_flow_features(attr: IGAttribution, feature_names: Sequence[str],
                      top_n: int = 5):
    """Return the top-N flow features by absolute IG attribution as
    [(name, attribution, actual_value), ...] sorted descending."""
    importance = np.abs(attr.flow_attr)
    if len(feature_names) != importance.shape[0]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but flow_attr "
            f"has shape {importance.shape}"
        )
    order = np.argsort(-importance)[:top_n]
    return [(feature_names[i], float(attr.flow_attr[i]), float(attr.flow_values[i]))
            for i in order]


def top_payload_bytes(attr: IGAttribution, top_n: int = 32) -> bytes:
    """Aggregate the per-packet payload-byte attributions, then return the
    top_n bytes as raw bytes (for the LLM Explainer's payload query).

    Algorithm 3, lines 19-24: normalize importance vectors, average across
    packets, sort descending, convert the top decimals to hex/ASCII.
    """
    if attr.packet_attr.size == 0:
        return b''
    importance = np.abs(attr.packet_attr)
    # Normalize each packet row (paper: "Normalize P_imp vectors").
    norms = np.linalg.norm(importance, axis=1, keepdims=True) + 1e-12
    importance = importance / norms
    avg = importance.mean(axis=0)  # shape: (1500,)
    order = np.argsort(-avg)[:top_n]
    selected = attr.packet_values.mean(axis=0)[order]
    selected_bytes = bytes(int(round(v)) & 0xFF for v in selected)
    return selected_bytes
