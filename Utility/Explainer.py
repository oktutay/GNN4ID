"""Explainability for the XG-NID HGNN: Integrated Gradients (block #5 of the paper).

The paper computes IG over the heterogeneous graph inputs and reports
feature-importance for both flow-level features and packet-level (payload)
features. Captum's IntegratedGradients attributes a target class score back
to a tuple of input tensors. Because PyG's HeteroData passes inputs as dicts,
we wrap the model forward into a closure that takes (flow_x, packet_x) as
positional tensors and threads the rest through as captured constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients


# Names of the 82 flow-level features in the order the dataset emits them.
# Derived from inspecting the column order of df_class_8_train.csv after
# dropping ['Label'] + the 14 udps.* packet-array columns. Matches what
# NIDSDataset._get_flow_node_features stacks into flow.x.
FLOW_FEATURE_NAMES_82 = [
    "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
    "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
    "bidirectional_min_ps", "bidirectional_mean_ps",
    "bidirectional_stddev_ps", "bidirectional_max_ps",
    "src2dst_min_ps", "src2dst_mean_ps", "src2dst_stddev_ps", "src2dst_max_ps",
    "dst2src_min_ps", "dst2src_mean_ps", "dst2src_stddev_ps", "dst2src_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms", "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_min_piat_ms", "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
    "src2dst_syn_packets", "src2dst_cwr_packets", "src2dst_ece_packets",
    "src2dst_urg_packets", "src2dst_ack_packets", "src2dst_psh_packets",
    "src2dst_rst_packets", "src2dst_fin_packets",
    "dst2src_syn_packets", "dst2src_cwr_packets", "dst2src_ece_packets",
    "dst2src_urg_packets", "dst2src_ack_packets", "dst2src_psh_packets",
    "dst2src_rst_packets", "dst2src_fin_packets",
    "packet_size_variation",
    "Rolling_UDP_Requests_SourceDestination", "Rolling_UDP_Requests_Destination",
    "Rolling_TCP_Requests_SourceDestination", "Rolling_TCP_Requests_Destination",
    "Rolling_ACK_Packets_SourceDestination", "Rolling_ACK_Packets_Destination",
    "Rolling_FIN_Packets_SourceDestination", "Rolling_FIN_Packets_Destination",
    "Rolling_rst_Packets_SourceDestination", "Rolling_rst_Packets_Destination",
    "Rolling_psh_Packets_SourceDestination", "Rolling_psh_Packets_Destination",
    "Rolling_SYN_Packets_SourceDestination", "Rolling_SYN_Packets_Destination",
    "Unique_Ports_In_SourceDestinationIP",
    "Rolling_ICMP_Requests_SourceDestination",
    "Rolling_ICMP_Requests_Destination",
    "Rolling_http_port_SourceDestination", "Rolling_http_port_Destination",
    "Rolling_Duration_Destination", "Rolling_Duration_SourceDestination",
    "Rolling_DNS_request_SourceDestination", "Rolling_DNS_request_Destination",
    "Rolling_DNS_request_SourceDestination2",
    "Rolling_DNS_request_Destination2",
    "Rolling_vulnerable_port",
    "Rolling_packets_destination", "Rolling_bipackets_destination",
    "Exp_0", "Exp_-1",
    "proto_1", "proto_2", "proto_6", "proto_17", "proto_58",
]
assert len(FLOW_FEATURE_NAMES_82) == 82


@dataclass
class IGExplanation:
    pred_class: int
    pred_logprob: float
    flow_attr: np.ndarray             # shape: (num_flow_nodes, 82)
    packet_attr: np.ndarray           # shape: (num_packet_nodes, 1500)
    top_flow: List[Tuple[str, float]]   # [(name, attribution), ...]
    top_payload_bytes: List[Tuple[int, float]]   # [(byte_index, attribution), ...]


def explain_graph_ig(
    model: torch.nn.Module,
    data,                      # a single HeteroData object
    device: torch.device,
    target: int | None = None,
    n_steps: int = 32,
    top_k_flow: int = 10,
    top_k_payload: int = 10,
) -> IGExplanation:
    """Compute Integrated Gradients on a single graph object.

    Args:
        model: trained HeteroGNN_Edge.
        data: a single HeteroData (un-batched).
        device: cuda or cpu.
        target: class index to attribute against; if None uses the predicted class.
        n_steps: IG interpolation steps. Higher = smoother but slower.
        top_k_flow / top_k_payload: how many top features to return.

    Returns:
        IGExplanation with raw attribution arrays + ranked top features.
    """
    from torch_geometric.data import Batch
    model.eval()
    data = data.clone().to(device)
    if not isinstance(data, Batch):
        # HeteroData with a single graph still needs batch tensors for global pool.
        data = Batch.from_data_list([data])

    flow_x = data["flow"].x.detach().clone().requires_grad_(True)
    packet_x = data["packet"].x.detach().clone().requires_grad_(True)

    edge_index_dict = data.edge_index_dict
    edge_attr_dict = data.edge_attr_dict
    batch_holder = data

    with torch.no_grad():
        full_logits = model({"flow": flow_x, "packet": packet_x},
                            edge_index_dict, edge_attr_dict, batch_holder)
    if target is None:
        target = int(full_logits.argmax(dim=1).item())
    pred_logprob = float(full_logits[0, target].item())

    def fwd(flow_in, packet_in):
        out = model({"flow": flow_in, "packet": packet_in},
                    edge_index_dict, edge_attr_dict, batch_holder)
        return out

    ig = IntegratedGradients(fwd)
    flow_baseline = torch.zeros_like(flow_x)
    packet_baseline = torch.zeros_like(packet_x)
    attrs = ig.attribute(
        inputs=(flow_x, packet_x),
        baselines=(flow_baseline, packet_baseline),
        target=target,
        n_steps=n_steps,
        internal_batch_size=1,
    )
    flow_attr = attrs[0].detach().cpu().numpy()
    packet_attr = attrs[1].detach().cpu().numpy()

    flow_scores = flow_attr.mean(axis=0)
    flow_order = np.argsort(-np.abs(flow_scores))
    top_flow = [(FLOW_FEATURE_NAMES_82[i], float(flow_scores[i]))
                for i in flow_order[:top_k_flow]]

    payload_scores = packet_attr.mean(axis=0)
    payload_order = np.argsort(-np.abs(payload_scores))
    top_payload = [(int(i), float(payload_scores[i]))
                   for i in payload_order[:top_k_payload]]

    return IGExplanation(
        pred_class=target,
        pred_logprob=pred_logprob,
        flow_attr=flow_attr,
        packet_attr=packet_attr,
        top_flow=top_flow,
        top_payload_bytes=top_payload,
    )
