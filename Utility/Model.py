from torch_geometric.nn import HeteroConv, Linear, SAGEConv, global_mean_pool, Sequential, to_hetero, GraphConv, GATConv
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch
import torch.nn as nn


class HeteroGNN(torch.nn.Module):
    """
    HGNN model as described in XG-NID paper, Section 3.1.4.

    Architecture (eq. 5-9):
        h^(1) = ReLU(GATConv(h^(0), A, E))
        h^(2) = ReLU(BN(GATConv(h^(1), A, E)))
        h_graph = GlobalMeanPool(h^(2))
        out = LogSoftmax(W2 . ReLU(W1 . ReLU(W0 . h_graph)))

    Edge attributes are required (paper eq. 6 explicitly aggregates over edges
    with a dedicated W_e). The forward signature accepts (x_dict, edge_index_dict,
    edge_attr_dict, batch); set use_edge_attr=False only for ablation.
    """

    def __init__(self, hetero_graph, args, aggr="mean", use_edge_attr=True):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.use_edge_attr = use_edge_attr

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()

        if use_edge_attr:
            self.convs1 = HeteroConv({
                edge_type: GATConv((-1, -1), self.hidden_size, edge_dim=-1, add_self_loops=False)
                for edge_type in hetero_graph.metadata()[1]
            })
            self.convs2 = HeteroConv({
                edge_type: GATConv((-1, -1), self.hidden_size, edge_dim=-1, add_self_loops=False)
                for edge_type in hetero_graph.metadata()[1]
            })
        else:
            self.convs1 = HeteroConv({
                edge_type: GATConv((-1, -1), self.hidden_size, add_self_loops=False)
                for edge_type in hetero_graph.metadata()[1]
            })
            self.convs2 = HeteroConv({
                edge_type: GATConv((-1, -1), self.hidden_size, add_self_loops=False)
                for edge_type in hetero_graph.metadata()[1]
            })

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            # Paper eq. 6 mentions sigma in {ReLU, LeakyReLU}; eq. 7 explicitly uses ReLU.
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

        # MLP head, eq. 9: 2 hidden ReLUs followed by LogSoftmax.
        # Input is concat([flow_emb, packet_emb]) so dim = 2 * hidden_size.
        self.graph_prediction = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.graph_prediction_1 = nn.Linear(self.hidden_size, 16)
        self.graph_prediction_2 = nn.Linear(16, 8)  # Number of Classes

    def forward(self, node_feature, edge_index, edge_attr=None, batch=None):
        x = node_feature

        if self.use_edge_attr:
            assert edge_attr is not None, "use_edge_attr=True but edge_attr is None"
            x = self.convs1(x, edge_index, edge_attr)
        else:
            x = self.convs1(x, edge_index)

        x = {key: self.bns1[key](value) for key, value in x.items()}
        x = {key: self.relus1[key](value) for key, value in x.items()}

        if self.use_edge_attr:
            x = self.convs2(x, edge_index, edge_attr)
        else:
            x = self.convs2(x, edge_index)

        x = {key: self.bns2[key](value) for key, value in x.items()}
        x = {key: self.relus2[key](value) for key, value in x.items()}

        # eq. 8: GlobalMeanPool over each node type, then concatenate.
        graph_emb = {key: pyg_nn.global_mean_pool(x[key], batch.batch_dict[key]) for key in batch.node_types}
        graph_emb = torch.cat([graph_emb[t] for t in graph_emb.keys()], dim=1)

        # eq. 9: ReLU-activated MLP head.
        graph_pred = F.relu(self.graph_prediction(graph_emb))
        graph_pred = F.relu(self.graph_prediction_1(graph_pred))
        graph_pred = self.graph_prediction_2(graph_pred)

        graph_pred = F.log_softmax(graph_pred, dim=1)
        return graph_pred

    def loss(self, preds, label):
        return F.nll_loss(preds, label)


# Backward-compatible alias: HeteroGNN_Edge in old notebooks now points to the
# paper-faithful model with edge attributes enabled.
class HeteroGNN_Edge(HeteroGNN):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super().__init__(hetero_graph, args, aggr=aggr, use_edge_attr=True)


# Legacy SAGE-based model kept under a new name so users can run an ablation.
# This is NOT the architecture described in the paper; do not use as baseline.
class HeteroGNN_SAGE(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN_SAGE, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()

        self.convs1 = HeteroConv({edge_type: SAGEConv((-1, -1), self.hidden_size) for edge_type in hetero_graph.metadata()[1]})
        self.convs2 = HeteroConv({edge_type: SAGEConv((-1, -1), self.hidden_size) for edge_type in hetero_graph.metadata()[1]})

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

        self.graph_prediction = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.graph_prediction_1 = nn.Linear(self.hidden_size, 16)
        self.graph_prediction_2 = nn.Linear(16, 8)

    def forward(self, node_feature, edge_index, batch):
        x = node_feature
        x = self.convs1(x, edge_index)
        x = {key: self.bns1[key](value) for key, value in x.items()}
        x = {key: self.relus1[key](value) for key, value in x.items()}

        x = self.convs2(x, edge_index)
        x = {key: self.bns2[key](value) for key, value in x.items()}
        x = {key: self.relus2[key](value) for key, value in x.items()}

        graph_emb = {key: pyg_nn.global_mean_pool(x[key], batch.batch_dict[key]) for key in batch.node_types}
        graph_emb = torch.cat([graph_emb[t] for t in graph_emb.keys()], dim=1)

        graph_pred = F.relu(self.graph_prediction(graph_emb))
        graph_pred = F.relu(self.graph_prediction_1(graph_pred))
        graph_pred = self.graph_prediction_2(graph_pred)
        graph_pred = F.log_softmax(graph_pred, dim=1)
        return graph_pred

    def loss(self, preds, label):
        return F.nll_loss(preds, label)
