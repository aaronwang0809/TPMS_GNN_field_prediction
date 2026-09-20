#!/usr/bin/env python3
"""GNN model definitions extracted from Notebook 07."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv

HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.10
# ============================================================
# MODULE 3 — MODEL DEFINITIONS
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing

def expand_graph_attr(data):
    return data.graph_attr.expand(data.x.size(0), -1)

class NodeMLP(nn.Module):
    """Non-graph baseline: local node features + scaffold design vector."""
    def __init__(self, node_dim=8, graph_dim=12, hidden=HIDDEN_DIM, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + graph_dim, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, out_dim),
        )
    def forward(self, data):
        return self.net(torch.cat([data.x, expand_graph_attr(data)], dim=-1))

class GraphSAGEBaseline(nn.Module):
    """Connectivity-aware baseline; intentionally does not consume edge_attr."""
    def __init__(self, node_dim=8, graph_dim=12, hidden=HIDDEN_DIM,
                 layers=NUM_LAYERS, out_dim=2):
        super().__init__()
        self.inp = nn.Linear(node_dim + graph_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, data):
        h = F.relu(self.inp(torch.cat([data.x, expand_graph_attr(data)], dim=-1)))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index)
            h = norm(h + F.relu(h_new))
            h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.head(h)

class EdgeMessageLayer(MessagePassing):
    """Edge-conditioned message passing with residual node update."""
    def __init__(self, hidden, edge_dim=9):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2*hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, edge_attr):
        agg = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        upd = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + upd)

    def message(self, h_i, h_j, edge_attr):
        return self.msg_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))

class EdgeAwareGNN(nn.Module):
    """Proposed topology-aware model using node, edge, and scaffold design features."""
    def __init__(self, node_dim=8, edge_dim=9, graph_dim=12,
                 hidden=HIDDEN_DIM, layers=NUM_LAYERS, out_dim=2):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim + graph_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            [EdgeMessageLayer(hidden, edge_dim=edge_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, out_dim),
        )

    def forward(self, data):
        h = self.node_encoder(torch.cat([data.x, expand_graph_attr(data)], dim=-1))
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
            h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.head(h)

MODEL_FACTORIES = {
    "NodeMLP": lambda: NodeMLP(),
    "GraphSAGE": lambda: GraphSAGEBaseline(),
    "EdgeAwareGNN": lambda: EdgeAwareGNN(),
}

for name, factory in MODEL_FACTORIES.items():
    m = factory()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:14s}: {n_params:,} trainable parameters")
