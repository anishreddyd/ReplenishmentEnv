# NEW: graph_mixer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GraphMixer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.gat    = GATConv(node_dim, hidden_dim, edge_dim=edge_dim, heads=4)
        self.hyper = nn.Linear(hidden_dim, 1)

    def forward(self, node_feats, edge_index, edge_attr, batch):
        h = F.relu(self.gat(node_feats, edge_index, edge_attr))
        g = global_mean_pool(h, batch)         # [B, hidden]
        out = self.hyper(g).squeeze(-1)         # [B]
        return out