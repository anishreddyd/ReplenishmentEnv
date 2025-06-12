import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GraphMixer(nn.Module):
    """
    A graph‐based mixing network for multi‐agent value decomposition.
    Expects:
      node_feats: [num_nodes, node_dim]
      edge_index: [2, num_edges]
      edge_attr : [num_edges, edge_dim]
      batch      : [num_nodes]  # which graph each node belongs to
    Returns:
      out        : [batch_size] # one value per graph in the batch
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # GATConv: in_channels=node_dim, out_channels=hidden_dim, edge_dim=edge_dim
        # heads=4 means final feature dim is hidden_dim (with concat=False) or hidden_dim*4 (if concat=True)
        self.gat    = GATConv(node_dim, hidden_dim, edge_dim=edge_dim, heads=4, concat=False)
        self.hyper = nn.Linear(hidden_dim, 1)

    def forward(self, node_feats, edge_index, edge_attr, batch):
        h = F.relu(self.gat(node_feats, edge_index, edge_attr))
        # aggregate node embeddings into graph embeddings
        g = global_mean_pool(h, batch)    # [batch_size, hidden_dim]
        out = self.hyper(g).squeeze(-1)    # [batch_size]
        return out
