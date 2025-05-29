# NEW: mf_gnn_actor_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class MF_GNN_ActorCritic(nn.Module):
    def __init__(self, node_feat_dim, edge_dim, gnn_hidden_dim, n_actions):
        super().__init__()
        # GAT layers
        self.gat1 = GATConv(node_feat_dim, gnn_hidden_dim, edge_dim=edge_dim, heads=4, concat=False)
        self.gat2 = GATConv(gnn_hidden_dim, gnn_hidden_dim, edge_dim=edge_dim, heads=4, concat=False)
        # actor & critic heads
        self.actor = nn.Sequential(
            nn.Linear(gnn_hidden_dim*2, gnn_hidden_dim), nn.ReLU(),
            nn.Linear(gnn_hidden_dim,   n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(gnn_hidden_dim*2, gnn_hidden_dim), nn.ReLU(),
            nn.Linear(gnn_hidden_dim,   1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, F]
        h = F.relu(self.gat1(x, edge_index, edge_attr))
        h = F.relu(self.gat2(h, edge_index, edge_attr))
        # mean-field: global mean pooling
        g = global_mean_pool(h, batch)         # [B, H]
        mf = g[batch]                          # [N, H]
        cat = torch.cat([h, mf], dim=-1)       # [N, 2H]
        logits = self.actor(cat)               # [N, A]
        values = self.critic(cat).squeeze(-1)   # [N]
        return logits, values