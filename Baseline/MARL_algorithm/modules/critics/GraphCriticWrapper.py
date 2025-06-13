import torch
import torch.nn as nn

from .graph_mixer import GraphMixer
# we import the two pieces we need

class GraphCriticWrapper(nn.Module):
    """
    Wraps your GraphMixer and a static graph (edge_index/edge_attr)
    and turns batch["obs"] (shape [B, T, N, D]) into per‐timestep values [B, T].
    """
    def __init__(self, graph_mixer: nn.Module, graph_data: dict):
        super().__init__()
        self.gm = graph_mixer

        # graph_data comes from GraphWrapper.get_graph()
        # which returns a dict:
        # {
        #   "edge_index": Tensor[2, E],
        #   "edge_attr":  Tensor[E, edge_dim],
        #   "batch":      Tensor[N]  # all zeros if single‐graph
        # }
        # Register edge tensors as buffers so they move with the module
        self.register_buffer("edge_index", graph_data["edge_index"])
        self.register_buffer("edge_attr",  graph_data["edge_attr"])
        # The original batch vector (all zeros) we ignore, we'll rebuild per-timestep
        # since we pool one graph per environment. Keep N for device bookkeeping
        self.register_buffer("_orig_batch", graph_data["batch"])

    def forward(self, batch):
        # batch["obs"]: [B, T, N, D]
        obs = batch["obs"]  # a tensor on the learner’s device
        B, T, N, D = obs.shape

        # We'll build a batch_idx vector of length N that points
        # all nodes -> graph 0, and then use it repeatedly.
        batch_idx = torch.zeros(N, dtype=torch.long, device=obs.device)
        edge_index = self.edge_index.to(obs.device)
        edge_attr = self.edge_attr.to(obs.device)

        # Collect values for each timestep
        values = []
        for t in range(T):
            x_t = obs[:, t]           # [B, N, D]
            vt = []
            for b in range(B):
                node_feats = x_t[b]   # [N, D]
                # run the graph mixer; uses same edges for each env
                v_b = self.gm(node_feats, edge_index, edge_attr, batch_idx)
                vt.append(v_b)        # scalar per graph
            values.append(torch.stack(vt, dim=0))  # [B]
        # now values: list of T × [B] → stack → [B, T]
        return torch.stack(values, dim=1)
