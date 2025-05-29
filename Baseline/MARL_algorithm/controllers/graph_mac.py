import torch

from ..components.action_selectors import REGISTRY as action_REGISTRY
from ..controllers.mf_gnn_actor_critic import MF_GNN_ActorCritic


class GraphMAC:
    def __init__(self, scheme, groups, args):
        self.model = MF_GNN_ActorCritic(
            node_feat_dim = args.node_feat_dim,
            edge_dim      = args.edge_attr_dim,
            gnn_hidden_dim= args.gnn_hidden_dim,
            n_actions     = args.n_actions
        )
        self.n_actions = args.n_actions
        # you’ll also want an action selector for sampling vs. greedy:
        self.action_selector = action_REGISTRY[args.action_selector](args)

    def select_actions(self, batch, t_ep, t_env, bs, test_mode=False):
        # 1) pull graph out cleanly
        graph = batch["obs"][t_ep]
        x = graph["node_feats"]
        edge_idx = graph["edge_index"]
        edge_attr = graph["edge_attr"]
        bch = graph["batch"]

        # 2) run GNN
        logits, _ = self.model(x, edge_idx, edge_attr, bch)

        # 3) mask out invalid actions from the runner’s avail_actions
        avail = batch["avail_actions"][t_ep].view(-1, self.n_actions).to(logits.device)
        logits[avail == 0] = -1e10

        # 4) softmax → flat [N, A]
        probs = torch.softmax(logits, dim=-1)

        # 5) select flat
        chosen_flat = self.action_selector.select_action(
            probs,  # [N, A]
            avail,  # [N, A]
            t_env,
            test_mode
        )  # returns [N] long

        # 6) reshape to [batch_size, n_agents]
        B = bch.max().item() + 1  # number of graphs / environments
        N = chosen_flat.size(0)
        A = N // B  # agents per graph
        return chosen_flat.view(B, A)
