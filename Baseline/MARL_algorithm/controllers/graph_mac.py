# file: Baseline/MARL_algorithm/controllers/graph_mac.py
import torch, torch.nn as nn
from ..components.action_selectors import REGISTRY as action_REGISTRY
from ..controllers.mf_gnn_actor_critic import MF_GNN_ActorCritic

class GraphMAC(nn.Module):
    def __init__(self, env, scheme, groups, args):
        super().__init__()
        self.env       = env
        self.n_actions = args.n_actions

        self.model = MF_GNN_ActorCritic(
            node_feat_dim = args.node_feat_dim,
            edge_dim      = args.edge_attr_dim,
            gnn_hidden_dim= args.gnn_hidden_dim,
            n_actions     = args.n_actions
        )
        self.action_selector = action_REGISTRY[args.action_selector](args)

    def init_hidden(self, batch_size):
        """
        Called by the runner at the start of each episode.
        If your GNN actor‐critic has no recurrent state, this can be a no‐op.
        """
        # e.g. if your MF_GNN_ActorCritic had a recurrent core:
        # self.model.init_hidden(batch_size * n_agents)
        pass

    def forward(self, batch, t, t_env, test_mode=False):
        """
        Called during training to get π(a|s) for all agents.
        Must return a float‐tensor of shape [B, A, n_actions].
        """
        # 1) grab per‐timestep obs & avail
        #    batch["obs"] is [B, T, A, feat]
        obs_t = batch["obs"][:, t]  # [B, A, feat]
        avail_t = batch["avail_actions"][:, t]  # [B, A, n_actions]

        B, A, F = obs_t.shape
        x = obs_t.reshape(B * A, F)  # flatten to [N, feat], N = B*A

        # 2) get graph once
        graph = self.env.get_graph()
        edge_index = graph["edge_index"].to(x.device)  # [2, E]
        edge_attr = graph["edge_attr"].to(x.device)  # [E, edge_dim]
        # single‐graph -> batch_idx all zeros
        batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 3) run through your GNN actor
        logits, _ = self.model(x, edge_index, edge_attr, batch_idx)  # [N, n_actions]
        logits = logits.view(B, A, -1)  # [B, A, n_actions]

        # 4) mask unavailable
        mask = avail_t.to(logits.device)
        logits = logits.masked_fill(mask == 0, -1e10)

        # 5) produce π‐distribution
        pi = torch.softmax(logits, dim=-1)  # [B, A, n_actions]
        return pi


    def select_actions(self, batch, t_ep, t_env, bs, test_mode=False):
        """
        For each parallel env index in bs we:
          1) pull out its [A, feat] obs,
          2) run one small GNN (same graph structure) to get [A, actions],
          3) mask + softmax + sample → [A],
        then stack back to [batch_size_run, A].

        bs: a list of the indices of the parallel envs that are still running
        """
        # 1) grab the full batched obs & avail masks, then select only those in bs
        # full_obs = batch["obs"][t_ep]  # [n_lambda, n_agents, feat]
        # full_avail = batch["avail_actions"][t_ep]  # [n_lambda, n_agents, actions]
        full_obs = batch["obs"][:, t_ep]  # [batch_size_run, n_agents, feat]
        full_avail = batch["avail_actions"][:, t_ep]  # [batch_size_run, n_agents, n_actions]
        obs_baft = full_obs[bs]  # [batch_size_run, n_agents, feat]
        avail_baft = full_avail[bs]  # [batch_size_run, n_agents, actions]

        # 2) single‐graph metadata (identical for each env)
        g = self.env.get_graph()
        edge_idx = g["edge_index"]
        edge_attr = g["edge_attr"]

        all_chosen = []
        batch_size_run, A, _ = obs_baft.shape

        for batch_idx, env_i in enumerate(bs):
            # pull out one env’s nodes by batch index
            x_i = obs_baft[batch_idx].view(-1, obs_baft.size(-1))  # [A, F]
            batch_i = torch.zeros(x_i.size(0), dtype=torch.long, device=x_i.device)

            # forward through GNN
            logits_i, _ = self.model(x_i, edge_idx, edge_attr, batch_i)  # [A, n_actions]

            # grab this env’s avail mask by batch index
            avail_i = avail_baft[batch_idx].to(logits_i.device)  # [A, n_actions]

            # DEBUG #2: per‐env mask sums
            sums_i = avail_i.sum(dim=-1)
            print(f"[GraphMAC] env={env_i} (batch_idx={batch_idx}) avail_i.shape={tuple(avail_i.shape)}, "
                  f"sums_i[:5]={sums_i[:5].tolist()}", flush=True)

            # apply mask with masked_fill
            logits_i = logits_i.masked_fill(avail_i == 0, -1e10)

            # DEBUG #3: check for NaNs in masked logits
            if torch.isnan(logits_i).any():
                nan_idx = torch.isnan(logits_i).nonzero(as_tuple=False)
                print(f"[GraphMAC] env={env_i} logits_i has NaNs at indices {nan_idx[:5]}", flush=True)

            # softmax → probabilities
            probs_i = torch.softmax(logits_i, dim=-1)

            # DEBUG #4: check for NaNs in probs
            if torch.isnan(probs_i).any():
                print(f"[GraphMAC] env={env_i} probs_i has NaNs", flush=True)

            # sample one action per agent
            chosen_i = self.action_selector.select_action(
                probs_i, avail_i, t_env, test_mode
            )  # → [A]

            all_chosen.append(chosen_i)

        # stack back to [batch_size_run, A]
        result = torch.stack(all_chosen, dim=0)
        print(f"[GraphMAC] returning result.shape={tuple(result.shape)}", flush=True)
        return result
