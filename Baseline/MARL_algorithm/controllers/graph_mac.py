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
        obs_t = batch["obs"][:, t]  # [B, A, feat]
        avail_t = batch["avail_actions"][:, t]  # [B, A, n_actions]
        B, A, F = obs_t.shape
        x = obs_t.reshape(B * A, F)
        g = self.env.get_graph()
        ei = g["edge_index"].to(x.device)
        ea = g["edge_attr"].to(x.device)
        batch_idx = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        logits, _ = self.model(x, ei, ea, batch_idx)
        logits = logits.view(B, A, -1)
        # clamp and sanitize logits to avoid NaNs/Infs after updates
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        logits = logits.clamp(-1e6, 1e6)

        # —— DEBUG HERE ——
        print(f"[GraphMAC‐TRAIN] t={t} logits.min={logits.min().item():.3f}, "
              f"max={logits.max().item():.3f}, mean={logits.mean().item():.3f}", flush=True)

        mask = avail_t.to(logits.device)
        logits = logits.masked_fill(mask == 0, -1e10)

        if torch.isnan(logits).any():
            idx = torch.isnan(logits).nonzero(as_tuple=False)[:5]
            print(f"[GraphMAC‐TRAIN][ERROR] logits NaN at {idx.tolist()}", flush=True)

        pi = torch.softmax(logits, dim=-1)
        pi = torch.nan_to_num(pi, nan=1e-10)
        pi = pi / pi.sum(dim=-1, keepdim=True)
        if torch.isnan(pi).any():
            idx = torch.isnan(pi).nonzero(as_tuple=False)[:5]
            print(f"[GraphMAC‐TRAIN][ERROR] pi NaN at {idx.tolist()}", flush=True)
        else:
            print(
                f"[GraphMAC-TRAIN] pi stats: min={pi.min().item():.3f}, max={pi.max().item():.3f}",
                flush=True,
            )

        return pi

    def select_actions(self, batch, t_ep, t_env, bs, test_mode=False):
        """
        batch["obs"]:         [B, T, A, F]
        batch["avail_actions"]:[B, T, A, n_actions]
        bs: indices of envs still running
        """
        B, A, F = batch["obs"].shape[0], batch["obs"].shape[2], batch["obs"].shape[3]
        full_obs   = batch["obs"][:, t_ep]        # [B, A, F]
        full_avail = batch["avail_actions"][:, t_ep]  # [B, A, n_actions]

        # 1) UP-FRONT: any entire env with zero avail for ALL its agents?
        zero_envs = (full_avail.sum(dim=-1)==0).all(dim=-1).nonzero(as_tuple=False).flatten()
        if zero_envs.numel() > 0:
            print(f"[GraphMAC][FATAL] t={t_ep}, envs with zero-avail for all agents: {zero_envs.tolist()}", flush=True)
            for e in zero_envs.tolist():
                print(f"    avail_actions[{e},:,:] =\n{full_avail[e].cpu().numpy()}", flush=True)
            raise RuntimeError("Encountered env(s) with zero legal actions for every agent.")

        # 2) slice down to only the still-running envs
        obs_baft   = full_obs[bs]    # [len(bs), A, F]
        avail_baft = full_avail[bs]  # [len(bs), A, n_actions]

        # 3) static graph metadata
        g = self.env.get_graph()
        edge_idx = g["edge_index"].to(full_obs.device)
        edge_attr = g["edge_attr"].to(full_obs.device)

        all_chosen = []
        for batch_idx, env_i in enumerate(bs):
            x_i      = obs_baft   [batch_idx].reshape(-1, F)       # [A, F]
            batch_i  = torch.zeros(x_i.size(0), dtype=torch.long, device=x_i.device)

            logits_i, _ = self.model(x_i, edge_idx, edge_attr, batch_i)  # [A, n_actions]
            logits_i    = logits_i.view(A, self.n_actions)

            avail_i = avail_baft[batch_idx].to(logits_i.device)         # [A, n_actions]

            # DEBUG: mask coverage & raw logits stats
            pct_masked = 100 * (avail_i==0).float().mean().item()
            print(f"[GraphMAC][DEBUG] env={env_i} mask coverage: {pct_masked:.1f}% zeroed", flush=True)
            print(f"[GraphMAC][DEBUG] env={env_i} logits before mask: min={logits_i.min().item():.3f}, max={logits_i.max().item():.3f}, mean={logits_i.mean().item():.3f}", flush=True)

            # 4) AGENT-LEVEL assert: each agent must have at least one legal move
            per_agent_sum = avail_i.sum(dim=-1)        # [A]
            zero_agents   = (per_agent_sum==0).nonzero(as_tuple=False).flatten()
            if zero_agents.numel() > 0:
                print(f"[GraphMAC][FATAL] env={env_i} these agent indices have NO legal actions: {zero_agents.tolist()}", flush=True)
                for a in zero_agents.tolist():
                    print(f"    agent={a} avail[{env_i},{a},:] = {avail_i[a].cpu().numpy()}", flush=True)
                raise RuntimeError(f"Found agent(s) with zero legal actions in env {env_i} at t={t_ep}")

            # 5) mask out illegal logits
            logits_i = logits_i.masked_fill(avail_i==0, -1e10)

            # DEBUG: masked logits stats
            print(f"[GraphMAC][DEBUG] env={env_i} logits after mask: min={logits_i.min().item():.3f}, max={logits_i.max().item():.3f}, mean={logits_i.mean().item():.3f}", flush=True)

            # 6) softmax → probabilities
            probs_i = torch.softmax(logits_i, dim=-1)
            probs_i = torch.nan_to_num(probs_i, nan=1e-10)
            probs_i = probs_i / probs_i.sum(dim=-1, keepdim=True)
            prob_sums = probs_i.sum(dim=-1)  # should all be ~1
            print(f"[GraphMAC][DEBUG] env={env_i} probs sum first 5 agents: {prob_sums[:5].tolist()}", flush=True)

            if torch.isnan(probs_i).any():
                nan_idx = torch.isnan(probs_i).nonzero(as_tuple=False)
                print(f"[GraphMAC][FATAL] env={env_i} got NaNs in softmax probs at {nan_idx[:10].tolist()}", flush=True)
                raise RuntimeError("NaNs in π(a|s) probability distribution!")

            # 7) sample
            chosen_i = self.action_selector.select_action(probs_i, avail_i, t_env, test_mode)
            all_chosen.append(chosen_i)

        result = torch.stack(all_chosen, dim=0)  # [len(bs), A]
        print(f"[GraphMAC] returning result.shape={tuple(result.shape)}", flush=True)
        return result

