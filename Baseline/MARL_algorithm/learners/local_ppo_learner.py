# Baseline/MARL_algorithm/learners/local_ppo_learner.py

import torch
import wandb
from components.action_selectors import categorical_entropy
from components.episode_buffer import EpisodeBatch
from modules.critics import REGISTRY as critic_registry
from modules.critics.GraphCriticWrapper import GraphCriticWrapper
from modules.critics.graph_mixer import GraphMixer
from torch.optim import Adam
from utils.rl_utils import build_gae_targets
from utils.value_norm import ValueNorm


class LocalPPOLearner:
    def __init__(self, mac, scheme, logger, args, graph_data=None):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        # Build critic
        if args.critic_type == "graph_mix":
            # 1) raw mixer
            gm = GraphMixer(
                node_dim=args.node_feat_dim,
                edge_dim=args.edge_attr_dim,
                hidden_dim=args.gnn_hidden_dim
            )
            if graph_data is None:
                raise ValueError("graph_data is required for graph_mix critic")
            self.critic = GraphCriticWrapper(gm, graph_data)
        else:
            # fallback to your other critics
            self.critic = critic_registry[args.critic_type](scheme, args)

        # combine actor + critic params
        self.params = list(self.mac.parameters()) + list(self.critic.parameters())
        self.optimiser = Adam(self.params, lr=float(args.lr))

        # optional value‐norm
        self.use_value_norm = getattr(args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=args.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        if self.args.use_individual_rewards:
            rewards = batch["individual_rewards"][:, :-1].to(batch.device)
        else:
            rewards = (
                batch["reward"][:, :-1]
                .to(batch.device)
                .unsqueeze(2)
                .repeat(1, 1, self.n_agents, 1)
            )
            if self.args.use_mean_team_reward:
                rewards /= self.n_agents
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = torch.log(torch.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        # for graph‐based critic we only have 1 “agent” per env,
        # so collapse out the real agent dimension for all critic ops:
        mask_critic = mask_agent
        rewards_critic = rewards

        if self.args.critic_type == "graph_mix":
         # pick the first (or mean) over the 1500 agents
            mask_critic = mask_agent[:, :, :1, :]  # → [B, T, 1, 1]
            rewards_critic = rewards[:, :, :1, :]  # → [B, T, 1, 1]
        # targets and advantages
        with torch.no_grad():
            if "rnn" in self.args.critic_type:
                old_values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_outs = self.critic.forward(batch, t=t)
                    old_values.append(agent_outs)
                old_values = torch.stack(old_values, dim=1)
                old_values = old_values.unsqueeze(-1)
            elif self.args.critic_type == "graph_mix":
                old_values = self.critic(batch)
                old_values = old_values.unsqueeze(-1)
            else :
                old_values = self.critic(batch)



            if self.use_value_norm:
                vshape = old_values.shape
                old_values = self.value_norm.denormalize(old_values.view(-1)).view(vshape)

        # 2) build advantages/targets with build_gae_targets
        advantages, targets = build_gae_targets(
            rewards_critic * 100,  # your scaling
            mask_critic,
            old_values,
            self.args.gamma,
            self.args.gae_lambda
        )
        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # 3) PPO mini‐epochs (critic + actor + entropy) exactly as before
        for _ in range(self.args.mini_epochs):
            # critic
            if "rnn" in self.args.critic_type:
                values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    agent_outs = self.critic.forward(batch, t=t)
                    values.append(agent_outs)
                values = torch.stack(values, dim=1)
                values = values.unsqueeze(-1)
            elif "graph_mix" in self.args.critic_type:
                values = self.critic(batch)[:, :-1]  # [B, T−1]
                values = values.unsqueeze(-1)
            else:
                values = self.critic(batch)[:, :-1]

            td_err = (values - targets.detach()) ** 2
            critic_loss = 0.5 * (td_err * mask_critic).sum() / mask_critic.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t, t_env)
                pi.append(agent_outs)
            pi = torch.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = torch.gather(pi, dim=3, index=actions)
            log_pi_taken = torch.log(pi_taken)

            ratios = torch.exp(log_pi_taken - old_logprob)
            surr1 = ratios * norm_adv
            surr2 = (
                    torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                    * norm_adv
            )
            actor_loss = (
                    -(torch.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            )

            # entropy
            entropy_loss = categorical_entropy(pi).mean(
                -1, keepdim=True
            )  # mean over agents
            entropy_loss[mask == 0] = 0  # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()

            loss = (
                    actor_loss
                    + self.args.critic_coef * critic_loss
                    - self.args.entropy_coef * entropy_loss
            )

            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

            if _ == self.args.mini_epochs - 1 and self.args.use_wandb:
                wandb.log({
                    'loss': loss.item(),
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'values': values.mean().item(),
                    'targets': targets.mean().item(),
                    'reward': rewards.mean(),
                }, step=t_env)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path, postfix=""):
        self.mac.save_models(path, postfix)
        torch.save(self.optimiser.state_dict(), f"{path}/agent_opt{postfix}.th")

    def load_models(self, path, postfix=""):
        self.mac.load_models(path, postfix)
        self.optimiser.load_state_dict(
            torch.load(f"{path}/agent_opt{postfix}.th",
                       map_location=lambda s, l: s)
        )
