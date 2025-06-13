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

        # — build critic —
        if args.critic_type == "graph_mix":
            if graph_data is None:
                raise ValueError("graph_data is required for graph_mix critic")
            mixer = GraphMixer(
                node_dim=args.node_feat_dim,
                edge_dim=args.edge_attr_dim,
                hidden_dim=args.gnn_hidden_dim
            )
            self.critic = GraphCriticWrapper(mixer, graph_data)
        else:
            self.critic = critic_registry[args.critic_type](scheme, args)

        # — split actor & critic optimizers —
        actor_lr = float(getattr(args, "actor_lr", args.lr))
        critic_lr = float(getattr(args, "critic_lr", args.lr))
        optim_alpha = float(getattr(args, "optim_alpha", 0.99))
        optim_eps = float(getattr(args, "optim_eps", 1e-5))

        self.actor_optimizer = Adam(
            self.mac.parameters(),
            lr=actor_lr,
            betas=(optim_alpha, 0.999),
            eps=optim_eps
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=(optim_alpha, 0.999),
            eps=optim_eps
        )

        # optional value‑norm
        self.use_value_norm = getattr(args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=args.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Prepare rewards and masks
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

        # GraphMix critic adjustments
        mask_critic = mask_agent
        rewards_critic = rewards
        if self.args.critic_type == "graph_mix":
            mask_critic = mask_agent[:, :, :1, :]
            rewards_critic = rewards[:, :, :1, :]

        # Compute baseline values
        with torch.no_grad():
            if "rnn" in self.args.critic_type:
                old_values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    old_values.append(self.critic.forward(batch, t=t))
                old_values = torch.stack(old_values, dim=1).unsqueeze(-1)
            elif self.args.critic_type == "graph_mix":
                # GraphMix critic returns [B, T]; expand to [B, T, 1]
                old_values = self.critic(batch).unsqueeze(-1)
            else:
                old_values = self.critic(batch)

            if self.use_value_norm:
                vshape = old_values.shape
                old_values = self.value_norm.denormalize(old_values.view(-1)).view(vshape)

        # Build advantages and targets with safe scaling and clamping
        with torch.no_grad():
            advantages, targets = build_gae_targets(
                rewards_critic * self.args.reward_scale,
                mask_critic,
                old_values,
                self.args.gamma,
                self.args.gae_lambda
            )
        print(f"[DEBUG] old_values shape: {tuple(old_values.shape)}", flush=True)
        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        norm_adv = torch.clamp(norm_adv, -self.args.max_adv, self.args.max_adv)

        # PPO mini‑epochs
        for epoch in range(self.args.mini_epochs):
            # Critic update
            if "rnn" in self.args.critic_type:
                values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    values.append(self.critic.forward(batch, t=t))
                values = torch.stack(values, dim=1).unsqueeze(-1)
            elif self.args.critic_type == "graph_mix":
                # critic gives [B, T]; trim and expand dims
                values = self.critic(batch)[:, :-1].unsqueeze(-1)
            else:
                values = self.critic(batch)[:, :-1]

            td_err = (values - targets.detach()) ** 2
            critic_loss = 0.5 * (td_err * mask_critic).sum() / mask_critic.sum()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.critic_clip)
            self.critic_optimizer.step()

            # ----- debug gradients and parameters for the actor -----
            for n, p in self.mac.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"[DEBUG] NaN/Inf grad in actor param {n}", flush=True)
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"[DEBUG] NaN/Inf value in actor param {n} BEFORE update", flush=True)

            # Actor + entropy update
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                pi.append(self.mac.forward(batch, t, t_env))
            pi = torch.stack(pi, dim=1)

            # Guard against invalid and NaN probabilities
            pi[avail_actions == 0] = 1e-10
            pi = pi + 1e-10
            pi = pi / pi.sum(-1, keepdim=True)

            pi_taken = torch.gather(pi, dim=3, index=actions)
            log_pi = torch.log(pi_taken)

            ratios = torch.exp(log_pi - old_logprob)
            ratios = torch.clamp(ratios, 0.0, self.args.max_ratio)

            surr1 = ratios * norm_adv
            surr2 = (torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * norm_adv)
            actor_loss = - (torch.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()

            entropy = categorical_entropy(pi).mean(-1, keepdim=True)
            entropy[mask == 0] = 0
            entropy_loss = (entropy * mask).sum() / mask.sum()

            total_actor_loss = actor_loss - self.args.entropy_coef * entropy_loss

            # Logging for debugging stability
            print(f"[PPO] epoch={epoch} actor_loss={actor_loss.item():.3e} "
                  f"ent_loss={entropy_loss.item():.3e} adv_mean={norm_adv.mean():.3e} "
                  f"adv_std={norm_adv.std():.3e}")

           self.actor_optimizer.zero_grad()
           total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mac.parameters(), self.args.actor_clip)
            for n, p in self.mac.named_parameters():
                if p.grad is not None:
                    print(f"[DEBUG] grad {n} norm {p.grad.norm().item():.4f}", flush=True)
            self.actor_optimizer.step()

            # Debug parameter stats
            for n, p in self.mac.named_parameters():
                if p.requires_grad:
                    print(f"[DEBUG] param {n} norm {p.norm().item():.4f} "
                          f"min {p.min().item():.4f} max {p.max().item():.4f}", flush=True)

            # ----- check actor parameters for NaN/Inf after update -----
            for n, p in self.mac.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"[DEBUG] NaN/Inf value in actor param {n} AFTER update", flush=True)

            # Optional wandb logging
            if epoch == self.args.mini_epochs - 1 and self.args.use_wandb:
                wandb.log({
                    'critic_loss': critic_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'values': values.mean().item(),
                    'targets': targets.mean().item(),
                    'reward': rewards.mean(),
                }, step=t_env)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path, postfix=""):
        self.mac.save_models(path, postfix)
        torch.save(self.actor_optimizer.state_dict(), f"{path}/actor_opt{postfix}.th")
        torch.save(self.critic_optimizer.state_dict(), f"{path}/critic_opt{postfix}.th")

    def load_models(self, path, postfix=""):
        self.mac.load_models(path, postfix)
        self.actor_optimizer.load_state_dict(
            torch.load(f"{path}/actor_opt{postfix}.th", map_location=lambda s, l: s)
        )
        self.critic_optimizer.load_state_dict(
            torch.load(f"{path}/critic_opt{postfix}.th", map_location=lambda s, l: s)
        )
