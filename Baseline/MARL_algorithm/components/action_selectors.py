import torch
import pdb
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .epsilon_schedules import DecayThenFlatSchedule


class GumbelSoftmax(OneHotCategorical):
    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -torch.log(-torch.log(U + self.eps))

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return torch.softmax(y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()


def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy()


REGISTRY = {}


class GumbelSoftmaxMultinomialActionSelector:
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, "save_probs", False)

    def select_action(self, agent_logits, avail_actions, t_env, test_mode=False):
        masked_policies = agent_logits.clone()
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = GumbelSoftmax(logits=masked_policies).sample()
            picked_actions = torch.argmax(picked_actions, dim=-1).long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions


REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector


from torch.distributions import Categorical
import torch

class MultinomialActionSelector:
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, "save_probs", False)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """
        agent_inputs: Tensor of shape [B, A, n_actions] (or [batch*agents, n_actions])
        avail_actions: same shape, with 0/1 masks
        """
        # existing debug for zero-avail rows
        zero_avail = (avail_actions.sum(dim=-1) == 0)
        num_zero = zero_avail.sum().item()
        if num_zero > 0:
            print(f"[DEBUG][t_env={t_env}] {num_zero} agents have ALL-ZERO avail_actions!")
            bad_idx = zero_avail.nonzero(as_tuple=False)
            print("  → bad_idx:", bad_idx.tolist())
            print("  → their logits:", agent_inputs[zero_avail])

        # mask out unavailable
        policies = agent_inputs.clone()
        policies[avail_actions == 0] = 0

        # normalize to valid distribution
        policies = policies / (policies.sum(dim=-1, keepdim=True) + 1e-8)

        if test_mode and self.test_greedy:
            picked_actions = policies.argmax(dim=-1)
        else:
            # epsilon adjustment
            self.epsilon = self.schedule.eval(t_env)
            counts = avail_actions.sum(dim=-1, keepdim=True) + 1e-8
            uniform = avail_actions / counts
            policies = (1 - self.epsilon) * policies + self.epsilon * uniform
            policies[avail_actions == 0] = 0

            # *** catch any rows where everything is zero ***
            zero_rows = (policies.sum(dim=-1) == 0)
            if zero_rows.any():
                fallback = torch.ones_like(policies)
                policies[zero_rows, :] = fallback[zero_rows, :]

            # renormalize so each row sums to 1
            policies = policies / policies.sum(dim=-1, keepdim=True)

            # now safe to sample
            picked_actions = Categorical(policies).sample().long()

        if self.save_probs:
            return picked_actions, policies
        else:
            return picked_actions

# register selector
REGISTRY["multinomial"] = MultinomialActionSelector



def categorical_entropy(probs):
    assert probs.size(-1) > 1
    return Categorical(probs=probs).entropy()


class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "test_noise", 0.0)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")  # should never be selected!

        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = (
            pick_random * random_actions
            + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        )
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class GaussianActionSelector:
    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # Expects the following input dimensions:
        # mu: [b x a x u]
        # sigma: [b x a x u x u]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(
            -1, self.args.n_agents, self.args.n_actions, self.args.n_actions
        )

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = torch.distributions.MultivariateNormal(
                mu.view(-1, mu.shape[-1]), sigma.view(-1, mu.shape[-1], mu.shape[-1])
            )
            try:
                picked_actions = dst.sample().view(*mu.shape)
            except Exception as e:
                a = 5
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector
