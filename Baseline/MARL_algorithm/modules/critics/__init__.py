from .GraphCriticWrapper import GraphCriticWrapper
from .coma import COMACritic
from .graph_mixer import GraphMixer
from .mappo_rnn_critic import MAPPORNNCritic
from .mappo_rnn_critic_share import MAPPORNNCriticShare

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["mappo_rnn_critic"] = MAPPORNNCritic
REGISTRY["mappo_rnn_critic_share"] = MAPPORNNCriticShare
REGISTRY["graph_mix"] = lambda args: GraphCriticWrapper(GraphMixer(...))
