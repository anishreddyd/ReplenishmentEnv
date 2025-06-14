from Baseline. MARL_algorithm. controllers. graph_mac import GraphMAC

REGISTRY = {}

from .basic_controller import BasicMAC
from .mappo_controller import MAPPOMAC
from .dqn_controller import DQNMAC
from .ldqn_controller import LDQNMAC
from .whittle_disc_controller import WhittleDiscreteMAC
from .whittle_cont_controller import WhittleContinuousMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mappo_mac"] = MAPPOMAC
REGISTRY["dqn_mac"] = DQNMAC
REGISTRY["ldqn_mac"] = LDQNMAC
REGISTRY["whittle_disc_mac"] = WhittleDiscreteMAC
REGISTRY["whittle_cont_mac"] = WhittleContinuousMAC
REGISTRY["graph_mac"]    = GraphMAC
