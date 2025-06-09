# file: Baseline/MARL_algorithm/envs/replenishment/ReplenishmentEnv.py
import gym
import torch
import numpy as np
from typing import Tuple

class GraphWrapper(gym.Wrapper):
    """
    *Pass-through* wrapper: always returns the EXACT same (obs,state) tuple
    that the upstream wrappers did, so your multi-agent stack never breaks.
    Meanwhile, on the first reset() we pull out agent_states and build
    edge_index/edge_attr/batch for your MAC to fetch via get_graph().
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._built = False
        self.edge_index = None
        self.edge_attr  = None
        self.batch      = None

    def _build_graph(self):
        # underlying MABIM env has agent_states ready after that first reset()
        num_skus = len(self.env.sku_list)
        num_wh   = self.env.supply_chain.get_warehouse_count()
        n_agents = num_skus * num_wh

        vlt = self.env.agent_states["all_warehouses", "vlt"]  # (num_wh,num_skus)
        edges, attrs = [], []
        for sku in range(num_skus):
            for wh in range(num_wh-1):
                u = wh*num_skus + sku
                v = (wh+1)*num_skus + sku
                lead = float(vlt[wh,sku])
                edges.append([u,v]); attrs.append([lead])
                edges.append([v,u]); attrs.append([lead])

        self.edge_index = torch.tensor(edges, dtype=torch.long).T    # [2, E]
        self.edge_attr  = torch.tensor(attrs, dtype=torch.float32)  # [E, 1]
        self.batch      = torch.zeros(n_agents, dtype=torch.long)   # single graph

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        # upstream returns exactly (obs,state)
        obs, state = self.env.reset()
        if not self._built:
            self._build_graph()
            self._built = True
        return obs, state

    def step(self, action: np.ndarray):
        # proxy everything
        return self.env.step(action)

    def get_graph(self) -> dict:
        assert self._built, "GraphWrapper.get_graph() called before reset()"
        return {
            "edge_index": self.edge_index,
            "edge_attr":  self.edge_attr,
            "batch":      self.batch
        }

    # pass‐throughs
    def render(self,*a,**k): return self.env.render(*a,**k)
    def close(self):       return self.env.close()
    def seed(self,*a,**k): return self.env.seed(*a,**k)
    def get_env_info(self):return self.env.get_env_info()
    def get_stats(self):   return self.env.get_stats()
