# file: ReplenishmentEnv/wrapper/graph_wrapper.py
import gym
import torch
import numpy as np
from typing import Tuple


class GraphWrapper(gym.Wrapper):
    """
    Wrap the MABIM env so that instead of returning a flat obs array per agent,
    we return a dict suitable for PyG:
     - node_feats: [N_agents, feat_dim]
     - edge_index: [2,  E]
     - edge_attr:  [E,    1]  (the lead-time along that edge)
     - batch:      [N_agents] zeros
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # how many SKUs and warehouses?
        self.num_skus = len(self.env.sku_list)
        self.num_wh = self.env.supply_chain.get_warehouse_count()
        self.n_agents = self.num_skus * self.num_wh

        # build a static edge list: for each SKU, connect wh j ↔ wh j+1
        edge_list = []
        edge_attr = []
        for sku in range(self.num_skus):
            for wh in range(self.num_wh - 1):
                u = wh * self.num_skus + sku
                v = (wh + 1) * self.num_skus + sku

                # you can pull real lead-times if available,
                # here we just read the env’s vlt for that SKU/warehouse
                lead = float(self.env.agent_states["all_warehouses", "vlt", -1, sku][wh])

                # add both directions
                edge_list.append([u, v]);
                edge_attr.append([lead])
                edge_list.append([v, u]);
                edge_attr.append([lead])

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).T  # [2, E]
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # [E,1]
        # batch is always zeros (one graph)
        self.batch = None  # we'll create on the fly

    def reset(self) -> Tuple[dict, ...]:
        # flatten-wrapper, oracle, etc all have already padded obs to shape [N_agents, feat_dim]
        obs, state = self.env.reset()
        return self._make_graph_obs(obs), state

    def step(self, actions: np.ndarray) -> Tuple[dict, float, list, dict]:
        obs, reward, done, info = self.env.step(actions)
        return self._make_graph_obs(obs), reward, done, info

    def _make_graph_obs(self, obs: np.ndarray) -> dict:
        """
        obs is an (N_agents, feat_dim) array.
        We convert to node_feats tensor, and tack on edges.
        """
        node_feats = torch.tensor(obs, dtype=torch.float32)  # [N_agents, feat_dim]
        # create batch vector once
        if self.batch is None:
            self.batch = torch.zeros(self.n_agents, dtype=torch.long)
        return {
            "node_feats": node_feats,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "batch": self.batch,
        }

    # proxy through all other methods
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()
