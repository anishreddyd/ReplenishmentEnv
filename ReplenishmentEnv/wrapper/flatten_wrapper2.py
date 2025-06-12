import numpy as np
import gym
from typing import Tuple, Any

class FlattenWrapper2(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # compute these once
        self.sku_count      = len(self.env.sku_list)
        self.warehouse_count= self.env.supply_chain.get_warehouse_count()
        self.agent_count    = self.sku_count * self.warehouse_count

    def reset(self) -> Any:
        ret = self.env.reset()
        # assume the first element is `obs`
        obs, *rest = ret if isinstance(ret, tuple) else (ret, [])
        print(f"[FlattenWrapper] reset: before shape = {obs.shape}")
        flat_obs = obs.reshape(self.agent_count, -1)
        print(f"[FlattenWrapper] reset: flattened shape = {flat_obs.shape}")
        # re-assemble the tuple
        return (flat_obs, *rest) if rest else flat_obs

    def step(self, actions: np.ndarray) -> Any:
        # reshape actions back into env’s expected shape
        actions = np.array(actions).reshape(self.warehouse_count, self.sku_count)
        ret = self.env.step(actions)
        obs, *rest = ret
        flat_obs = obs.reshape(self.agent_count, -1)
        rewards, done, info = rest
        # if rewards/info need flattening, do it here:
        rewards = np.array(rewards).flatten()
        if isinstance(info, dict) and 'profit' in info:
            info['profit'] = np.array(info['profit']).flatten()
        return (flat_obs, rewards, done, info)
