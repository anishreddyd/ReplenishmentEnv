# file: ReplenishmentEnv/wrapper/curriculum_wrapper.py
import gym
import numpy as np
from typing import Tuple, List, AnyStr, Any


class CurriculumWrapper(gym.Wrapper):
    """
    Every `every_episodes` episodes, increase the number of SKUs in the ReplenishmentEnv
    via its `set_local_SKU` method, until we hit `max_sku`.
    """
    def __init__(
        self,
        env: gym.Env,
        start_sku: int        = 200,
        step: int             = 200,
        max_sku: int          = 2000,
        every_episodes: int   = 10,
    ):
        super().__init__(env)
        self.current_sku    = start_sku
        self.step           = step
        self.max_sku        = max_sku
        self.every_episodes = every_episodes
        self.episode_count  = 0

        # initialize the inner env’s SKU count
        if hasattr(self.env, "set_local_SKU"):
            self.env.set_local_SKU(self.current_sku)

    def reset(self) -> Tuple[Any, Any]:
        # bump the counter
        self.episode_count += 1

        # every `every_episodes` episodes, increase SKUs
        if (self.episode_count % self.every_episodes == 0
            and self.current_sku < self.max_sku
            and hasattr(self.env, "set_local_SKU")
        ):
            self.current_sku = min(self.current_sku + self.step, self.max_sku)
            self.env.set_local_SKU(self.current_sku)

        return self.env.reset()

    def step(self, action: np.ndarray):
        return self.env.step(action)

    # proxy everything else so nothing breaks:
    def render(self, *args, **kwargs):    return self.env.render(*args, **kwargs)
    def close(self):                      return self.env.close()
    def seed(self, *args, **kwargs):     return self.env.seed(*args, **kwargs)
    def get_env_info(self):               return self.env.get_env_info()
    def get_stats(self):                 return self.env.get_stats()
