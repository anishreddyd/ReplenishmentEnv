import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from ..multiagentenv import MultiAgentEnv
from ReplenishmentEnv import make_env, GraphWrapper


class TimeLimit(GymTimeLimit):
    """Exactly like Gym’s, but reset() must expose **only** obs to our wrappers."""
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = self.env.spec.max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        assert self._elapsed_steps is not None, "Must call reset() first"
        obs, rew, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = [True] * len(obs)
        return obs, rew, done, info


class ReplenishmentEnv(MultiAgentEnv):
    """
    MARL‐side wrapper.  Internally builds:
      ObservationWrapper4OldCode → FlattenWrapper2 → OracleWrapper
      → GraphWrapper → CurriculumWrapper
    then TimeLimit, then presents a **flat** [n_agents,feat] obs to PyMARL.
    """

    def __init__(self,
                 n_agents=100,
                 task_type="Standard",
                 mode="train",
                 time_limit=1460,
                 vis_path=None,
                 **kwargs):
        # 1) build the MABIM core + wrappers (including GraphWrapper)
        update_config = {
            "action": {
                "mode":  "demand_mean_discrete",
                "space": [0.00,0.16,0.33,0.40,0.45,0.50,0.55,0.60,0.66,0.83,
                          1.00,1.16,1.33,1.50,1.66,1.83,2.00,2.16,2.33,2.50,
                          2.66,2.83,3.00,3.16,3.33,3.50,3.66,3.83,4.00,5.00,
                          6.00,7.00,9.00,12.00]
            }
        }
        env_base = make_env(
            task_type,
            wrapper_names=[
                "ObservationWrapper4OldCode",
                "FlattenWrapper2",
                "OracleWrapper",
                "GraphWrapper",  # ← stays in the chain
                "CurriculumWrapper"
            ],
            mode=mode,
            vis_path=vis_path,
            update_config=update_config
        )
        # expose base so we can reach GraphWrapper later:
        self._graph_env = self.find_graph_wrapper(env_base)

        # time limit wrapper
        horizon = env_base.config["env"]["horizon"]
        self.episode_limit = min(time_limit, horizon)
        self._env = TimeLimit(env_base, max_episode_steps=horizon)

        self.n_warehouses = self._env.n_warehouses
        self.n_agents = self._env.get_agent_count()
        self._obs = None

        self._max_act_space = max(self._env.action_space, key=lambda s: s.n)
        self._max_obs_space = max(self._env.observation_space, key=lambda s: s.shape)
        self.env_t = 0
        self.C_trajectory = None


    def reset(self):
        # now _env.reset() → (obs: np.ndarray, state: np.ndarray)
        obs, state = self._env.reset()
        # obs is guaranteed np.ndarray
        self._obs = obs
        self.env_t = 0
        self.C_trajectory = np.zeros((self.episode_limit+1,3,self.n_agents))
        return obs, state


    def step(self, actions):
        self.env_t += 1
        actions = [int(a) for a in actions]
        obs, reward, done, info = self._env.step(actions)
        self._obs = obs
        # build stats dict exactly as before
        stats = {
            "individual_rewards": np.array(reward, dtype=np.float32)/1e4,
            "cur_balance":        info["profit"],
            "max_in_stock_sum":   info["max_in_stock_sum"],
            "mean_in_stock_sum":  info["mean_in_stock_sum"]
        }
        for i in range(self.n_warehouses):
            stats[f"mean_in_stock_sum_store_{i+1}"] = info[f"mean_in_stock_sum_store_{i+1}"]
            stats[f"mean_excess_sum_store_{i+1}"]   = info[f"mean_excess_sum_store_{i+1}"]
            stats[f"mean_backlog_sum_store_{i+1}"]  = info[f"mean_backlog_sum_store_{i+1}"]

        return float(sum(reward))/1e4, done, stats


    # ─── MARL interface ──────────────────────────────────────────────────────

    def get_obs(self):
        assert isinstance(self._obs, np.ndarray)
        assert not np.isnan(self._obs).any()
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self._obs.shape[1]

    def get_state(self):
        return self._obs.reshape(-1).astype(np.float32)

    def get_state_size(self):
        return self.get_state().shape[0]

    def get_avail_actions(self):
        out = []
        for i in range(self.n_agents):
            valid   = [1]*self._env.action_space[i].n
            invalid = [0]*(self._max_act_space.n - len(valid))
            out.append(valid+invalid)
        return out

    def get_total_actions(self):
        return self._max_act_space.n


    def find_graph_wrapper(self, env) -> GraphWrapper:
        """Walk .env until you hit your GraphWrapper."""
        cur = env
        while True:
            if isinstance(cur, GraphWrapper):
                return cur
            if not hasattr(cur, "env"):
                raise RuntimeError("no GraphWrapper in your env chain!")
            cur = cur.env

    # ─── Graph access for GraphMAC ───────────────────────────────────────────

    # def get_graph(self):
    #     """
    #     Walk down the TimeLimit → CurriculumWrapper → … chain
    #     until we find the GraphWrapper instance and call its get_graph().
    #     """
    #     w = self._env.env   # TimeLimit.env → top of make_env chain (CurriculumWrapper)
    #     while hasattr(w, "env") and not hasattr(w, "get_graph"):
    #         w = w.env
    #     assert hasattr(w, "get_graph"), "GraphWrapper missing"
    #     return w.get_graph()

        # expose graph to GraphMAC
    def get_graph(self):
        return self._graph_env.get_graph()

    def get_env_info(self):
        # Lazily call reset() once so that self._obs is set
        if getattr(self, "_obs", None) is None:
            obs, _ = self._env.reset()
            self._obs = obs
        return super().get_env_info()

    # ─── pass-throughs ────────────────────────────────────────────────────────

    def render(self):       return self._env.render()
    def close(self):        return self._env.close()
    def seed(self):         return self._env.seed()
    def get_stats(self):    return {}
    def switch_mode(self, m):return self._env.switch_mode(m)
    def get_profit(self):   return self._env.per_balance.copy()
    def set_C_trajectory(self, C): return self._env.set_C_trajectory(C)
    def set_local_SKU(self, sku):   return self._env.set_local_SKU(sku)
    def get_C_trajectory(self):    return self.C_trajectory
    def visualize_render(self, p):  return self._env.render()
