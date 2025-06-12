import os
import pdb
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np
import torch.nn.functional as F
import torch

import wandb
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from utils.timehelper import TimeStat


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Grab the env-factory and its args
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]

        # === Local env for inspection ===
        # Use the same kwargs you'll give to workers, wrapped fully
        self.env = env_fn(**env_args[0])
        self.env.reset()
        self.env_fn = env_fn

        # Setup worker pipes and processes
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        for i in range(len(env_args)):
            env_args[i]["seed"] += i
        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        # Reset each worker once so that env._obs is populated
        for pc in self.parent_conns:
            pc.send(("reset", None))
        for pc in self.parent_conns:
            _ = pc.recv()
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def _get_all_avail_actions(self):
        """
        Ask every worker for its avail_actions mask, then collect them.
        Returns a list of (n_agents × n_actions) arrays, one per env.
        """
        for conn in self.parent_conns:
            conn.send({"cmd": "get_avail_actions"})
        # now receive from each
        return [conn.recv() for conn in self.parent_conns]

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False, storage_capacity=None):

        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        # Get the obs, state and avail_actions back

        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(np.zeros([1, self.args.n_agents, self.args.n_actions]))


        print(
            ">>> pre-transition_data['avail_actions'] shapes & sums:",
            [
                (
                    type(a),
                    np.array(a).shape,
                    np.array(a).sum()
                )
                for a in pre_transition_data["avail_actions"]
            ]
        )


         # Write into the batch for ts=0 and mark these as filled
        self.batch.update(pre_transition_data, ts=0, mark_filled=True)

        batch_masks = self.batch["avail_actions"]  # shape [batch_size, T+1, n_agents, n_actions]
        sums_at_t0 = batch_masks[:, 0]  # [batch_size, n_agents, n_actions]
        sums_at_t0 = sums_at_t0.reshape(self.batch_size, -1).sum(dim=1).tolist()
        print(">>> BATCH after update, sums at t=0:", sums_at_t0)

        self.t = 0
        self.env_steps_this_run = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []

        if storage_capacity is not None:
            for parent_conn in self.parent_conns:
                parent_conn.send(("set_storage_capacity", storage_capacity))

    def run(self, lbda_index=None, test_mode=False,
            visual_outputs_path=None, storage_capacity=None):

        self.reset(test_mode=test_mode, storage_capacity=storage_capacity)

        all_terminated = False
        episode_returns = np.zeros([self.batch_size, self.args.n_lambda])
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_balance = [0 for _ in range(self.batch_size)]
        if self.args.use_n_lambda:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents, self.args.n_lambda])
        else:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents])

        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = []  # may store extra stats like battle won.

        save_probs = getattr(self.args, "save_probs", False)

        while True:

            if self.args.mac == "mappo_mac" or self.args.mac == "graph_mac":
                mac_output = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env,
                    bs=envs_not_terminated, test_mode=test_mode
                )
            elif self.args.mac in ("dqn_mac", "ldqn_mac"):
                mac_output = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env,
                    lbda_indices=None, bs=envs_not_terminated, test_mode=test_mode
                )

            if save_probs:
                actions, probs = mac_output
            else:
                actions = mac_output

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1).to("cpu")}
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu").detach()

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated and not terminated[idx]:
                    parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            if all(terminated):
                break

            post_transition_data = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            }
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )
                    post_transition_data["cur_balance"].append(
                        data["info"]["cur_balance"]
                    )

                    episode_returns[idx] += data["reward"]
                    if self.args.n_agents > 1:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"]
                    else:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"][0]

                    episode_balance[idx] = data["info"]["cur_balance"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["mean_action"].append(
                        F.one_hot(actions[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )

            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False
            )

            self.t += 1

            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get profit for each env
        episode_profits = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        for parent_conn in self.parent_conns:
            episode_profit = parent_conn.recv()
            episode_profits.append(episode_profit / self.t * (self.episode_limit))

        # Get stats back for each env
        env_stats = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_profits = self.test_profits if test_mode else self.train_profits

        # log_prefix = "test_" if test_mode else ""
        if test_mode:
            log_prefix = "test" if visual_outputs_path is not None else "val"
        else:
            log_prefix = "train"
        if visual_outputs_path is not None:
            self.parent_conns[0].send(("visualize_render", visual_outputs_path))
            self.parent_conns[0].recv()
        infos = [cur_stats] + final_env_infos

        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        max_in_stock_seq = [d['max_in_stock_sum'] for d in final_env_infos]
        cur_stats['max_in_stock_sum'] = np.mean(max_in_stock_seq)

        mean_in_stock_seq = [d['mean_in_stock_sum'] for d in final_env_infos]
        cur_stats['mean_in_stock_sum'] = np.mean(mean_in_stock_seq)

        cur_returns.extend(episode_returns)
        cur_profits.extend(episode_profits)

        n_test_runs = (
                max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )

        if test_mode:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]

            cur_profits = np.array(cur_profits)
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)

            return cur_stats, lambda_return, profits
        else:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]

            cur_profits = np.array(cur_profits)
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)
            return self.batch, cur_stats, lambda_return, profits

    def get_overall_avg_balance(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        cur_balances = []
        for parent_conn in self.parent_conns:
            cur_balances.append(parent_conn.recv())

        return np.mean(np.sum(np.array(cur_balances), axis=1))

    def _log(self, returns, individual_returns, profits, stats, prefix):
        self.logger.log_stat(prefix + "_return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "_return_std", np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "_profit_mean", np.mean(profits), self.t_env)
        self.logger.log_stat(prefix + "_profit_std", np.std(profits), self.t_env)
        profits.clear()

        if self.args.use_wandb and self.args.n_agents <= 100:
            for i in range(self.args.n_agents):
                wandb.log(
                    {
                        f"SKUReturn/joint_{prefix}_sku{i + 1}_mean": individual_returns[
                                                                     :, i
                                                                     ].mean()
                    },
                    step=self.t_env,
                )

            for i in range(self.args.n_agents):
                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_reward_dict", None))
                reward_dicts = []
                for parent_conn in self.parent_conns:
                    reward_dicts.append(parent_conn.recv())

                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_profit", None))
                cur_balances = []
                for parent_conn in self.parent_conns:
                    cur_balances.append(parent_conn.recv())
                wandb.log(
                    {
                        f"SKUReturn_{k}/joint_{prefix}_sku{i + 1}_mean": np.mean(
                            [np.array(rd[k])[:, i].sum() / 1e6 for rd in reward_dicts]
                        )
                        for k in reward_dicts[0].keys()
                    },
                    step=self.t_env,
                )
                wandb.log(
                    {
                        f"SKUBalance/joint_{prefix}_sku{i + 1}_mean": np.mean(
                            np.array(cur_balances)[:, i]
                        )
                    },
                    step=self.t_env,
                )
            wandb.log(
                {
                    f"SumBalance/joint_{prefix}_sum": np.mean(
                        np.sum(np.array(cur_balances), 1)
                    )
                },
                step=self.t_env,
            )

        if self.args.use_wandb:
            wandb.log(
                {
                    f"instock_sum/{prefix}_max_in_stock_sum_mean": stats['max_in_stock_sum_mean'],
                    f"instock_sum/{prefix}_max_in_stock_sum_min": stats['max_in_stock_sum_min'],
                    f"instock_sum/{prefix}_max_in_stock_sum_max": stats['max_in_stock_sum_max'],
                },
                step=self.t_env,
            )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_mean", stats['max_in_stock_sum_mean'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_min", stats['max_in_stock_sum_min'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_max", stats['max_in_stock_sum_max'], self.t_env
        )

        for k, v in stats.items():
            if k not in ["n_episodes", "individual_rewards"]:
                self.logger.log_stat(
                    prefix + "_" + k + "_mean", v / stats["n_episodes"], self.t_env
                )

        stats.clear()


def env_worker(remote, env_fn):
    pid = os.getpid()
    print(f"[env_worker {pid}] starting with wrapper stack", flush=True)
    # real_env = env_fn.x()
    env = env_fn.x()
    step_count = 0

    while True:
        cmd, data = remote.recv()
        print(f"[env_worker {pid}] got cmd '{cmd}'", flush=True)

        if cmd == "reset":
            env.reset()
            masks = env.get_avail_actions()
            zero_agents = [i for i, m in enumerate(masks) if sum(m) == 0]
            if zero_agents:
                print(f"   → [reset] zero idx (first 10): {zero_agents[:10]}", flush=True)
                # Fallback: allow all actions for those agents
                n_actions = len(masks[0])
                for i in zero_agents:
                    masks[i] = [1] * n_actions
                print(f"   → [reset] fallback masks set to all ones for zero-avail agents", flush=True)
            remote.send({
                "state": env.get_state(),
                "avail_actions": masks,
                "obs": env.get_obs(),
            })

        elif cmd == "step":
            actions = data
            reward, terminated, info = env.step(actions)
            masks = env.get_avail_actions()
            zero_agents = [i for i, m in enumerate(masks) if sum(m) == 0]
            if zero_agents:
                print(f"   → [step t={step_count}] zero idx (first 10): {zero_agents[:10]}", flush=True)
                n_actions = len(masks[0])
                for i in zero_agents:
                    masks[i] = [1] * n_actions
                print(f"   → [step] fallback masks set to all ones for zero-avail agents", flush=True)
            remote.send({
                "state": env.get_state(),
                "avail_actions": masks,
                "obs": env.get_obs(),
                "reward": reward,
                "terminated": terminated,
                "info": info,
            })
            step_count += 1

        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "switch_mode":
            env.switch_mode(data)
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "get_profit":
            remote.send(env.get_profit())
        elif cmd == "get_reward_dict":
            remote.send(env._env.reward_monitor)
        elif cmd == "visualize_render":
            env.visualize_render(data)
        elif cmd == "get_storage_capacity":
            remote.send(env._env.storage_capacity)
        elif cmd == "set_storage_capacity":
            env.set_storage_capacity(data)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class DebugEnv:
    """
    Wraps any env to intercept get_avail_actions calls and print full wrapper chain and mask stats.
    """
    def __init__(self, real_env):
        self._env = real_env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_avail_actions(self):
        # Walk the wrapper chain, printing class names and mask summaries
        chain = []
        env = self
        while hasattr(env, '_env'):
            chain.append(type(env).__name__)
            env = env._env
        chain.append(type(env).__name__)
        print(f"[WRAPPER CHAIN] {' -> '.join(chain)}", flush=True)

        masks = self._env.get_avail_actions()
        sums = [int(sum(m)) for m in masks]
        print(f"[WRAPPER DEBUG:{type(self._env).__name__}] sums[:5]={sums[:5]} (min={min(sums)})", flush=True)
        return masks