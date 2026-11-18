"""Agents: tabular Q-learning plus two baselines (random, shortest-path planner)."""
from __future__ import annotations

import math
import random

import numpy as np

from .env import N_ACTIONS, N_STATES, DELTAS, SIZE, RUNNING, WarehouseEnv, idx, pos
from .maps import bfs_distance


class RandomAgent:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, s):
        return self.rng.randrange(N_ACTIONS)


class ShortestPathAgent:
    """Closed-loop BFS planner that knows the true map.

    It treats hazards as blocked cells and, from wherever it currently is,
    takes a step that reduces the distance to the goal. It is *not*
    risk-aware: it happily hugs hazards if that is shortest.
    """
    name = "Shortest-path"

    def __init__(self, layout):
        dist = bfs_distance(layout, blocked_hazards=True)
        self.policy = np.zeros(N_STATES, dtype=int)
        for s in range(N_STATES):
            r, c = pos(s)
            best, best_d = 0, math.inf
            for a, (dr, dc) in enumerate(DELTAS):
                n = (r + dr, c + dc)
                d = dist.get(n, math.inf)
                if d < best_d:
                    best, best_d = a, d
            self.policy[s] = best

    def act(self, s):
        return int(self.policy[s])


class QAgent:
    """Greedy policy over a learned Q-table (ties broken by lowest index)."""
    name = "Q-learning"

    def __init__(self, q):
        self.q = q

    def act(self, s):
        return int(np.argmax(self.q[s]))


def train_q_learning(layout, reliability, episodes=3000, seed=0, alpha=0.2, alpha_end=0.02,
                     gamma=0.98, eps_start=1.0, eps_end=0.05, q_init=None, eps_decay_frac=0.7):
    """Standard one-step Q-learning with linearly decaying epsilon-greedy
    and a linearly decaying learning rate (alpha -> alpha_end), which helps
    the noisy (slippery) settings converge to the same policy across seeds.

    Returns (Q-table, per-episode returns list).
    `reliability` may be a float, or a (low, high) tuple: then a fresh
    reliability is sampled uniformly for every episode ("randomized" training).
    `q_init` lets us fine-tune from a previously trained table (retraining study).
    """
    randomized = isinstance(reliability, (tuple, list))
    env = WarehouseEnv(layout, reliability[0] if randomized else reliability, seed=seed)
    rng = random.Random(seed + 7919)
    q = np.zeros((N_STATES, N_ACTIONS)) if q_init is None else q_init.copy()
    qs = q.tolist()  # python lists are faster for scalar updates
    returns = []
    decay_eps = max(1, int(episodes * eps_decay_frac))
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        lr = alpha - (alpha - alpha_end) * ep / max(1, episodes - 1)
        if randomized:
            env.reliability = rng.uniform(reliability[0], reliability[1])
        s = env.reset()
        total = 0.0
        while True:
            row = qs[s]
            if rng.random() < eps:
                a = rng.randrange(N_ACTIONS)
            else:
                m = max(row)
                a = row.index(m)
            s2, r, done, status, _ = env.step(a)
            total += r
            target = r if done else r + gamma * max(qs[s2])
            row[a] += lr * (target - row[a])
            s = s2
            if done or status != RUNNING:
                break
        returns.append(total)
    return np.array(qs), returns
