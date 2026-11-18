"""Local-view Q-learning: a tabular agent whose state is what the robot SEES,
not which cell it is in, so the same Q-table can be used on any floor plan.

Observation = one of 5 categories for each neighbour UP/RIGHT/DOWN/LEFT (5^4 = 625 states):
    0 wall / outside   1 hazard   2 goal
    3 free cell        4 free cell that is the planner's suggested next step
The suggestion comes from a risk-aware route planner on the robot's floor map:
Dijkstra where entering a cell costs 1 + RISK_WEIGHT x (hazards next to it, max 2).
The planner itself does not know the slip probability; Q-learning decides when
to trust the suggestion and when a neighbouring hazard makes another move better.
Because slips are 90 degrees, the four neighbours are exactly the cells one
action can land in, so the observation is enough to judge one-step risk.
"""
from __future__ import annotations

import heapq
import random

import numpy as np

from .env import DELTAS, N_ACTIONS, RUNNING, SIZE, WarehouseEnv, pos

N_OBS = 5 ** 4
RISK_WEIGHT = 8.0


def _inside(p):
    return 0 <= p[0] < SIZE and 0 <= p[1] < SIZE


def risk_aware_distance(layout, w=RISK_WEIGHT):
    """Cost-to-goal where entering a cell costs 1 + w * min(2, adjacent hazards)."""
    hz, walls, goal = layout["hazards"], layout["walls"], layout["goal"]

    def risk(p):
        return min(2, sum((p[0] + dr, p[1] + dc) in hz for dr, dc in DELTAS))

    d = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        enter_u = 1 + (0 if u == goal else w * risk(u))
        for dr, dc in DELTAS:
            n = (u[0] + dr, u[1] + dc)
            if not _inside(n) or n in walls or n in hz:
                continue
            nd = du + enter_u
            if nd < d.get(n, float("inf")):
                d[n] = nd
                heapq.heappush(pq, (nd, n))
    return d


def suggested_actions(layout, w=RISK_WEIGHT):
    """Planner's next move for every cell (used by the observation and by baselines)."""
    d = risk_aware_distance(layout, w)
    out = np.zeros(SIZE * SIZE, dtype=int)
    for s in range(SIZE * SIZE):
        r, c = pos(s)
        vals = []
        for dr, dc in DELTAS:
            n = (r + dr, c + dc)
            vals.append(d.get(n, float("inf")) if _inside(n) and n not in layout["walls"] else float("inf"))
        out[s] = int(np.argmin(vals))
    return out


def observation_table(layout):
    """obs[s] for every cell s of this layout (int array of length 64)."""
    walls, hazards, goal = layout["walls"], layout["hazards"], layout["goal"]
    best = suggested_actions(layout)
    obs = np.zeros(SIZE * SIZE, dtype=int)
    for s in range(SIZE * SIZE):
        r, c = pos(s)
        code = 0
        for a, (dr, dc) in enumerate(DELTAS):
            n = (r + dr, c + dc)
            if not _inside(n) or n in walls:
                k = 0
            elif n in hazards:
                k = 1
            elif n == goal:
                k = 2
            else:
                k = 4 if a == best[s] else 3
            code = code * 5 + k
        obs[s] = code
    return obs


class LocalViewAgent:
    name = "Local-view Q"

    def __init__(self, q, layout):
        self.q = q
        self.obs = observation_table(layout)

    def act(self, s):
        return int(np.argmax(self.q[self.obs[s]]))

    def cell_q(self):
        """Q-values re-indexed by cell (for policy / value overlays)."""
        return self.q[self.obs]


def train_local_view(layouts, reliability=(0.7, 1.0), episodes=20000, seed=0, alpha=0.2, alpha_end=0.02,
                     gamma=0.98, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.7, q_init=None):
    """Q-learning over observations; each episode samples one floor from `layouts`."""
    rng = random.Random(seed + 104729)
    envs = [WarehouseEnv(l, 1.0, seed=seed * 1000 + i) for i, l in enumerate(layouts)]
    obs_tabs = [observation_table(l).tolist() for l in layouts]
    randomized = isinstance(reliability, (tuple, list))
    q = np.zeros((N_OBS, N_ACTIONS)) if q_init is None else q_init.copy()
    qs = q.tolist()
    decay = max(1, int(episodes * eps_decay_frac))
    returns = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay)
        lr = alpha - (alpha - alpha_end) * ep / max(1, episodes - 1)
        k = rng.randrange(len(envs))
        env, ob = envs[k], obs_tabs[k]
        env.reliability = rng.uniform(*reliability) if randomized else reliability
        s = env.reset()
        total = 0.0
        while True:
            row = qs[ob[s]]
            if rng.random() < eps:
                a = rng.randrange(N_ACTIONS)
            else:
                a = row.index(max(row))
            s2, r, done, status, _ = env.step(a)
            total += r
            target = r if done else r + gamma * max(qs[ob[s2]])
            row[a] += lr * (target - row[a])
            s = s2
            if done or status != RUNNING:
                break
        returns.append(total)
    return np.array(qs), returns
