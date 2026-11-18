"""Exact optimal policy by value iteration (an analysis reference, not a learner).

It knows the true map AND the true movement reliability, so it is the best any
agent could do. We use it to (a) check that Q-learning converged to the right
behaviour, (b) see which route is genuinely optimal at each reliability and
(c) certify that every level can be cleared.
"""
from __future__ import annotations

import numpy as np

from .env import (GOAL_REWARD, HAZARD_REWARD, N_ACTIONS, N_STATES, STEP_REWARD,
                  WarehouseEnv)


def transition_model(layout, reliability):
    """Dense model: P[s, a, n] probabilities, R[s, a, n] rewards, T[n] terminal mask."""
    env = WarehouseEnv(layout, reliability)
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for n, p in env.transition_probs(s, a).items():
                P[s, a, n] += p
    rew = np.full(N_STATES, STEP_REWARD)
    term = np.zeros(N_STATES, dtype=bool)
    for h in env.hazards:
        rew[h], term[h] = HAZARD_REWARD, True
    rew[env.goal], term[env.goal] = GOAL_REWARD, True
    skip = np.zeros(N_STATES, dtype=bool)
    skip[list(env.walls | env.hazards | {env.goal})] = True
    return P, rew, term, skip


def value_iteration(layout, reliability, gamma=0.98, tol=1e-9, max_iter=5000):
    P, rew, term, skip = transition_model(layout, reliability)
    expected_r = P @ rew                       # (S, A)
    cont = gamma * (~term)                     # no bootstrap past terminal cells
    V = np.zeros(N_STATES)
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(max_iter):
        Q = expected_r + P @ (cont * V)
        Q[skip] = 0.0
        V2 = Q.max(1)
        if np.max(np.abs(V2 - V)) < tol:
            break
        V = V2
    return Q


class OracleAgent:
    name = "Optimal (value iteration)"

    def __init__(self, layout, reliability, gamma=0.98):
        self.q = value_iteration(layout, reliability, gamma)

    def act(self, s):
        return int(np.argmax(self.q[s]))
