"""SafeRoute warehouse grid environment.

Movement model ("reliability" r):
    with probability r          the robot moves in the intended direction
    with probability (1 - r)/2  it slips 90 degrees to the left
    with probability (1 - r)/2  it slips 90 degrees to the right
Bumping into a shelf or the outer wall leaves the robot where it is.

Rewards (kept small and interpretable):
    each step  -0.2      reach goal  +10      enter hazard  -10 (terminal)
An episode also ends (timeout, not terminal for bootstrapping) after
`max_steps` moves.
"""
from __future__ import annotations

import random

from .maps import SIZE, TRAIN_MAP, parse

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
N_STATES = SIZE * SIZE
N_ACTIONS = 4

STEP_REWARD = -0.2
GOAL_REWARD = 10.0
HAZARD_REWARD = -10.0
MAX_STEPS = 60

SUCCESS, HAZARD, TIMEOUT, RUNNING = "success", "hazard", "timeout", "running"


def idx(pos):
    return pos[0] * SIZE + pos[1]


def pos(i):
    return divmod(i, SIZE)


class WarehouseEnv:
    """Fast tabular grid world. All moves are pre-computed per (state, action)."""

    def __init__(self, layout=None, reliability=1.0, max_steps=MAX_STEPS, seed=None):
        self.layout = layout if layout is not None else parse(TRAIN_MAP)
        self.reliability = float(reliability)
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.start = idx(self.layout["start"])
        self.goal = idx(self.layout["goal"])
        self.walls = {idx(p) for p in self.layout["walls"]}
        self.hazards = {idx(p) for p in self.layout["hazards"]}
        # deterministic move result for every (state, direction)
        self._move = [[self._raw_move(s, a) for a in range(N_ACTIONS)] for s in range(N_STATES)]
        self.state = self.start
        self.t = 0

    def _raw_move(self, s, a):
        r, c = pos(s)
        dr, dc = DELTAS[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            return s
        n = nr * SIZE + nc
        return s if n in self.walls else n

    def seed(self, seed):
        self.rng.seed(seed)

    def reset(self):
        self.state = self.start
        self.t = 0
        return self.state

    def actual_direction(self, a):
        """Sample the direction that is really executed (slip model)."""
        u = self.rng.random()
        if u < self.reliability:
            return a
        return (a + 1) % 4 if u < self.reliability + (1 - self.reliability) / 2 else (a + 3) % 4

    def step(self, a):
        """Returns next_state, reward, done (terminal), status, executed_direction."""
        d = self.actual_direction(a)
        s2 = self._move[self.state][d]
        self.t += 1
        self.state = s2
        if s2 in self.hazards:
            return s2, HAZARD_REWARD, True, HAZARD, d
        if s2 == self.goal:
            return s2, GOAL_REWARD, True, SUCCESS, d
        if self.t >= self.max_steps:
            return s2, STEP_REWARD, False, TIMEOUT, d
        return s2, STEP_REWARD, False, RUNNING, d

    def transition_probs(self, s, a):
        """Exact next-state distribution (used by tests / analysis)."""
        r = self.reliability
        out = {}
        for d, p in ((a, r), ((a + 1) % 4, (1 - r) / 2), ((a + 3) % 4, (1 - r) / 2)):
            if p > 0:
                n = self._move[s][d]
                out[n] = out.get(n, 0) + p
        return out
