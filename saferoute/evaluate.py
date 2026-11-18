"""Evaluation helpers: run test episodes, record trajectories, aggregate metrics."""
from __future__ import annotations

import numpy as np

from .env import RUNNING, SUCCESS, HAZARD, TIMEOUT, WarehouseEnv, pos

TEST_SEED_OFFSET = 100_000  # test RNG streams never overlap with training seeds


def run_episode(env, agent, record=False):
    s = env.reset()
    traj = [{"pos": list(pos(s)), "intended": None, "executed": None}] if record else None
    total, steps = 0.0, 0
    while True:
        a = agent.act(s)
        s, r, done, status, d = env.step(a)
        total += r
        steps += 1
        if record:
            traj.append({"pos": list(pos(s)), "intended": int(a), "executed": int(d)})
        if done or status != RUNNING:
            return {"status": status, "steps": steps, "return": total, "trajectory": traj}


def evaluate(agent, layout, reliability, episodes=300, seed=0, record=False):
    env = WarehouseEnv(layout, reliability, seed=TEST_SEED_OFFSET + seed)
    eps = [run_episode(env, agent, record=record) for _ in range(episodes)]
    return eps


def summarize(episodes):
    n = len(episodes)
    status = [e["status"] for e in episodes]
    succ_steps = [e["steps"] for e in episodes if e["status"] == SUCCESS]
    return {
        "success_rate": status.count(SUCCESS) / n,
        "hazard_rate": status.count(HAZARD) / n,
        "timeout_rate": status.count(TIMEOUT) / n,
        "steps_to_goal": float(np.mean(succ_steps)) if succ_steps else float("nan"),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
    }


def ci95(x):
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
