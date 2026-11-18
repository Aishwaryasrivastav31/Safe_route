"""Level campaign: fine-tune an agent on a level in rounds until it clears it.

Used both by the dashboard (live, round by round) and by experiment E4.
A level is CLEARED when the greedy policy succeeds in >= CLEAR_BAR of
`eval_episodes` fresh test episodes at the level's test reliability.
"""
from __future__ import annotations

import numpy as np

from .agents import QAgent, train_q_learning
from .evaluate import evaluate, summarize
from .local_view import LocalViewAgent, train_local_view

CLEAR_BAR = 0.80
ROUND_EPISODES = 300
MAX_ROUNDS = 15
FINE_TUNE = dict(eps_start=0.3, eps_end=0.05, eps_decay_frac=0.8, alpha=0.2, alpha_end=0.05)


def make_agent(kind, q, layout):
    return LocalViewAgent(q, layout) if kind == "local" else QAgent(q)


def finetune_round(kind, q, layout, reliability, round_idx, seed=0, episodes=ROUND_EPISODES):
    """One round of fine-tuning; returns the new Q-table (input is not modified)."""
    rseed = 900_000 + 1000 * seed + round_idx
    if kind == "local":
        q2, _ = train_local_view([layout], reliability, episodes=episodes, seed=rseed, q_init=q, **FINE_TUNE)
    else:
        q2, _ = train_q_learning(layout, reliability, episodes=episodes, seed=rseed, q_init=q, **FINE_TUNE)
    return q2


def check(kind, q, layout, reliability, seed=0, eval_episodes=200, record=False):
    eps = evaluate(make_agent(kind, q, layout), layout, reliability, eval_episodes, seed=seed, record=record)
    return summarize(eps), eps


def run_campaign_level(kind, q, layout, reliability, seed=0, max_rounds=MAX_ROUNDS, eval_episodes=200):
    """Evaluate; while not cleared, fine-tune one round and re-evaluate.
    Returns (final_q, history) where history[i] = metrics after i rounds (0 = before any)."""
    m, _ = check(kind, q, layout, reliability, seed, eval_episodes)
    hist = [dict(round=0, episodes=0, **m)]
    r = 0
    while m["success_rate"] < CLEAR_BAR and r < max_rounds:
        r += 1
        q = finetune_round(kind, q, layout, reliability, r, seed)
        m, _ = check(kind, q, layout, reliability, seed, eval_episodes)
        hist.append(dict(round=r, episodes=r * ROUND_EPISODES, **m))
    return q, hist
