"""Warehouse maps for SafeRoute.

Legend
    S  start (loading dock)
    G  goal (delivery point)
    #  shelf / obstacle (impassable, agent bumps and stays put)
    H  hazard (spill / forklift zone, episode ends with a penalty)
    .  free floor

The training map is deliberately built so that the *shortest* hazard-free
route runs through a narrow aisle flanked by hazards, while a longer outer
aisle is completely hazard-free. Under perfect movement the short aisle is
optimal; under slippery movement it is a trap. That tension is exactly what
the research question probes.
"""
from __future__ import annotations

import random
from collections import deque

TRAIN_MAP = [
    "S.......",
    "###.###.",
    "..H.H.#.",
    ".#H.H.H.",
    ".#H.H.#.",
    ".#....H.",
    ".##.###.",
    "...G....",
]
# Route A (centre aisle): 10 steps, hazards on BOTH sides for 3 cells.
# Route B (outer aisle):  18 steps, hazard on ONE side for 2 cells.

SIZE = 8


def parse(rows):
    """Return dict with start, goal, walls (set), hazards (set)."""
    walls, hazards = set(), set()
    start = goal = None
    for r, row in enumerate(rows):
        assert len(row) == SIZE, f"row {r} has length {len(row)}"
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            elif ch == "H":
                hazards.add((r, c))
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    assert start and goal
    return {"start": start, "goal": goal, "walls": walls, "hazards": hazards}


def to_rows(layout):
    rows = []
    for r in range(SIZE):
        line = []
        for c in range(SIZE):
            p = (r, c)
            if p == layout["start"]:
                line.append("S")
            elif p == layout["goal"]:
                line.append("G")
            elif p in layout["walls"]:
                line.append("#")
            elif p in layout["hazards"]:
                line.append("H")
            else:
                line.append(".")
        rows.append("".join(line))
    return rows


def bfs_distance(layout, blocked_hazards=True):
    """Distance-to-goal for every reachable cell (hazards treated as blocked)."""
    blocked = set(layout["walls"])
    if blocked_hazards:
        blocked |= layout["hazards"]
    dist = {layout["goal"]: 0}
    q = deque([layout["goal"]])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in blocked and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def _rearrange(rng, n_hazards, wall_flips):
    """One rearranged floor: same start/goal, hazards moved, a few shelves shifted.
    Returns None if the result has no hazard-free route (or one that is too long)."""
    base = parse(TRAIN_MAP)
    walls = set(base["walls"])
    free = [(r, c) for r in range(SIZE) for c in range(SIZE)
            if (r, c) not in walls and (r, c) not in (base["start"], base["goal"])]
    for w in rng.sample(sorted(walls), wall_flips):
        walls.discard(w)
    for p in rng.sample(free, wall_flips):
        walls.add(p)
    free = [(r, c) for r in range(SIZE) for c in range(SIZE)
            if (r, c) not in walls and (r, c) not in (base["start"], base["goal"])]
    # keep the cells right next to start/goal clear of hazards
    protected = {base["start"], base["goal"], (0, 1), (6, 3), (7, 2), (7, 4)}
    cand = [p for p in free if p not in protected]
    hazards = set(rng.sample(cand, min(n_hazards, len(cand))))
    lay = {"start": base["start"], "goal": base["goal"], "walls": walls, "hazards": hazards}
    dist = bfs_distance(lay)
    if base["start"] not in dist or dist[base["start"]] > 24:
        return None
    return lay


def make_unseen_layouts(n=5, seed=2026, n_hazards=6, wall_flips=2):
    """Test-only layouts with a fixed difficulty (kept for backwards compatibility)."""
    rng = random.Random(seed)
    layouts, seen = [], {tuple(TRAIN_MAP)}
    while len(layouts) < n:
        lay = _rearrange(rng, n_hazards, wall_flips)
        if lay is None or tuple(to_rows(lay)) in seen:
            continue
        seen.add(tuple(to_rows(lay)))
        layouts.append(lay)
    return layouts


def optimal_success_prob(layout, reliability, horizon=60, gamma=0.98):
    """Exact probability that the optimal (value-iteration) policy delivers within `horizon` steps."""
    from .env import idx
    from .oracle import transition_model, value_iteration
    import numpy as np
    q = value_iteration(layout, reliability, gamma)
    pol = q.argmax(1)
    P, _, term, skip = transition_model(layout, reliability)
    Ppi = P[np.arange(SIZE * SIZE), pol]          # (S, S) under the optimal policy
    goal = idx(layout["goal"])
    S = np.zeros(SIZE * SIZE)
    for _ in range(horizon):
        nxt = np.where(term, 0.0, S)
        nxt[goal] = 1.0
        S = Ppi @ nxt
        S[skip] = 0.0
    return float(S[idx(layout["start"])])


# Level schedule: more hazards and more shifted shelves as levels rise.
LEVEL_HAZARDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LEVEL_WALL_FLIPS = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def make_levels(n=10, seed=2026, test_reliability=0.9, min_optimal=0.85, candidates=30):
    """10 graded test-only levels, easy -> hard.

    For each level we sample `candidates` rearranged floors with that level's
    hazard/shelf budget, keep only those where the optimal policy still delivers
    with probability >= `min_optimal` at `test_reliability` (so an 80% clear bar
    is reachable), and pick the one closest to a target difficulty that falls
    linearly from 'almost free' (level 1) to the feasibility edge (level 10).
    Difficulty = 1 - optimal success + 0.01 * shortest safe route length.
    """
    rng = random.Random(seed)
    seen = {tuple(TRAIN_MAP)}
    levels = []
    for i in range(n):
        target = (1 - min_optimal) * i / max(1, n - 1)
        pool = []
        tries = 0
        while len(pool) < candidates and tries < candidates * 40:
            tries += 1
            lay = _rearrange(rng, LEVEL_HAZARDS[i % len(LEVEL_HAZARDS)], LEVEL_WALL_FLIPS[i % len(LEVEL_WALL_FLIPS)])
            if lay is None or tuple(to_rows(lay)) in seen:
                continue
            p = optimal_success_prob(lay, test_reliability)
            if p < min_optimal:
                continue
            pool.append((abs((1 - p) - target), p, lay))
        pool.sort(key=lambda x: x[0])
        _, p, lay = pool[0]
        lay["optimal_success"] = p
        seen.add(tuple(to_rows(lay)))
        levels.append(lay)
    return levels


def make_random_training_maps(n=300, seed=777, exclude=()):
    """Diverse random floors used ONLY to train the local-view agent.
    Hazard/shelf budgets span the whole level range; levels are excluded."""
    rng = random.Random(seed)
    seen = {tuple(TRAIN_MAP)} | {tuple(e) for e in exclude}
    maps = []
    while len(maps) < n:
        lay = _rearrange(rng, rng.randint(2, 12), rng.randint(0, 5))
        if lay is None or tuple(to_rows(lay)) in seen:
            continue
        seen.add(tuple(to_rows(lay)))
        maps.append(lay)
    return maps
