"""Reproduce every SafeRoute result from scratch.

    python experiments/run_experiments.py            # full run (config.json), ~4 min
    python experiments/run_experiments.py --quick    # 3 seeds, smoke test

Experiments
    E1  Reliability study  - fixed map; train at each reliability (and with a
        randomized reliability); test on every reliability incl. unseen ones.
    E2  Generalization     - frozen agents tested on 10 unseen, graded levels.
    E3  Retraining         - fine-tune (two exploration settings) vs. train from
        scratch on each level with small episode budgets.
    E4  Level campaign     - the dashboard's procedure: fine-tune in 300-episode
        rounds until the agent clears the level (>= 80% success).

Training and testing use disjoint random streams: training uses `seed`,
testing uses `100000 + seed` (see saferoute/evaluate.py). Levels and the local
view agent's random training maps are generated from fixed seeds and never
overlap (see saferoute/maps.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute import campaign  # noqa: E402
from saferoute.agents import QAgent, RandomAgent, ShortestPathAgent, train_q_learning  # noqa: E402
from saferoute.evaluate import evaluate, summarize  # noqa: E402
from saferoute.local_view import LocalViewAgent, train_local_view  # noqa: E402
from saferoute.maps import TRAIN_MAP, make_levels, make_random_training_maps, parse, to_rows  # noqa: E402
from saferoute.oracle import OracleAgent  # noqa: E402
from saferoute.planners import SafePlannerAgent  # noqa: E402

RESULTS = ROOT / "results"


def rkey(r):
    return "rand" if isinstance(r, (list, tuple)) else f"{r:.2f}"


def rlabel(r):
    return f"randomized {r[0]:.1f}-{r[1]:.1f}" if isinstance(r, (list, tuple)) else f"{r:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--quick", action="store_true", help="3 seeds, fewer episodes")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:3]
        cfg["eval_episodes"] = 100
        cfg["local_view"]["episodes"] = 8000
    RESULTS.mkdir(exist_ok=True)
    for old in ("e2_generalization.csv", "e3_retraining.csv", "retrained_qtables.npz", "e4_campaign.csv"):
        (RESULTS / old).unlink(missing_ok=True)
    (RESULTS / "config_used.json").write_text(json.dumps(cfg, indent=2))

    hp = cfg["q_learning"]
    seeds, E, N = cfg["seeds"], cfg["train_episodes"], cfg["eval_episodes"]
    layout = parse(TRAIN_MAP)
    train_rs = list(cfg["train_reliabilities"]) + [tuple(cfg["randomized_range"])]
    t0 = time.time()

    # ---------------- levels + local-view training maps ----------------
    levels = make_levels(**cfg["levels"])
    lv = cfg["local_view"]
    lv_maps = make_random_training_maps(lv["n_training_maps"], lv["maps_seed"], exclude=[to_rows(l) for l in levels])
    (RESULTS / "layouts.json").write_text(json.dumps({
        "train": TRAIN_MAP,
        "levels": [to_rows(l) for l in levels],
        "level_optimal_success": [l["optimal_success"] for l in levels],
        "level_test_reliability": cfg["levels"]["test_reliability"],
    }, indent=2))
    print(f"levels ready ({time.time() - t0:.0f}s)")

    # ---------------- E1: reliability study ----------------
    rows, curves, qtables = [], [], {}
    w = cfg["learning_curve_window"]

    def log_curve(name, rets, seed):
        rets = np.array(rets)
        step = max(w, len(rets) // 60)
        for i in range(0, len(rets), step):
            curves.append({"train_rel": name, "seed": seed, "episode": i + step,
                           "mean_return": float(rets[i:i + step].mean())})

    for tr in train_rs:
        for seed in seeds:
            q, rets = train_q_learning(layout, tr, episodes=E, seed=seed, **hp)
            qtables[f"q_{rkey(tr)}_s{seed}"] = q
            log_curve(rlabel(tr), rets, seed)
            agent = QAgent(q)
            for te in cfg["test_reliabilities"]:
                rows.append({"agent": "Q-learning", "train_rel": rlabel(tr), "test_rel": te, "seed": seed,
                             **summarize(evaluate(agent, layout, te, N, seed=seed))})
        print(f"E1 trained/tested Q @ {rlabel(tr)}  ({time.time() - t0:.0f}s)")
    lv_range = tuple(lv["reliability_range"])
    for seed in seeds:
        q, _ = train_local_view(lv_maps, lv_range, episodes=lv["episodes"], seed=seed, **hp)
        qtables[f"local_s{seed}"] = q
        agent = LocalViewAgent(q, layout)
        for te in cfg["test_reliabilities"]:
            rows.append({"agent": "Local-view Q", "train_rel": "random maps", "test_rel": te, "seed": seed,
                         **summarize(evaluate(agent, layout, te, N, seed=seed))})
    print(f"E1 trained/tested local-view Q ({time.time() - t0:.0f}s)")
    sp = ShortestPathAgent(layout)
    safe = SafePlannerAgent(layout)
    oracles = {te: OracleAgent(layout, te, hp["gamma"]) for te in cfg["test_reliabilities"]}
    for seed in seeds:
        for te in cfg["test_reliabilities"]:
            rows.append({"agent": "Optimal (VI, knows test r)", "train_rel": "baseline", "test_rel": te,
                         "seed": seed, **summarize(evaluate(oracles[te], layout, te, N, seed=seed))})
            rows.append({"agent": "Shortest-path", "train_rel": "baseline", "test_rel": te, "seed": seed,
                         **summarize(evaluate(sp, layout, te, N, seed=seed))})
            rows.append({"agent": "Risk-aware planner", "train_rel": "baseline", "test_rel": te, "seed": seed,
                         **summarize(evaluate(safe, layout, te, N, seed=seed))})
            rows.append({"agent": "Random", "train_rel": "baseline", "test_rel": te, "seed": seed,
                         **summarize(evaluate(RandomAgent(seed), layout, te, N, seed=seed))})
    pd.DataFrame(rows).to_csv(RESULTS / "e1_reliability.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "learning_curves.csv", index=False)
    np.savez_compressed(RESULTS / "qtables.npz", **qtables)
    print(f"E1 done ({time.time() - t0:.0f}s)")

    # ---------------- E2: levels, frozen agents ----------------
    rows = []
    frozen_rs = [1.0, cfg["retraining"]["source_train_reliability"], 0.8, tuple(cfg["randomized_range"])]
    for li, lay in enumerate(levels, start=1):
        sp, safe = ShortestPathAgent(lay), SafePlannerAgent(lay)
        for te in cfg["generalization_test_reliabilities"]:
            orc = OracleAgent(lay, te, hp["gamma"])
            for seed in seeds:
                def add(name, agent):
                    rows.append({"level": li, "agent": name, "test_rel": te, "seed": seed,
                                 **summarize(evaluate(agent, lay, te, N, seed=seed))})
                for tr in frozen_rs:
                    add(f"Q frozen ({rlabel(tr)})", QAgent(qtables[f"q_{rkey(tr)}_s{seed}"]))
                add("Local-view Q (frozen)", LocalViewAgent(qtables[f"local_s{seed}"], lay))
                add("Shortest-path (replans)", sp)
                add("Risk-aware planner", safe)
                add("Optimal (VI)", orc)
                add("Random", RandomAgent(seed))
        print(f"E2 level {li} done ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(RESULTS / "e2_generalization.csv", index=False)

    # ---------------- E3: limited retraining ----------------
    rc = cfg["retraining"]
    src, rte, NE3 = rc["source_train_reliability"], rc["test_reliability"], rc["eval_episodes"]
    low_eps = dict(hp, **{k: v for k, v in campaign.FINE_TUNE.items()})
    rows = []
    for li, lay in enumerate(levels, start=1):
        for seed in seeds:
            q_src = qtables[f"q_{rkey(src)}_s{seed}"]
            q_loc = qtables[f"local_s{seed}"]
            for budget in rc["budgets"]:
                variants = {
                    "cell Q · fine-tune (ε 1.0→0.05)": ("cell", q_src, hp),
                    "cell Q · fine-tune (ε 0.3→0.05)": ("cell", q_src, low_eps),
                    "cell Q · from scratch": ("cell", None, hp),
                    "local-view Q · fine-tune (ε 0.3→0.05)": ("local", q_loc, low_eps),
                }
                for mode, (kind, qi, h) in variants.items():
                    if budget == 0:
                        q = qi if qi is not None else np.zeros((64, 4))
                    elif kind == "cell":
                        q, _ = train_q_learning(lay, rte, episodes=budget, seed=seed + 50_000, q_init=qi, **h)
                    else:
                        q, _ = train_local_view([lay], rte, episodes=budget, seed=seed + 50_000, q_init=qi, **h)
                    rows.append({"level": li, "mode": mode, "budget": budget, "seed": seed,
                                 **summarize(evaluate(campaign.make_agent(kind, q, lay), lay, rte, NE3, seed=seed))})
        print(f"E3 level {li} done ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(RESULTS / "e3_retraining.csv", index=False)

    # ---------------- E4: level campaign (rounds to clear) ----------------
    cc = cfg["campaign"]
    campaign.CLEAR_BAR, campaign.ROUND_EPISODES = cc["clear_bar"], cc["round_episodes"]
    rows = []
    starters = {"Q (trained r=1.00)": ("cell", "q_1.00"), "Q (trained r=0.90)": ("cell", "q_0.90"),
                "Q (randomized 0.7-1.0)": ("cell", "q_rand"), "Local-view Q": ("local", "local")}
    for name, (kind, key) in starters.items():
        for seed in seeds:
            for li, lay in enumerate(levels, start=1):
                _, hist = campaign.run_campaign_level(kind, qtables[f"{key}_s{seed}"], lay, rte, seed=seed,
                                                      max_rounds=cc["max_rounds"], eval_episodes=cc["eval_episodes"])
                last = hist[-1]
                rows.append({"agent": name, "seed": seed, "level": li, "rounds": last["round"],
                             "episodes": last["episodes"], "cleared": last["success_rate"] >= cc["clear_bar"],
                             "start_success": hist[0]["success_rate"], "final_success": last["success_rate"],
                             "final_hazard": last["hazard_rate"], "final_timeout": last["timeout_rate"]})
        print(f"E4 campaign {name} done ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(RESULTS / "e4_campaign.csv", index=False)
    print(f"All experiments finished in {time.time() - t0:.0f}s -> {RESULTS}")


if __name__ == "__main__":
    main()
