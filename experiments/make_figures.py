"""Build the static report figures from results/*.csv  ->  report/figures/*.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saferoute.evaluate import ci95  # noqa: E402
from saferoute.style import (GRID, INK, INK_2, SURFACE, STATUS, FINE_TUNE_COLOR,  # noqa: E402
                             SCRATCH_COLOR, SHORTEST_COLOR, agent_style)

RES, FIG = ROOT / "results", ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DASH = {"solid": "-", "dash": "--", "dashdot": "-.", "dot": ":"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def load_e1():
    return pd.read_csv(RES / "e1_reliability.csv", dtype={"train_rel": str})


def fig_map():
    rows = json.loads((RES / "layouts.json").read_text())["train"]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    colors = {"#": "#2b3a55", "H": "#e34948", ".": "#f1f0ec", "S": "#86b6ef", "G": "#1baf7a"}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            ax.add_patch(plt.Rectangle((c, 7 - r), 0.94, 0.94, color=colors[ch], lw=0))
            if ch in "SGH":
                ax.text(c + .47, 7 - r + .47, ch, ha="center", va="center", color="white", weight="bold")
    route_a = [(0, 0), (0, 3), (7, 3)]
    route_b = [(0, 0), (0, 7), (7, 7), (7, 3)]
    for route, col, lab in ((route_a, "#eda100", "Route A: centre aisle (10 steps)"),
                            (route_b, "#6250d6", "Route B: outer aisle (18 steps)")):
        xs = [c + .47 for r, c in route]
        ys = [7 - r + .47 for r, c in route]
        ax.plot(xs, ys, color=col, lw=3, alpha=.85, label=lab, solid_capstyle="round")
    ax.set_xlim(-.1, 8)
    ax.set_ylim(-.1, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.01), ncol=1, fontsize=9)
    ax.set_title("Training warehouse (S start, G goal, H hazard)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_map.png", dpi=170)
    plt.close(fig)


def fig_reliability(d):
    metrics = [("success_rate", "Success rate"), ("hazard_rate", "Hazard rate"),
               ("timeout_rate", "Timeout rate"), ("steps_to_goal", "Steps to goal (successful runs)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    groups = d.groupby(["agent", "train_rel"])
    # 8 series max for legibility; Q @ 0.95 / 0.80 and Random are in the heatmap / tables
    order = sorted([k for k in groups.groups.keys() if k[1] not in ("0.95", "0.80") and k[0] != "Random"],
                   key=lambda k: (k[0] != "Q-learning", k[0], k[1]))
    for ax, (m, title) in zip(axes, metrics):
        for key in order:
            g = groups.get_group(key)
            label, col, dash = agent_style(*key)
            agg = g.groupby("test_rel")[m].agg(["mean", ci95]).reset_index()
            ax.plot(agg.test_rel, agg["mean"], DASH[dash], color=col, lw=2, marker="o", ms=4, label=label)
            ax.fill_between(agg.test_rel, agg["mean"] - agg["ci95"], agg["mean"] + agg["ci95"],
                            color=col, alpha=.12, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Test movement reliability r")
        ax.invert_xaxis()
        if m != "steps_to_goal":
            ax.set_ylim(-0.03, 1.03)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.08))
    fig.suptitle("E1 · Training reliability vs test reliability (mean ± 95% CI over 10 seeds; x axis = more slip →)",
                 weight="bold", y=1.0)
    fig.tight_layout(rect=(0, .08, 1, .97))
    fig.savefig(FIG / "fig2_reliability_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(d):
    q = d[d.agent == "Q-learning"]
    for m, cmap, name in (("success_rate", "Blues", "fig3_success_heatmap.png"),
                          ("hazard_rate", "Reds", "fig3b_hazard_heatmap.png")):
        p = q.pivot_table(index="train_rel", columns="test_rel", values=m)
        p = p.reindex(["1.00", "0.95", "0.90", "0.80", "0.70"] +
                      [i for i in p.index if i.startswith("randomized")])
        p = p[sorted(p.columns, reverse=True)]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        im = ax.imshow(p.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(p.shape[1]), [f"{c:.2f}" for c in p.columns])
        ax.set_yticks(range(p.shape[0]), p.index)
        ax.set_xlabel("Test reliability (unseen levels: 0.6, 0.5)")
        ax.set_ylabel("Training reliability")
        ax.grid(False)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > .6 else INK)
        ax.set_title(f"Q-learning {m.replace('_', ' ')}: train × test reliability")
        fig.colorbar(im, ax=ax, fraction=.03)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=160)
        plt.close(fig)


def fig_generalization():
    g = pd.read_csv(RES / "e2_generalization.csv")
    g = g[np.isclose(g.test_rel, 0.9)]
    order = ["Optimal (VI)", "Risk-aware planner", "Shortest-path (replans)", "Local-view Q (frozen)",
             "Q frozen (1.00)", "Q frozen (0.90)", "Q frozen (0.80)", "Q frozen (randomized 0.7-1.0)", "Random"]
    p = g.pivot_table(index="agent", columns="level", values="success_rate").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    im = ax.imshow(p.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(p.shape[1]), [f"L{c}" for c in p.columns])
    ax.set_yticks(range(p.shape[0]), p.index)
    ax.grid(False)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > .6 else INK)
    ax.set_xlabel("Level (easy → hard; never seen in training)")
    ax.set_title("E2 · Success on the 10 levels before any fine-tuning (r = 0.9, 10 seeds)")
    fig.colorbar(im, ax=ax, fraction=.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_generalization.png", dpi=160)
    plt.close(fig)


def fig_retraining():
    r = pd.read_csv(RES / "e3_retraining.csv")
    modes = {"cell Q · from scratch": SCRATCH_COLOR, "cell Q · fine-tune (ε 1.0→0.05)": "#1c5cab",
             "cell Q · fine-tune (ε 0.3→0.05)": "#86b6ef", "local-view Q · fine-tune (ε 0.3→0.05)": "#6250d6"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for ax, m, title in ((axes[0], "success_rate", "Success rate"), (axes[1], "timeout_rate", "Timeout rate")):
        for mode, col in modes.items():
            per_seed = r[r["mode"] == mode].groupby(["budget", "seed"])[m].mean().reset_index()
            agg = per_seed.groupby("budget")[m].agg(["mean", ci95]).reset_index()
            x = np.arange(len(agg))
            ax.plot(x, agg["mean"], "-o", color=col, lw=2, ms=5, label=mode)
            ax.fill_between(x, agg["mean"] - agg["ci95"], agg["mean"] + agg["ci95"], color=col, alpha=.15, lw=0)
            ax.set_xticks(x, agg.budget)
        ax.set_xlabel("Retraining episodes on the level")
        ax.set_ylim(-.03, 1.03)
        ax.set_title(title)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.1))
    fig.suptitle("E3 · Fixed retraining budgets, averaged over 10 levels (r = 0.9, mean ± 95% CI)", weight="bold")
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(FIG / "fig5_retraining.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_campaign():
    e = pd.read_csv(RES / "e4_campaign.csv")
    agents = {"Q (trained r=1.00)": "#86b6ef", "Q (trained r=0.90)": "#2a78d6",
              "Q (randomized 0.7-1.0)": "#eb6834", "Local-view Q": "#6250d6"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.2
    levels = sorted(e.level.unique())
    for k, (name, col) in enumerate(agents.items()):
        g = e[e.agent == name].groupby("level")["rounds"].agg(["mean", ci95]).reindex(levels)
        x = np.arange(len(levels)) + (k - 1.5) * w
        ax.bar(x, g["mean"], w * 0.92, color=col, label=name, edgecolor=SURFACE, lw=1)
        ax.errorbar(x, g["mean"], yerr=g["ci95"], fmt="none", ecolor=INK_2, elinewidth=.8, capsize=2)
    ax.set_xticks(range(len(levels)), [f"L{l}" for l in levels])
    ax.set_ylabel("rounds of 300 episodes")
    ax.set_title("E4 · Fine-tuning rounds needed to clear each level (≥ 80% success; 0 = cleared zero-shot)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_campaign.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_learning_curves():
    lc = pd.read_csv(RES / "learning_curves.csv", dtype={"train_rel": str})
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for tr, g in lc.groupby("train_rel"):
        label, col, _ = agent_style("Q-learning", tr)
        agg = g.groupby("episode")["mean_return"].mean()
        ax.plot(agg.index, agg.values, color=col, lw=2, label=label)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean return (50-episode window)")
    ax.set_title("Q-learning training curves (mean over 10 seeds)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_learning_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    d = load_e1()
    fig_map()
    fig_reliability(d)
    fig_heatmap(d)
    fig_generalization()
    fig_retraining()
    fig_campaign()
    fig_learning_curves()
    print("figures ->", FIG)
