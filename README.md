# SafeRoute 

**Does a reinforcement-learning robot trained under uncertain movement navigate more safely than one trained with perfect movement?**

An agent crosses an 8 × 8 warehouse grid to a delivery point, avoiding shelves and hazard zones. We train tabular Q-learning at different movement reliabilities, compare it with random and shortest-path baselines (plus an exact value-iteration reference), test on unseen noise levels and 10 unseen levels, and fine-tune live until each level is cleared — all across 10 seeds with separate train/test runs.

The Streamlit dashboard has an **arcade panel** on one side and the **research dashboard** on the other. The arcade runs a **10-level campaign**: pick an agent and a level; if it scores below 80% it is **fine-tuned live** in 300-episode rounds until it clears the level, which unlocks the next one. You can replay any test episode before or after fine-tuning, overlay the policy and values, or drive the robot yourself under the same slip.

## Quick start

```bash
pip install -r requirements.txt
python experiments/run_experiments.py   # ~1.5 min; results/ is already included, so this is optional
python experiments/make_figures.py      # report figures
streamlit run app/streamlit_app.py
```

Tests: `python tests/test_core.py` (or `pytest -q`), `python tests/smoke_app.py` (headless dashboard check), `node tests/smoke_game.js tests/_last_game.html` (arcade logic).

## Headline results (10 seeds)

| Agent | Success @ r=1.0 | Success / hazard @ r=0.8 | Success / hazard @ r=0.6 (unseen) |
|---|---|---|---|
| Q trained with perfect movement | 100% | 51% / 49% | 18% / 82% |
| Q trained at r = 0.9 | 100% | 70% / 30% | 46% / 54% |
| Q trained at r = 0.7 | **0%** (stalls) | 88% / 0% | 94% / 0% |
| Q trained with randomised r ∈ [0.7, 1] | 90% | 78% / 22% | 57% / 43% |
| Shortest-path planner | 100% | 51% / 49% | 18% / 82% |

Slip-aware training is much safer under slip, but too much training noise produces a wall-crawling policy that fails when movement is perfect; randomised training is the most robust.

On 10 unseen levels, frozen cell-index Q-tables average 19–40% success; a **local-view Q** agent trained on 300 random floors averages 71% zero-shot (a risk-aware planner: 92%). In the live campaign every learning agent clears all 10 levels on all 10 seeds, usually after a single 300-episode fine-tuning round. Full write-up: [`report/REPORT.md`](report/REPORT.md).

## Layout

```
saferoute/        env (slip model), maps + levels, agents, local-view agent, planners, campaign, oracle, evaluation
experiments/      run_experiments.py (E1–E4), make_figures.py
app/              streamlit_app.py + game_template.html (arcade canvas)
results/          CSVs, Q-tables, layouts, config_used.json
report/           REPORT.md + figures/
tests/            unit tests + headless smoke tests
config.json       every experimental setting
```
