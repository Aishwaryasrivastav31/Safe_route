"""SafeRoute dashboard.

    streamlit run app/streamlit_app.py

Left: the arcade panel. Pick the training warehouse or one of 10 levels.
On a level, a learning agent that fails is fine-tuned live, in 300-episode
rounds, until it clears the level (>= 80% of 200 test runs). Clearing a level
unlocks the next one. Right: the research dashboard.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute import campaign  # noqa: E402
from saferoute.agents import QAgent, RandomAgent, ShortestPathAgent  # noqa: E402
from saferoute.env import GOAL_REWARD, HAZARD_REWARD, MAX_STEPS, STEP_REWARD  # noqa: E402
from saferoute.evaluate import ci95, evaluate, summarize  # noqa: E402
from saferoute.local_view import LocalViewAgent  # noqa: E402
from saferoute.maps import parse  # noqa: E402
from saferoute.oracle import OracleAgent  # noqa: E402
from saferoute.planners import SafePlannerAgent  # noqa: E402
from saferoute.style import DARK, agent_style  # noqa: E402

RES = ROOT / "results"
TEMPLATE = (Path(__file__).parent / "game_template.html").read_text(encoding="utf-8")
TRAIN_LEVELS = ["1.00", "0.95", "0.90", "0.80", "0.70"]
DASHES = {"solid": "solid", "dash": "dash", "dashdot": "dashdot", "dot": "dot"}
N_TEST = 200

st.set_page_config(page_title="SafeRoute", page_icon="🤖", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=JetBrains+Mono:wght@500;700&family=Inter:wght@400;600;800&display=swap');
.stApp {{ background: radial-gradient(1200px 600px at 20% -10%, #12224a 0%, {DARK['surface']} 55%); }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.4rem; max-width: 1500px; }}
.sr-title {{ font-family:'Press Start 2P',monospace; font-size:26px; color:#2ee6ff; letter-spacing:2px;
            text-shadow:0 0 8px #2ee6ffaa, 0 0 26px #2ee6ff44; margin:0; }}
.sr-title span {{ color:#ffd166; text-shadow:0 0 8px #ffd166aa; }}
.sr-sub {{ color:{DARK['ink2']}; font-size:14px; margin:.4rem 0 1rem; }}
.kpi {{ background:linear-gradient(180deg,#0e1830,#0a1122); border:1px solid {DARK['grid']}; border-radius:14px;
       padding:12px 14px; height:100%; }}
.kpi b {{ display:block; font-size:11px; letter-spacing:1.2px; color:{DARK['ink2']}; font-weight:600; text-transform:uppercase; }}
.kpi i {{ font-style:normal; font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:700; color:{DARK['ink']}; }}
.kpi small {{ display:block; color:{DARK['ink2']}; font-size:12px; margin-top:2px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin:.4rem 0; }}
.chip {{ font-family:'JetBrains Mono',monospace; font-size:12px; border-radius:999px; padding:4px 10px;
        border:1px solid {DARK['grid']}; background:#0b1222; color:{DARK['ink']}; }}
.chip .d {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
.finding {{ border-left:3px solid #2ee6ff; background:#0b1222; padding:10px 14px; border-radius:0 10px 10px 0; margin-bottom:8px;
           color:{DARK['ink']}; font-size:14px; }}
.lvstrip {{ display:grid; grid-template-columns:repeat(10,1fr); gap:6px; margin:.2rem 0 .6rem; }}
.lv {{ font-family:'Press Start 2P',monospace; font-size:9px; text-align:center; padding:8px 0 6px; border-radius:10px;
      border:1px solid {DARK['grid']}; background:#0b1222; color:{DARK['ink2']}; line-height:1.7; }}
.lv.ok {{ border-color:#3dd6a0; color:#3dd6a0; box-shadow:0 0 10px #3dd6a033 inset; }}
.lv.open {{ border-color:#2ee6ff; color:#2ee6ff; }}
.lv.cur {{ outline:2px solid #ffd166; outline-offset:1px; }}
.lv.lock {{ opacity:.45; }}
.banner {{ font-family:'Press Start 2P',monospace; font-size:12px; padding:12px 14px; border-radius:12px; margin:.3rem 0 .5rem;
          line-height:1.8; }}
.banner.win {{ background:#0c2a22; border:1px solid #3dd6a0; color:#3dffa8; box-shadow:0 0 18px #3dd6a022; }}
.banner.fail {{ background:#2a0c16; border:1px solid #ff5a7a; color:#ff8aa2; }}
.banner.info {{ background:#0b1a33; border:1px solid #2a6fb8; color:#9fd0ff; }}
.banner small {{ display:block; font-family:'Inter',sans-serif; font-size:12px; color:{DARK['ink2']}; margin-top:4px; }}
.stTabs [data-baseweb="tab"] {{ font-family:'JetBrains Mono',monospace; font-weight:700; }}
section[data-testid="stSidebar"] {{ background:#070d1c; border-right:1px solid {DARK['grid']}; }}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f"<div class='kpi'><b>{label}</b><i>{value}</i><small>{sub}</small></div>", unsafe_allow_html=True)


def style_fig(fig, height=360, title=None):
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=14, color=DARK["ink"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK["ink2"], family="Inter"), margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=11, color=DARK["ink"])),
        hoverlabel=dict(bgcolor="#0b1222", bordercolor="#2ee6ff", font=dict(color=DARK["ink"], family="JetBrains Mono")),
    )
    fig.update_xaxes(gridcolor=DARK["grid"], zerolinecolor=DARK["grid"], linecolor=DARK["grid"])
    fig.update_yaxes(gridcolor=DARK["grid"], zerolinecolor=DARK["grid"], linecolor=DARK["grid"])
    return fig


def _width_kw(fn):
    """Full-width argument for this Streamlit version: older versions take use_container_width,
    newer ones replaced it with width="stretch"."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {"use_container_width": True}
    return {"use_container_width": True} if "use_container_width" in params else {"width": "stretch"}


def show_chart(fig, key, where=None):
    """plotly_chart with a unique key (avoids DuplicateElementId) and the right width argument."""
    where = where or st
    where.plotly_chart(fig, key=key, **_width_kw(st.plotly_chart))


def show_table(df, **kw):
    st.dataframe(df, **_width_kw(st.dataframe), **kw)


# ------------------------------------------------------------------ data
NEEDED = ["e1_reliability.csv", "e2_generalization.csv", "e3_retraining.csv", "e4_campaign.csv",
          "layouts.json", "qtables.npz", "config_used.json", "learning_curves.csv"]


def results_ready():
    if not all((RES / f).exists() for f in NEEDED):
        return False
    return "levels" in json.loads((RES / "layouts.json").read_text())


@st.cache_data(show_spinner=False)
def load_results():
    e1 = pd.read_csv(RES / "e1_reliability.csv", dtype={"train_rel": str})
    e2 = pd.read_csv(RES / "e2_generalization.csv")
    e3 = pd.read_csv(RES / "e3_retraining.csv")
    e4 = pd.read_csv(RES / "e4_campaign.csv")
    lc = pd.read_csv(RES / "learning_curves.csv", dtype={"train_rel": str})
    layouts = json.loads((RES / "layouts.json").read_text())
    cfg = json.loads((RES / "config_used.json").read_text())
    return e1, e2, e3, e4, lc, layouts, cfg


@st.cache_resource(show_spinner=False)
def load_qtables():
    return dict(np.load(RES / "qtables.npz"))


if not results_ready():
    st.markdown("<p class='sr-title'>SAFE<span>ROUTE</span></p>", unsafe_allow_html=True)
    st.warning("No experiment results found yet. Run `python experiments/run_experiments.py` "
               "(a few minutes), or use the button below.")
    if st.button("Run experiments now"):
        with st.spinner("Training and testing agents across seeds…"):
            out = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_experiments.py")],
                                 capture_output=True, text=True)
        st.code(out.stdout[-2000:] + out.stderr[-2000:])
        st.rerun()
    st.stop()

e1, e2, e3, e4, lc, LAYOUTS, CFG = load_results()
QT = load_qtables()
SEEDS = CFG["seeds"]
N_LEVELS = len(LAYOUTS["levels"])
LEVEL_R = float(LAYOUTS["level_test_reliability"])
CLEAR = float(CFG["campaign"]["clear_bar"])
MAX_ROUNDS = int(CFG["campaign"]["max_rounds"])
campaign.CLEAR_BAR = CLEAR
campaign.ROUND_EPISODES = int(CFG["campaign"]["round_episodes"])
MAP_NAMES = ["Training warehouse"] + [f"Level {i + 1}" for i in range(N_LEVELS)]

AGENTS = ([f"Q-learning (trained r={r})" for r in TRAIN_LEVELS] + ["Q-learning (randomized 0.7–1.0)",
          "Local-view Q (random maps)", "Shortest-path planner", "Risk-aware planner",
          "Optimal (value iteration)", "Random"])


def level_of(map_name):
    return 0 if map_name == MAP_NAMES[0] else int(map_name.split()[1])


def map_rows(map_name):
    lv = level_of(map_name)
    return LAYOUTS["train"] if lv == 0 else LAYOUTS["levels"][lv - 1]


def learner(kind, seed):
    """(learner kind 'cell'|'local', base Q-table) for learning agents, else None."""
    if kind.startswith("Q-learning (trained r="):
        return "cell", QT[f"q_{kind.split('=')[1].rstrip(')')}_s{seed}"]
    if kind.startswith("Q-learning (randomized"):
        return "cell", QT[f"q_rand_s{seed}"]
    if kind.startswith("Local-view"):
        return "local", QT[f"local_s{seed}"]
    return None


def make_agent(kind, layout, seed, test_rel, q=None):
    """Return (agent, policy_by_cell or None, values_by_cell or None)."""
    lr = learner(kind, seed)
    if lr:
        lk, q0 = lr
        q = q0 if q is None else q
        if lk == "local":
            a = LocalViewAgent(q, layout)
            cq = a.cell_q()
            return a, [int(x) for x in cq.argmax(1)], [float(x) for x in cq.max(1)]
        return QAgent(q), [int(x) for x in q.argmax(1)], [float(x) for x in q.max(1)]
    if kind.startswith("Shortest"):
        a = ShortestPathAgent(layout)
        return a, [int(x) for x in a.policy], None
    if kind.startswith("Risk"):
        a = SafePlannerAgent(layout)
        return a, [int(x) for x in a.policy], None
    if kind.startswith("Optimal"):
        a = OracleAgent(layout, test_rel)
        return a, [int(x) for x in a.q.argmax(1)], [float(x) for x in a.q.max(1)]
    return RandomAgent(seed), None, None


def game_html(rows, ep, label, policy, values, test_rel, note, autoplay, level=0, cleared=False):
    data = {
        "grid": rows, "traj": ep["trajectory"], "status": ep["status"], "label": label,
        "policy": policy, "values": values, "reliability": test_rel, "max_steps": MAX_STEPS,
        "rewards": {"step": STEP_REWARD, "goal": GOAL_REWARD, "hazard": HAZARD_REWARD},
        "note": note, "autoplay": autoplay, "level": level, "cleared": cleared,
    }
    return TEMPLATE.replace("__DATA__", json.dumps(data))


# ------------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("tuned", {})      # (kind, seed, level) -> {"q": array, "hist": [...], "cleared": bool}
ss.setdefault("cleared", {})    # (kind, seed) -> set(levels)
ss.setdefault("ep_default", 0)   # starting value of the episode picker
ss.setdefault("ep_nonce", 0)     # bumping it recreates the picker with ep_default
ss.setdefault("ep_shown", 0)
ss.setdefault("last_cfg", None)


def set_episode(i):
    ss.ep_default = int(i)
    ss.ep_nonce += 1
ss.setdefault("map_name", MAP_NAMES[0])


def cleared_set(kind, seed):
    return ss.cleared.setdefault((kind, seed), set())


def unlocked(kind, seed, unlock_all):
    c = cleared_set(kind, seed)
    return [lv for lv in range(1, N_LEVELS + 1) if unlock_all or lv == 1 or (lv - 1) in c]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎮 Arcade controls")
    kind = st.selectbox("Agent", AGENTS, index=AGENTS.index("Q-learning (trained r=1.00)"), key="kind")
    seed = st.selectbox("Seed", SEEDS, index=0, help="Training seed. Test episodes always use a separate random stream.")
    unlock_all = st.toggle("Unlock all levels (research mode)", value=False)
    open_lv = unlocked(kind, seed, unlock_all)
    allowed = [MAP_NAMES[0]] + [MAP_NAMES[lv] for lv in open_lv]
    if ss.map_name not in allowed:
        ss.map_name = allowed[-1]
    st.selectbox("Map", allowed, key="map_name",
                 help="Levels unlock one by one: clear a level (≥ 80% success) to open the next.")
    lv_now = level_of(ss.map_name)
    is_learner = learner(kind, seed) is not None
    if lv_now:
        auto_ft = st.toggle("Auto fine-tune when the agent fails", value=True, disabled=not is_learner)
        test_rel = LEVEL_R
        st.caption(f"Levels are played at movement reliability r = {LEVEL_R:.2f} "
                   f"(the robot slips {1 - LEVEL_R:.0%} of the time).")
    else:
        auto_ft = False
        test_rel = st.slider("Test movement reliability r", 0.5, 1.0, 0.9, 0.05,
                             help="Probability the robot moves where it intends. Otherwise it slips 90° left or right.")
    view = "after"
    if lv_now and is_learner:
        view = st.radio("Replay", ["after", "before"], horizontal=True,
                        format_func=lambda v: "After fine-tune" if v == "after" else "Before fine-tune")
    # Streamlit forbids changing a widget's state after it is drawn, so the app never writes to this
    # widget's key: to jump to an episode it bumps the key (fresh widget) and sets its starting value.
    ep_choice = int(st.number_input(f"Episode # (of {N_TEST} test runs)", 0, N_TEST - 1,
                                    value=int(ss.ep_default), key=f"ep_idx_{ss.ep_nonce}"))
    autoplay = st.toggle("Autoplay on change", value=True)
    st.divider()
    st.caption("Train/test separation: training seed s, test stream 100000+s. Levels and r < 0.7 are never seen in "
               "training; the local-view agent trains on 300 other random floors.")

layout_rows = map_rows(ss.map_name)
layout = parse(layout_rows)

# ------------------------------------------------------------------ header
st.markdown("<p class='sr-title'>SAFE<span>ROUTE</span></p>", unsafe_allow_html=True)
st.markdown("<p class='sr-sub'>Does a Q-learning robot trained under <b>uncertain movement</b> navigate an 8×8 warehouse "
            "more safely than one trained with <b>perfect movement</b>, and can it <b>adapt</b> to new floors? "
            "Play the 10-level campaign on the left; explore the evidence on the right.</p>", unsafe_allow_html=True)

left, right = st.columns([1, 1.5], gap="large")

# ------------------------------------------------------------------ arcade panel
with left:
    # level strip
    cset = cleared_set(kind, seed)
    cells = []
    for lv in range(1, N_LEVELS + 1):
        cls = "ok" if lv in cset else ("open" if lv in open_lv else "lock")
        icon = "★" if lv in cset else ("▶" if lv in open_lv else "🔒")
        cur = " cur" if lv == lv_now else ""
        cells.append(f"<div class='lv {cls}{cur}'>L{lv}<br>{icon}</div>")
    st.markdown("<div class='lvstrip'>" + "".join(cells) + "</div>", unsafe_allow_html=True)

    tkey = (kind, seed, lv_now)
    status_box = st.empty()
    progress_box = st.empty()
    chart_box = st.empty()
    q_play = None
    level_cleared = False
    drawn_live = False

    if lv_now and is_learner:
        lk, q0 = learner(kind, seed)
        if tkey not in ss.tuned:
            m0, _ = campaign.check(lk, q0, layout, test_rel, seed, N_TEST)
            rec = {"q0": q0, "q": q0, "hist": [dict(round=0, episodes=0, **m0)], "cleared": m0["success_rate"] >= CLEAR}
            if not rec["cleared"] and auto_ft:
                q = q0
                for r in range(1, MAX_ROUNDS + 1):
                    last = rec["hist"][-1]["success_rate"]
                    status_box.markdown(
                        f"<div class='banner info'>FINE-TUNING · ROUND {r}/{MAX_ROUNDS}"
                        f"<small>Agent scored {last:.0%} &lt; {CLEAR:.0%}. Training "
                        f"{campaign.ROUND_EPISODES} episodes on Level {lv_now}…</small></div>", unsafe_allow_html=True)
                    progress_box.progress(min(1.0, last / CLEAR), text=f"success {last:.0%} / target {CLEAR:.0%}")
                    q = campaign.finetune_round(lk, q, layout, test_rel, r, seed)
                    m, _ = campaign.check(lk, q, layout, test_rel, seed, N_TEST)
                    rec["hist"].append(dict(round=r, episodes=r * campaign.ROUND_EPISODES, **m))
                    h = pd.DataFrame(rec["hist"])
                    f = go.Figure()
                    for col, lab in (("success_rate", "success"), ("hazard_rate", "hazard"), ("timeout_rate", "timeout")):
                        f.add_trace(go.Scatter(x=h["episodes"], y=h[col], name=lab, mode="lines+markers",
                                               line=dict(color=DARK["status"][lab], width=2)))
                    f.add_hline(y=CLEAR, line=dict(color="#ffd166", dash="dash", width=1))
                    style_fig(f, 220, None)
                    f.update_yaxes(range=[-.03, 1.03], tickformat=".0%")
                    f.update_xaxes(title="fine-tuning episodes")
                    show_chart(f, f"ft_live_{lv_now}_{r}", chart_box)
                    drawn_live = True
                    if m["success_rate"] >= CLEAR:
                        rec["cleared"] = True
                        break
                rec["q"] = q
                progress_box.empty()
            ss.tuned[tkey] = rec
        rec = ss.tuned[tkey]
        q_play = rec["q"] if view == "after" else rec["q0"]
        level_cleared = rec["cleared"]
        hist = rec["hist"]
        start_s, end_s, rounds = hist[0]["success_rate"], hist[-1]["success_rate"], hist[-1]["round"]
        if level_cleared:
            cset.add(lv_now)
            msg = ("CLEARED WITHOUT TRAINING" if rounds == 0 else
                   f"LEVEL {lv_now} CLEARED · {rounds} ROUND{'S' if rounds > 1 else ''}")
            nxt = (f" Level {lv_now + 1} unlocked." if lv_now < N_LEVELS else " Campaign complete! 🏆")
            sub = (f"{start_s:.0%} → {end_s:.0%} success after {rounds * campaign.ROUND_EPISODES} fine-tuning episodes."
                   if rounds else f"{start_s:.0%} success straight away.") + nxt
            status_box.markdown(f"<div class='banner win'>{msg}<small>{sub}</small></div>", unsafe_allow_html=True)
        elif not auto_ft and rounds == 0:
            status_box.markdown(f"<div class='banner fail'>LEVEL FAILED · {start_s:.0%}<small>Turn on auto fine-tune "
                                "to let the agent learn this floor.</small></div>", unsafe_allow_html=True)
        else:
            status_box.markdown(f"<div class='banner fail'>NOT CLEARED · {end_s:.0%}<small>Still below {CLEAR:.0%} "
                                f"after {rounds} rounds. Reset to try again or pick another seed.</small></div>",
                                unsafe_allow_html=True)
        if rounds and not drawn_live:
            h = pd.DataFrame(hist)
            f = go.Figure()
            for col, lab in (("success_rate", "success"), ("hazard_rate", "hazard"), ("timeout_rate", "timeout")):
                f.add_trace(go.Scatter(x=h["episodes"], y=h[col], name=lab, mode="lines+markers",
                                       line=dict(color=DARK["status"][lab], width=2)))
            f.add_hline(y=CLEAR, line=dict(color="#ffd166", dash="dash", width=1))
            style_fig(f, 220, None)
            f.update_yaxes(range=[-.03, 1.03], tickformat=".0%")
            f.update_xaxes(title="fine-tuning episodes")
            show_chart(f, f"ft_hist_{lv_now}", chart_box)

        def _reset(k=tkey):
            ss.tuned.pop(k, None)
            ss.cleared.get((k[0], k[1]), set()).discard(k[2])

        st.button("↺ Reset this level's learning (watch it again)", on_click=_reset)

    # ---- the episodes shown in the arcade
    agent, policy, values = make_agent(kind, layout, seed, test_rel, q_play)
    eps = evaluate(agent, layout, test_rel, N_TEST, seed=seed, record=True)
    summ = summarize(eps)
    if lv_now and not is_learner and summ["success_rate"] >= CLEAR:
        cset.add(lv_now)
        level_cleared = True
        status_box.markdown(f"<div class='banner win'>LEVEL {lv_now} CLEARED<small>{summ['success_rate']:.0%} success. "
                            "This agent does not learn; it plans on the map.</small></div>", unsafe_allow_html=True)
    elif lv_now and not is_learner:
        status_box.markdown(f"<div class='banner fail'>LEVEL FAILED · {summ['success_rate']:.0%}<small>This agent "
                            "cannot learn, so it cannot be fine-tuned.</small></div>", unsafe_allow_html=True)

    # pick a telling episode when the configuration changes
    cfg_key = (kind, seed, ss.map_name, round(test_rel, 2), view, level_cleared)
    if ss.last_cfg != cfg_key:
        ss.last_cfg = cfg_key
        want = "success" if (view == "after" and (level_cleared or not lv_now)) else None
        if lv_now and view == "before":
            want = next((e["status"] for e in eps if e["status"] != "success"), None)
        idx = next((i for i, e in enumerate(eps) if want and e["status"] == want), 0)
        if idx != ep_choice:
            set_episode(idx)   # the episode widget already exists this run -> rebuild it on a rerun
            st.rerun()
    ss.ep_shown = int(ep_choice)
    ep = eps[ss.ep_shown]

    label_bits = {"after": "", "before": " · before fine-tune"}
    label = (f"L{lv_now} · " if lv_now else "") + kind.split(" (")[0] + label_bits[view] + f" · seed {seed}"
    slips = sum(1 for s_ in ep["trajectory"][1:] if s_["intended"] != s_["executed"])
    note = f"Episode {ss.ep_shown}: {ep['status'].upper()} in {ep['steps']} steps, {slips} slips. Space = play/pause."
    components.html(game_html(layout_rows, ep, label, policy, values, round(test_rel, 2), note, autoplay,
                              lv_now, level_cleared and view == "after"), height=830, scrolling=False)
    s = DARK["status"]
    st.markdown(
        "<div class='chips'>"
        f"<span class='chip'><span class='d' style='background:{s['success']}'></span>success {summ['success_rate']:.0%}</span>"
        f"<span class='chip'><span class='d' style='background:{s['hazard']}'></span>hazard {summ['hazard_rate']:.0%}</span>"
        f"<span class='chip'><span class='d' style='background:{s['timeout']}'></span>timeout {summ['timeout_rate']:.0%}</span>"
        f"<span class='chip'>steps→goal {summ['steps_to_goal']:.1f}</span>"
        "</div>", unsafe_allow_html=True)
    opt = f" · optimal policy: {LAYOUTS['level_optimal_success'][lv_now - 1]:.0%}" if lv_now else ""
    st.caption(f"{N_TEST} test episodes at r = {test_rel:.2f} on {ss.map_name}{opt}.")

# ------------------------------------------------------------------ dashboard
with right:
    tabs = st.tabs(["📊 Overview", "🎲 Reliability", "🗺️ Levels", "🏁 Campaign", "🔁 Retraining", "💥 Failures"])

    # ---------------- overview
    with tabs[0]:
        q = e1[e1.agent == "Q-learning"]

        def cell(tr, te, m="success_rate"):
            return q[(q.train_rel == tr) & (np.isclose(q.test_rel, te))][m].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Perfect-trained @ r=0.8", f"{cell('1.00', 0.8):.0%}", f"hazard {cell('1.00', 0.8, 'hazard_rate'):.0%}")
        with c2:
            kpi("Noise-trained (0.9) @ r=0.8", f"{cell('0.90', 0.8):.0%}", f"hazard {cell('0.90', 0.8, 'hazard_rate'):.0%}")
        with c3:
            g = e2[np.isclose(e2.test_rel, LEVEL_R)]
            lvq = g[g.agent == "Local-view Q (frozen)"].groupby("level").success_rate.mean()
            kpi("Local-view Q, levels ≥80%", f"{(lvq >= CLEAR).sum()}/{N_LEVELS}", "zero-shot, no fine-tuning")
        with c4:
            cq = e4[e4.agent == "Q (trained r=0.90)"]
            kpi("Cell-Q campaign", f"{cq.cleared.mean():.0%}", f"levels cleared · median {cq.rounds.median():.0f} round(s)")
        st.write("")
        st.markdown((ROOT / "report" / "findings.html").read_text(encoding="utf-8")
                    if (ROOT / "report" / "findings.html").exists() else "", unsafe_allow_html=True)
        fig = go.Figure()
        for tr, gg in lc.groupby("train_rel"):
            lab, col, _ = agent_style("Q-learning", tr, dark=True)
            a = gg.groupby("episode")["mean_return"].mean()
            fig.add_trace(go.Scatter(x=a.index, y=a.values, name=lab, line=dict(color=col, width=2),
                                     hovertemplate="ep %{x}<br>return %{y:.2f}<extra>" + lab + "</extra>"))
        style_fig(fig, 320, "Training curves on the training map (mean return, 10 seeds)")
        fig.update_xaxes(title="training episode")
        show_chart(fig, "overview_curves")

    # ---------------- reliability
    with tabs[1]:
        mcol, acol = st.columns([1, 2])
        metric = mcol.selectbox("Metric", ["success_rate", "hazard_rate", "timeout_rate", "steps_to_goal", "mean_return"],
                                format_func=lambda m: m.replace("_", " "))
        groups = sorted(e1.groupby(["agent", "train_rel"]).groups.keys(), key=lambda k: (k[0] != "Q-learning", k))

        def gname(g):
            if g[0] == "Local-view Q":
                return "Local-view Q (random maps)"
            if g[0] == "Risk-aware planner":
                return "Risk-aware planner"
            return agent_style(*g, dark=True)[0]

        default = [g for g in groups if g[1] in ("1.00", "0.90", "0.70") or g[1].startswith("randomized")
                   or g[0].startswith("Shortest")]
        chosen = acol.multiselect("Agents", groups, default=default, format_func=gname)
        extra = {"Local-view Q": ("#c9a8ff", "solid"), "Risk-aware planner": ("#ffd166", "dashdot")}
        fig = go.Figure()
        for g in chosen:
            lab = gname(g)
            col, dash = extra.get(g[0], agent_style(*g, dark=True)[1:])
            d = e1[(e1.agent == g[0]) & (e1.train_rel == g[1])]
            a = d.groupby("test_rel")[metric].agg(["mean", ci95]).reset_index().sort_values("test_rel", ascending=False)
            fig.add_trace(go.Scatter(x=a.test_rel, y=a["mean"], name=lab, mode="lines+markers",
                                     line=dict(color=col, width=2, dash=DASHES[dash]), marker=dict(size=8),
                                     error_y=dict(type="data", array=a["ci95"], color=col, thickness=1, width=3),
                                     hovertemplate="test r=%{x:.2f}<br>%{y:.3f}<extra>" + lab + "</extra>"))
        style_fig(fig, 400, f"{metric.replace('_', ' ')} vs test reliability (mean ± 95% CI, 10 seeds)")
        fig.update_xaxes(title="test movement reliability r (→ more slip)", autorange="reversed")
        show_chart(fig, "rel_lines")

        hm_metric = st.radio("Heatmap", ["success_rate", "hazard_rate", "timeout_rate"], horizontal=True,
                             format_func=lambda m: m.replace("_", " "))
        p = e1[e1.agent == "Q-learning"].pivot_table(index="train_rel", columns="test_rel", values=hm_metric)
        order = TRAIN_LEVELS + [i for i in p.index if i.startswith("randomized")]
        p = p.reindex(order)[sorted(p.columns, reverse=True)]
        scale = {"success_rate": "Blues", "hazard_rate": "Reds", "timeout_rate": "YlOrBr"}[hm_metric]
        fig = go.Figure(go.Heatmap(z=p.values, x=[f"{c:.2f}" for c in p.columns], y=list(p.index), zmin=0, zmax=1,
                                   colorscale=scale, text=np.round(p.values, 2), texttemplate="%{text}",
                                   hovertemplate="train %{y}<br>test %{x}<br>%{z:.2f}<extra></extra>"))
        style_fig(fig, 330, "Q-learning: train reliability × test reliability")
        fig.update_xaxes(title="test r (0.6, 0.5 never seen in training)", type="category")
        fig.update_yaxes(title="training r", type="category", autorange="reversed")
        show_chart(fig, "rel_heat")

    # ---------------- levels (E2)
    with tabs[2]:
        te = st.radio("Test reliability", sorted(e2.test_rel.unique(), reverse=True), horizontal=True, index=1,
                      format_func=lambda v: f"r = {v:.1f}")
        lm = st.radio("Metric  ", ["success_rate", "hazard_rate", "timeout_rate"], horizontal=True,
                      format_func=lambda m: m.replace("_", " "))
        d = e2[np.isclose(e2.test_rel, te)]
        order = ["Optimal (VI)", "Risk-aware planner", "Shortest-path (replans)", "Local-view Q (frozen)",
                 "Q frozen (1.00)", "Q frozen (0.90)", "Q frozen (0.80)", "Q frozen (randomized 0.7-1.0)", "Random"]
        p = d.pivot_table(index="agent", columns="level", values=lm).reindex([o for o in order if o in d.agent.unique()])
        scale = {"success_rate": "Blues", "hazard_rate": "Reds", "timeout_rate": "YlOrBr"}[lm]
        fig = go.Figure(go.Heatmap(z=p.values, x=[f"L{c}" for c in p.columns], y=list(p.index), zmin=0, zmax=1,
                                   colorscale=scale, text=np.round(p.values, 2), texttemplate="%{text}",
                                   hovertemplate="%{y}<br>%{x}<br>%{z:.2f}<extra></extra>"))
        style_fig(fig, 420, f"Frozen agents on the 10 levels, before any fine-tuning (r = {te:.1f}, 10 seeds)")
        fig.update_yaxes(autorange="reversed")
        show_chart(fig, "levels_heat")
        st.caption("Cell-index Q-tables memorise the training floor, so they walk old routes into moved hazards or "
                   "bump new shelves. The local-view agent sees its neighbours plus a planner hint, so it transfers. "
                   "Optimal (VI) knows the map and the slip rate: the ceiling for each level.")
        cols = st.columns(5)
        colour = {"#": "🟦", "H": "🟥", "S": "🔷", "G": "🟨", ".": "⬛"}
        for i, rows in enumerate(LAYOUTS["levels"]):
            cols[i % 5].markdown(f"**Level {i + 1}** · {sum(r.count('H') for r in rows)} hazards<br>"
                                 "<span style='font-size:10px;line-height:1'>" +
                                 "<br>".join("".join(colour[ch] for ch in r) for r in rows) + "</span>",
                                 unsafe_allow_html=True)

    # ---------------- campaign (E4)
    with tabs[3]:
        st.markdown(f"Each agent starts every level from its original Q-table and fine-tunes in "
                    f"{campaign.ROUND_EPISODES}-episode rounds until it reaches {CLEAR:.0%} success "
                    f"(max {MAX_ROUNDS} rounds). This is exactly what the arcade does live.")
        agg = e4.groupby(["agent", "level"]).agg(rounds=("rounds", "mean"), cleared=("cleared", "mean")).reset_index()
        palette = {"Q (trained r=1.00)": DARK["q"]["1.00"], "Q (trained r=0.90)": DARK["q"]["0.90"],
                   "Q (randomized 0.7-1.0)": DARK["randomized"], "Local-view Q": "#c9a8ff"}
        fig = go.Figure()
        for a_name, gg in agg.groupby("agent"):
            fig.add_trace(go.Bar(x=[f"L{l}" for l in gg.level], y=gg.rounds, name=a_name,
                                 marker_color=palette.get(a_name, "#888"),
                                 marker_line=dict(color=DARK["surface"], width=2),
                                 customdata=gg.cleared,
                                 hovertemplate="%{x}<br>mean rounds %{y:.1f}<br>cleared in %{customdata:.0%} of seeds"
                                               "<extra>" + a_name + "</extra>"))
        style_fig(fig, 380, "Fine-tuning rounds needed to clear each level (mean over 10 seeds; 0 = cleared zero-shot)")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title=f"rounds of {campaign.ROUND_EPISODES} episodes")
        show_chart(fig, "campaign_bars")
        t = e4.groupby("agent").agg(levels_cleared=("cleared", "mean"), zero_shot=("rounds", lambda x: (x == 0).mean()),
                                    mean_episodes=("episodes", "mean"),
                                    final_success=("final_success", "mean")).reset_index()
        t["levels_cleared"] = (t.levels_cleared * 100).round(0).astype(int).astype(str) + "%"
        t["zero_shot"] = (t.zero_shot * 100).round(0).astype(int).astype(str) + "%"
        t["mean_episodes"] = t.mean_episodes.round(0).astype(int)
        t["final_success"] = (t.final_success * 100).round(0).astype(int).astype(str) + "%"
        show_table(t.rename(columns={"agent": "starting agent", "levels_cleared": "cleared",
                                     "zero_shot": "cleared with 0 rounds", "mean_episodes": "mean fine-tune episodes",
                                     "final_success": "final success"}), hide_index=True)

    # ---------------- retraining (E3)
    with tabs[4]:
        rm = st.radio("Metric ", ["success_rate", "hazard_rate", "timeout_rate"], horizontal=True,
                      format_func=lambda m: m.replace("_", " "))
        lvsel = st.selectbox("Level ", ["All levels"] + [f"Level {i}" for i in range(1, N_LEVELS + 1)])
        d = e3 if lvsel == "All levels" else e3[e3.level == int(lvsel.split()[1])]
        mode_col = {"cell Q · fine-tune (ε 1.0→0.05)": DARK["q"]["0.80"], "cell Q · fine-tune (ε 0.3→0.05)": DARK["fine"],
                    "cell Q · from scratch": DARK["scratch"], "local-view Q · fine-tune (ε 0.3→0.05)": "#c9a8ff"}
        fig = go.Figure()
        for mode, col in mode_col.items():
            ps = d[d["mode"] == mode].groupby(["budget", "seed"])[rm].mean().reset_index()
            a = ps.groupby("budget")[rm].agg(["mean", ci95]).reset_index()
            fig.add_trace(go.Scatter(x=[str(b) for b in a.budget], y=a["mean"], name=mode, mode="lines+markers",
                                     line=dict(color=col, width=2.5), marker=dict(size=9),
                                     error_y=dict(type="data", array=a["ci95"], color=col, thickness=1, width=3)))
        style_fig(fig, 420, f"Fixed retraining budgets on the levels (r = {LEVEL_R:.1f}, mean ± 95% CI)")
        fig.update_xaxes(title="retraining episodes on the level", type="category")
        fig.update_yaxes(range=[-.03, 1.03], tickformat=".0%")
        show_chart(fig, "retrain_lines")
        st.caption("Restarting exploration at ε = 1.0 throws away much of what the old table knew; a low starting ε "
                   "(0.3) lets fine-tuning reuse it.")

    # ---------------- failure explorer
    with tabs[5]:
        st.markdown(f"**Current arcade config:** {label} · r = {test_rel:.2f} · {ss.map_name}")
        rows_tbl = []
        for i, e in enumerate(eps):
            if e["status"] == "success":
                continue
            tr = e["trajectory"]
            sl = sum(1 for s_ in tr[1:] if s_["intended"] != s_["executed"])
            last_slip = tr[-1]["intended"] != tr[-1]["executed"] if len(tr) > 1 else False
            rows_tbl.append({"episode": i, "outcome": e["status"], "steps": e["steps"], "slips": sl,
                             "final cell": tuple(tr[-1]["pos"]), "fatal step was a slip": last_slip})
        if not rows_tbl:
            st.success(f"No failures in {N_TEST} test episodes for this configuration.")
        else:
            fdf = pd.DataFrame(rows_tbl)
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi("Failures", f"{len(fdf)}/{N_TEST}")
            with c2:
                hz = fdf[fdf.outcome == "hazard"]
                kpi("Hazard deaths caused by a slip",
                    f"{hz['fatal step was a slip'].mean():.0%}" if len(hz) else "—", f"{len(hz)} hazard episodes")
            with c3:
                kpi("Timeouts", f"{(fdf.outcome == 'timeout').sum()}", "agent stalled or looped")
            z = np.zeros((8, 8))
            for fc in fdf["final cell"]:
                z[fc[0], fc[1]] += 1
            text = [[("■" if layout_rows[r][c] == "#" else "G" if layout_rows[r][c] == "G" else
                      (str(int(z[r, c])) if z[r, c] else ("H" if layout_rows[r][c] == "H" else ""))) for c in range(8)]
                    for r in range(8)]
            fig = go.Figure(go.Heatmap(z=z, text=text, texttemplate="%{text}", colorscale=[[0, "#0b1222"], [1, "#ff5a7a"]],
                                       showscale=False, xgap=3, ygap=3,
                                       hovertemplate="row %{y}, col %{x}<br>%{z} failures end here<extra></extra>"))
            style_fig(fig, 330, "Where failed episodes end (count per cell)")
            fig.update_yaxes(autorange="reversed", showgrid=False, dtick=1)
            fig.update_xaxes(showgrid=False, dtick=1)
            show_chart(fig, "fail_heat")
            show_table(fdf, hide_index=True, height=220)
            pick = st.selectbox("Pick a failure to replay", fdf.episode.tolist(),
                                format_func=lambda i: f"Episode {i} · {fdf.set_index('episode').loc[i, 'outcome']} · "
                                                      f"{fdf.set_index('episode').loc[i, 'steps']} steps")

            def _replay(i=pick):
                set_episode(i)

            st.button("▶ Replay this failure in the arcade", on_click=_replay, type="primary")
