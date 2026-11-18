"""One colour system for every chart (static figures + dashboard).

Q-learning agents are ordered by training reliability, so they share ONE
sequential blue ramp (light = perfect movement, dark = most slip).
Other agents take distinct categorical hues; baselines are dashed/dotted.
"""

# sequential blue ramp (ordinal steps, all >= 2:1 on light surface)
Q_COLORS = {
    "1.00": "#86b6ef",
    "0.95": "#5598e7",
    "0.90": "#2a78d6",
    "0.80": "#1c5cab",
    "0.70": "#104281",
}
RANDOMIZED_COLOR = "#eb6834"   # categorical orange
SHORTEST_COLOR = "#1baf7a"     # categorical aqua
ORACLE_COLOR = "#52514e"       # neutral ink (reference line)
RANDOM_COLOR = "#a3a29c"       # recessive gray
LOCAL_COLOR = "#6250d6"        # categorical violet
SAFE_PLANNER_COLOR = "#eda100"  # categorical yellow

FINE_TUNE_COLOR = "#2a78d6"
SCRATCH_COLOR = "#eb6834"

# status colours, reserved for outcomes (always shipped with labels)
STATUS = {"success": "#1baf7a", "hazard": "#e34948", "timeout": "#eda100"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"


# Dark-surface steps of the same hues (dashboard runs on a dark surface)
DARK = {
    "q": {"1.00": "#cde2fb", "0.95": "#9ec5f4", "0.90": "#6da7ec", "0.80": "#3987e5", "0.70": "#256abf"},
    "randomized": "#ff8a4c",
    "shortest": "#3dd6a0",
    "oracle": "#c3c2b7",
    "random": "#6b6a64",
    "fine": "#6da7ec",
    "scratch": "#ff8a4c",
    "status": {"success": "#3dd6a0", "hazard": "#ff5a7a", "timeout": "#f5b73b"},
    "surface": "#060a14", "panel": "#0b1222", "ink": "#e8f1ff", "ink2": "#8ea3c7", "grid": "#1b2a48",
}


def agent_style(agent, train_rel, dark=False):
    """Return (label, colour, dash) for an E1 row."""
    if agent == "Q-learning":
        if str(train_rel).startswith("randomized"):
            return f"Q · {train_rel}", DARK["randomized"] if dark else RANDOMIZED_COLOR, "solid"
        ramp = DARK["q"] if dark else Q_COLORS
        return f"Q · trained r={train_rel}", ramp.get(str(train_rel), "#2a78d6"), "solid"
    if agent.startswith("Local-view"):
        return "Local-view Q (random maps)", "#c9a8ff" if dark else LOCAL_COLOR, "solid"
    if agent.startswith("Risk-aware"):
        return "Risk-aware planner", "#ffd166" if dark else SAFE_PLANNER_COLOR, "dashdot"
    if agent.startswith("Optimal"):
        return "Optimal (knows test r)", DARK["oracle"] if dark else ORACLE_COLOR, "dash"
    if agent.startswith("Shortest"):
        return "Shortest-path", DARK["shortest"] if dark else SHORTEST_COLOR, "dashdot"
    return "Random", DARK["random"] if dark else RANDOM_COLOR, "dot"
