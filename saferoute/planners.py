"""Risk-aware planner baseline: always follows the local-view agent's hint.

It plans on the true floor map with Dijkstra, charging extra for cells next
to hazards (see local_view.risk_aware_distance). It needs no learning, but it
does not know the slip probability either. Comparing it with Local-view Q
shows what learning adds on top of the hint.
"""
from __future__ import annotations

from .local_view import suggested_actions


class SafePlannerAgent:
    name = "Risk-aware planner"

    def __init__(self, layout):
        self.policy = suggested_actions(layout)

    def act(self, s):
        return int(self.policy[s])
