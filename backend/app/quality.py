from __future__ import annotations

from typing import Dict, List


def evaluate_tco(tco: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    goals = tco.get("goals") or []
    constraints = tco.get("constraints") or []
    blockers = tco.get("blockers") or []
    decision_points = tco.get("decision_points") or []
    asking = tco.get("what_they_are_asking") or []

    if len(goals) < 2:
        issues.append("goals needs at least 2 items")
    if len(blockers) < 2 and len(constraints) < 2:
        issues.append("need at least 2 blockers or 2 constraints")
    if len(decision_points) < 1:
        issues.append("decision_points required")
    if len(asking) < 2:
        issues.append("what_they_are_asking needs at least 2 items")
    return issues
