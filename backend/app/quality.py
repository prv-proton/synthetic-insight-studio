from __future__ import annotations

import re
from typing import Dict, List


EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s().-]{6,}\d")
LONG_ID_PATTERN = re.compile(r"\b\d{7,}\b")


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


def evaluate_pseudo_email(email: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    body = str(email.get("body") or "")
    motivations = email.get("motivations") or []
    decision_points = email.get("decision_points") or []
    paragraphs = [p for p in body.split("\n\n") if p.strip()]

    if len(body) < 700 and len(paragraphs) < 3:
        issues.append("body should be >=700 chars or at least 3 paragraphs")
    if len(motivations) < 2:
        issues.append("motivations should include at least 2 items")
    if len(decision_points) < 1:
        issues.append("decision_points should include at least 1 item")
    if _body_is_listy(body):
        issues.append("body reads like a list of questions")
    if EMAIL_PATTERN.search(body) or PHONE_PATTERN.search(body) or LONG_ID_PATTERN.search(body):
        issues.append("body contains potential PII patterns")
    return issues


def _body_is_listy(body: str) -> bool:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return True
    bullet_lines = [line for line in lines if line.startswith("-") or line.startswith("*")]
    if bullet_lines and len(bullet_lines) >= len(lines) * 0.6:
        return True
    return False
