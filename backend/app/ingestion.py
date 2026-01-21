from __future__ import annotations

import re
from typing import Dict, List, Tuple


EMAIL_MARKERS = {
    "from_line": re.compile(r"^from:", re.IGNORECASE),
    "to_line": re.compile(r"^to:", re.IGNORECASE),
    "subject_line": re.compile(r"^subject:", re.IGNORECASE),
    "quoted_line": re.compile(r"^>", re.IGNORECASE),
    "on_wrote": re.compile(r"^on .+ wrote:", re.IGNORECASE),
}
SIGNATURE_MARKERS = (
    "Regards,",
    "--",
    "Sent from",
)
ATTACHMENT_HINTS = re.compile(r"\b(attachment|attached|see attached|attachment:)\b", re.IGNORECASE)

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]


def trim_excess_whitespace(text: str) -> str:
    trimmed = text.strip()
    trimmed = re.sub(r"[\t ]{2,}", " ", trimmed)
    trimmed = re.sub(r"\n{3,}", "\n\n", trimmed)
    return trimmed


def _count_email_markers(text: str) -> int:
    markers_found = 0
    lines = text.splitlines()
    if any(EMAIL_MARKERS["from_line"].match(line) for line in lines):
        markers_found += 1
    if any(EMAIL_MARKERS["to_line"].match(line) for line in lines):
        markers_found += 1
    if any(EMAIL_MARKERS["subject_line"].match(line) for line in lines):
        markers_found += 1
    if any(EMAIL_MARKERS["on_wrote"].match(line) for line in lines):
        markers_found += 1
    if any(EMAIL_MARKERS["quoted_line"].match(line) for line in lines):
        markers_found += 1
    if any(marker.lower() in text.lower() for marker in SIGNATURE_MARKERS):
        markers_found += 1
    return markers_found


def infer_source_type(text: str) -> str:
    return "email_like" if _count_email_markers(text) >= 2 else "plain_text"


def detect_quoted_thread(text: str) -> bool:
    lines = text.splitlines()
    return any(EMAIL_MARKERS["quoted_line"].match(line) for line in lines) or any(
        EMAIL_MARKERS["on_wrote"].match(line) for line in lines
    )


def detect_attachment_hints(text: str) -> bool:
    return bool(ATTACHMENT_HINTS.search(text))


def assess_risk(
    redaction_stats: Dict[str, int],
    post_redaction_findings: List[Dict[str, str]],
    email_like: bool,
    quoted_thread: bool,
    attachment_hints: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    total_redactions = sum(redaction_stats.values())
    level_index = 0

    if total_redactions >= 10:
        level_index = 2
        reasons.append("High PII redaction volume")
    elif total_redactions >= 5:
        level_index = max(level_index, 1)
        reasons.append("Moderate PII redaction volume")

    if post_redaction_findings:
        level_index = max(level_index, 1)
        reasons.append("Residual PII detected after redaction")

    if email_like and (quoted_thread or attachment_hints):
        level_index = min(level_index + 1, 2)
        if quoted_thread:
            reasons.append("Quoted thread detected")
        if attachment_hints:
            reasons.append("Attachment hints detected")

    if not reasons:
        reasons.append("Standard risk profile")

    return RISK_LEVELS[level_index], reasons


def should_store_sanitized(risk_level: str) -> bool:
    return risk_level != "HIGH"
