from __future__ import annotations

import re


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
