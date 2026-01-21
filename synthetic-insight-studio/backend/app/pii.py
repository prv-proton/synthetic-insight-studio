import re
from typing import Dict, List, Tuple


PATTERNS = {
    "EMAIL": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE),
    "PHONE": re.compile(
        r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b"
    ),
    "URL": re.compile(r"\bhttps?://[^\s]+\b", re.IGNORECASE),
    "POSTAL": re.compile(r"\b\d{5}(-\d{4})?\b"),
    "ID": re.compile(r"\b\d{9,}\b"),
    "DATE": re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
        re.IGNORECASE,
    ),
}


def detect_pii(text: str) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"type": label, "value": match.group(0)})
    return findings


def redact(text: str) -> Tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    redacted = text
    for label, pattern in PATTERNS.items():
        redacted, count = pattern.subn(f"[{label}]", redacted)
        if count:
            stats[label] = count
    return redacted, stats
