import re
from typing import Dict, List, Tuple


PATTERNS = {
    "EMAIL": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE),
    "PHONE": re.compile(
        r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b"
    ),
    "URL": re.compile(r"\bhttps?://[^\s]+\b", re.IGNORECASE),
    "PARCEL_ID": re.compile(r"\bPID[:#]?\s*\d{3}-\d{3}-\d{3}\b", re.IGNORECASE),
    "FILE_NO": re.compile(r"\b(?:DP|BP)-\d{2}-\d{3,5}\b", re.IGNORECASE),
    "ADDRESS": re.compile(
        r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,4}"
        r"(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|"
        r"way|wy|circle|cir)\b"
        r"(?:,\s*[A-Za-z.\s]+)?"
        r"(?:,\s*[A-Za-z.\s]+)?"
        r"(?:\s+\d{5}(?:-\d{4})?)?",
        re.IGNORECASE,
    ),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "POSTAL": re.compile(r"\b\d{5}(-\d{4})?\b"),
    "DATE": re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
        re.IGNORECASE,
    ),
    "NAME": re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"),
    "ID": re.compile(r"\b\d{7,}\b"),
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
    redacted, at_count = re.subn("@", "[EMAIL]", redacted)
    if at_count:
        stats["EMAIL"] = stats.get("EMAIL", 0) + at_count
    redacted, id_count = re.subn(r"\d{7,}", "[ID]", redacted)
    if id_count:
        stats["ID"] = stats.get("ID", 0) + id_count
    return redacted, stats
