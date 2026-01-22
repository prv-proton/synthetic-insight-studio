from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .pii import redact
from .tag_redact import redact_tagged


HEADER_LINE = re.compile(r"^(from|to|cc|bcc|subject|date|sent):", re.IGNORECASE)
FROM_LINE = re.compile(r"^from:\s*(.+)", re.IGNORECASE)
SUBJECT_LINE = re.compile(r"^subject:\s*(.+)", re.IGNORECASE)
ON_WROTE = re.compile(r"^on .+ wrote:", re.IGNORECASE)
ORIGINAL_MESSAGE = re.compile(r"^-{2,}\s*original message\s*-{2,}$", re.IGNORECASE)
QUOTE_LINE = re.compile(r"^>", re.IGNORECASE)
MIME_LINE = re.compile(
    r"^(content-type|content-transfer-encoding|mime-version):",
    re.IGNORECASE,
)
BOUNDARY_LINE = re.compile(r"^--[A-Za-z0-9'()+=_,./:-]{6,}$")

DISCLAIMER_MARKERS = (
    "confidential",
    "privileged",
    "intended only",
    "do not distribute",
    "unauthorized",
    "virus",
    "disclaimer",
)
SIGNATURE_MARKERS = (
    "regards,",
    "sincerely,",
    "thanks,",
    "thank you,",
    "sent from",
    "--",
)

STAFF_HINTS = (
    "@gov",
    ".gov",
    ".ca",
    "city of",
    "county",
    "district",
    "planning",
    "permit",
    "navigator",
    "agency",
    "department",
)

EXTRA_PATTERNS: Dict[str, re.Pattern[str]] = {
    "PARCEL_ID": re.compile(r"\b\d{3}-\d{3}-\d{3}\b"),
    "FILE_NO": re.compile(r"\b[A-Z]{1,3}-\d{2}-\d{3,5}\b", re.IGNORECASE),
    "POSTAL": re.compile(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", re.IGNORECASE),
    "ADDRESS": re.compile(
        r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,4}"
        r"(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|"
        r"way|wy|circle|cir)\b",
        re.IGNORECASE,
    ),
}


def infer_email_like(text: str) -> bool:
    markers = 0
    lines = text.splitlines()
    if any(HEADER_LINE.match(line) for line in lines):
        markers += 1
    if any(ON_WROTE.match(line) for line in lines):
        markers += 1
    if any(QUOTE_LINE.match(line) for line in lines):
        markers += 1
    if any(marker in text.lower() for marker in SIGNATURE_MARKERS):
        markers += 1
    return markers >= 2


def normalize_email_text(raw: str) -> Dict[str, object]:
    collapsed = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = collapsed.splitlines()

    removed_headers = False
    removed_mime = False
    removed_disclaimers = False
    removed_signatures = False
    removed_quotes = False
    subject_line = None

    header_end = None
    for idx, line in enumerate(lines[:40]):
        if HEADER_LINE.match(line):
            removed_headers = True
            if SUBJECT_LINE.match(line):
                subject_line = SUBJECT_LINE.match(line).group(1).strip()
            continue
        if removed_headers and line.strip() == "":
            header_end = idx + 1
            break
        if removed_headers and not HEADER_LINE.match(line):
            header_end = idx
            break
    if removed_headers and header_end is not None:
        lines = lines[header_end:]

    cleaned_lines: List[str] = []
    for line in lines:
        if MIME_LINE.match(line) or BOUNDARY_LINE.match(line):
            removed_mime = True
            continue
        if "boundary=" in line.lower():
            removed_mime = True
            continue
        cleaned_lines.append(line)

    cleaned_lines, removed_disclaimers = _remove_disclaimer_blocks(cleaned_lines)
    cleaned_lines, removed_signatures = _remove_signature_blocks(cleaned_lines)

    cleaned_full = _collapse_blank_lines("\n".join(cleaned_lines)).strip()

    latest_message = cleaned_full
    if cleaned_full:
        split_index = _find_quote_break(cleaned_full.splitlines())
        if split_index is not None:
            removed_quotes = True
            latest_message = "\n".join(cleaned_full.splitlines()[:split_index]).strip()

    meta = {
        "removed_headers": removed_headers,
        "removed_signatures": removed_signatures,
        "removed_disclaimers": removed_disclaimers,
        "removed_mime": removed_mime,
        "removed_quotes": removed_quotes,
        "subject": subject_line,
    }
    return {
        "cleaned_full": cleaned_full,
        "latest_message": latest_message,
        "meta": meta,
    }


def split_thread_into_turns(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    turns: List[Dict[str, str]] = []
    current_lines: List[str] = []
    current_hint = ""
    in_quote = False

    def flush() -> None:
        nonlocal current_lines, current_hint
        body = "\n".join(current_lines).strip()
        if body:
            role, hint = _infer_speaker(current_hint or body)
            turns.append(
                {
                    "speaker_role": role,
                    "speaker_hint": hint,
                    "body": body,
                }
            )
        current_lines = []
        current_hint = ""

    for line in lines:
        if ON_WROTE.match(line) or ORIGINAL_MESSAGE.match(line):
            if current_lines:
                flush()
            current_hint = line.strip()
            continue

        if FROM_LINE.match(line):
            if current_lines:
                flush()
            current_hint = FROM_LINE.match(line).group(1).strip()
            continue

        if HEADER_LINE.match(line):
            if SUBJECT_LINE.match(line) and not current_hint:
                current_hint = SUBJECT_LINE.match(line).group(1).strip()
            continue

        if QUOTE_LINE.match(line):
            if not in_quote:
                if current_lines:
                    flush()
                in_quote = True
            current_lines.append(line.lstrip("> ").rstrip())
            continue

        if in_quote and line.strip() == "":
            current_lines.append("")
            continue

        if in_quote and line.strip():
            if current_lines:
                flush()
            in_quote = False

        current_lines.append(line)

    if current_lines:
        flush()

    return turns


def redact_uniform(text: str) -> Tuple[str, Dict[str, int]]:
    tagged_text, tag_stats = redact_tagged(text)
    redacted_text, pii_stats = redact(tagged_text)
    stats: Dict[str, int] = {}
    stats.update(tag_stats)
    stats.update(pii_stats)
    for label, pattern in EXTRA_PATTERNS.items():
        redacted_text, count = pattern.subn(f"[{label}]", redacted_text)
        if count:
            stats[label] = stats.get(label, 0) + count
    return redacted_text, stats


def _remove_disclaimer_blocks(lines: List[str]) -> Tuple[List[str], bool]:
    if not lines:
        return lines, False
    removed = False
    output: List[str] = []
    skip = False
    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in DISCLAIMER_MARKERS):
            removed = True
            skip = True
        if not skip:
            output.append(line)
        if skip and line.strip() == "":
            skip = False
    return output, removed


def _remove_signature_blocks(lines: List[str]) -> Tuple[List[str], bool]:
    if not lines:
        return lines, False
    tail_window = 12
    removed = False
    start_index = None
    for idx in range(len(lines) - 1, max(-1, len(lines) - tail_window - 1), -1):
        if lines[idx].strip().lower() in SIGNATURE_MARKERS:
            start_index = idx
            break
    if start_index is not None:
        removed = True
        return lines[:start_index], removed
    return lines, removed


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _find_quote_break(lines: List[str]) -> int | None:
    for idx, line in enumerate(lines):
        if ON_WROTE.match(line) or ORIGINAL_MESSAGE.match(line):
            return idx
        if FROM_LINE.match(line):
            return idx
        if QUOTE_LINE.match(line):
            return idx
    return None


def _infer_speaker(hint: str) -> Tuple[str, str]:
    normalized = hint.lower()
    if any(token in normalized for token in STAFF_HINTS):
        return "staff", hint.strip()
    if hint.strip():
        return "citizen", hint.strip()
    return "unknown", ""
