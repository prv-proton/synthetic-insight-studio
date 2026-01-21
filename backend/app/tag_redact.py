import re
from typing import Dict, Tuple


TAG_TOKEN_MAP = {
    "PII:NAME": "[NAME]",
    "PII:EMAIL": "[EMAIL]",
    "PII:PHONE": "[PHONE]",
    "PII:ADDRESS": "[ADDRESS]",
    "PII:PARCEL_ID": "[PARCEL_ID]",
    "PII:FILE_NO": "[FILE_NO]",
    "SENSITIVE:ATTACHMENT": "[ATTACHMENT]",
    "SENSITIVE:AGENCY_COMMENT": "[AGENCY_COMMENT]",
    "CONFIDENTIAL:FINANCING": "[FINANCING]",
    "CONFIDENTIAL:LEGAL": "[LEGAL]",
}

ADDRESS_PATTERN = (
    r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,4}"
    r"(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|"
    r"way|wy|circle|cir)\b"
    r"(?:,\s*[A-Za-z.\s]+)?"
    r"(?:,\s*[A-Za-z.\s]+)?"
    r"(?:\s+\d{5}(?:-\d{4})?)?"
)

TAG_VALUE_PATTERNS = {
    "PII:NAME": r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}",
    "PII:EMAIL": r"[\w.+-]+@[\w.-]+\.\w+",
    "PII:PHONE": r"\+?\d[\d\s().-]{6,}\d",
    "PII:ADDRESS": ADDRESS_PATTERN,
    "PII:PARCEL_ID": r"(?:PID\s*)?\d{3}-\d{3}-\d{3}",
    "PII:FILE_NO": r"(?:DP|BP)-\d{2}-\d{3,5}",
    "SENSITIVE:ATTACHMENT": r"[^\n]{0,120}?(?=(?:\[\[|$|\n|[.!?](?:\s|$)))",
    "SENSITIVE:AGENCY_COMMENT": r"[^\n]{0,120}?(?=(?:\[\[|$|\n|[.!?](?:\s|$)))",
    "CONFIDENTIAL:FINANCING": r"[^\n]{0,120}?(?=(?:\[\[|$|\n|[.!?](?:\s|$)))",
    "CONFIDENTIAL:LEGAL": r"[^\n]{0,120}?(?=(?:\[\[|$|\n|[.!?](?:\s|$)))",
}


def redact_tagged(text: str) -> Tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    redacted = text
    for tag, token in TAG_TOKEN_MAP.items():
        value_pattern = TAG_VALUE_PATTERNS.get(tag)
        tag_pattern = re.escape(tag)
        total_count = 0
        if value_pattern:
            pattern_with_value = re.compile(
                rf"\[\[{tag_pattern}\]\]\s*({value_pattern})",
                re.IGNORECASE,
            )
            redacted, count = pattern_with_value.subn(token, redacted)
            total_count += count
            pattern_fallback = re.compile(
                rf"\[\[{tag_pattern}\]\]\s*(\S+)",
                re.IGNORECASE,
            )
            redacted, count = pattern_fallback.subn(token, redacted)
            total_count += count
        pattern = re.compile(rf"\[\[{tag_pattern}\]\]", re.IGNORECASE)
        redacted, count = pattern.subn(token, redacted)
        total_count += count
        if total_count:
            stats[tag] = total_count
    return redacted, stats


if __name__ == "__main__":
    sample = (
        "Subject: Permit update for [[PII:ADDRESS]] 1186 Cedar Ridge Ave, Victoria, BC\n"
        "Contact: [[PII:NAME]] Jane Doe / [[PII:EMAIL]] jane@example.com\n"
        "PID: [[PII:PARCEL_ID]] 009-671-324 and file [[PII:FILE_NO]] DP-24-1682\n"
        "Attachments: [[SENSITIVE:ATTACHMENT]] site-plan-v2.pdf\n"
        "Notes: [[CONFIDENTIAL:LEGAL]] pending easement review."
    )
    result, stats = redact_tagged(sample)
    print(result)
    print(stats)
