from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import re


@dataclass(frozen=True)
class NormalizedEmail:
    normalized_text: str
    meta: Dict[str, object]


def normalize_email(text: str) -> NormalizedEmail:
    original_length = len(text)
    collapsed = re.sub(r"\r\n", "\n", text)
    collapsed = re.sub(r"[\t ]{2,}", " ", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    collapsed = collapsed.strip()

    meta = {
        "original_length": original_length,
        "normalized_length": len(collapsed),
        "collapsed_blank_lines": True,
        "normalized_whitespace": True,
    }
    return NormalizedEmail(normalized_text=collapsed, meta=meta)
