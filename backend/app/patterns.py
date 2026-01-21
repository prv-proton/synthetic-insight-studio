import re
from collections import Counter
from typing import Dict, Iterable, List


THEMES = {
    "Inspections & closeout": [
        "inspection",
        "framing",
        "foundation",
        "footing",
        "blower door",
        "letters of assurance",
        "loa",
        "final",
        "issuance",
    ],
    "Permitting intake & status": [
        "permit",
        "dp",
        "bp",
        "intake",
        "submitted",
        "processing",
        "status",
        "timeline",
        "pending",
    ],
    "Site constraints & environmental review": [
        "riparian",
        "creek",
        "qep",
        "stormwater",
        "arborist",
        "tree",
        "covenant",
        "geotech",
    ],
    "Design revisions & review comments": [
        "comment",
        "revision",
        "resubmit",
        "blocking",
        "coordination",
        "setback",
        "circulation",
    ],
    "Fire access & safety": [
        "fire",
        "access",
        "swept path",
        "turning",
        "template",
        "emergency",
    ],
    "Expedite & financing pressure": [
        "expedite",
        "lender",
        "financing",
        "term sheet",
        "interest",
        "carry costs",
        "deadline",
    ],
    "Document upload & technical issues": [
        "upload",
        "document",
        "error",
        "portal",
        "technical",
    ],
    "Account access & authentication": [
        "login",
        "password",
        "authenticate",
        "account",
        "verify",
    ],
    "Payment & fees": ["fee", "payment", "refund", "charge", "invoice"],
    "Appeals & reconsideration": ["appeal", "reconsideration", "review", "denied"],
}


def classify(text: str) -> str:
    lowered = text.lower()
    for theme, keywords in THEMES.items():
        if any(keyword in lowered for keyword in keywords):
            return theme
    return "General enquiry"


def _top_terms(texts: Iterable[str], limit: int = 8) -> List[str]:
    tokens: List[str] = []
    for text in texts:
        tokens.extend(re.findall(r"[a-zA-Z]{4,}", text.lower()))
    counter = Counter(tokens)
    return [term for term, _ in counter.most_common(limit)]


def _common_phrases(texts: Iterable[str], limit: int = 5) -> List[str]:
    phrases: List[str] = []
    for text in texts:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        for idx in range(len(words) - 2):
            phrases.append(" ".join(words[idx : idx + 3]))
    counter = Counter(phrases)
    return [phrase for phrase, _ in counter.most_common(limit)]


def _sentiment_proxy(texts: Iterable[str]) -> Dict[str, int]:
    negative_words = {"issue", "error", "unable", "delay", "problem", "denied"}
    positive_words = {"thanks", "resolved", "clear", "helpful", "approved"}
    negative = 0
    positive = 0
    for text in texts:
        tokens = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
        if tokens & negative_words:
            negative += 1
        if tokens & positive_words:
            positive += 1
    return {"negative": negative, "positive": positive}


def build_pattern(texts: Iterable[str]) -> Dict[str, object]:
    text_list = list(texts)
    return {
        "count": len(text_list),
        "top_terms": _top_terms(text_list),
        "common_phrases": _common_phrases(text_list),
        "sentiment_proxy": _sentiment_proxy(text_list),
    }


def extract_patterns(enquiries: Iterable[str]) -> Dict[str, Dict[str, object]]:
    per_theme: Dict[str, List[str]] = {}
    for text in enquiries:
        theme = classify(text)
        per_theme.setdefault(theme, []).append(text)

    patterns: Dict[str, Dict[str, object]] = {}
    for theme, texts in per_theme.items():
        patterns[theme] = build_pattern(texts)
    return patterns
