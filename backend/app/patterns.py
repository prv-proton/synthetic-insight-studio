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
    
    # Score each theme based on keyword matches
    theme_scores = {}
    for theme, keywords in THEMES.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > 0:
            theme_scores[theme] = score
    
    # Return the theme with the highest score, or General enquiry if no matches
    if theme_scores:
        return max(theme_scores, key=theme_scores.get)
    return "General enquiry"


def _top_terms(texts: Iterable[str], limit: int = 8) -> List[str]:
    tokens: List[str] = []
    # Domain-specific terms to prioritize
    domain_terms = {
        'permit', 'inspection', 'review', 'submission', 'approval', 'compliance',
        'zoning', 'setback', 'variance', 'covenant', 'easement', 'stormwater',
        'arborist', 'riparian', 'geotech', 'foundation', 'framing', 'footing',
        'blower', 'door', 'assurance', 'issuance', 'intake', 'processing',
        'timeline', 'milestone', 'coordination', 'circulation', 'resubmit',
        'blocking', 'revision', 'comment', 'fire', 'access', 'emergency',
        'swept', 'path', 'turning', 'template', 'expedite', 'financing',
        'lender', 'deadline', 'carry', 'costs', 'upload', 'document',
        'portal', 'technical', 'login', 'password', 'authenticate', 'account',
        'verify', 'payment', 'refund', 'charge', 'invoice', 'appeal',
        'reconsideration', 'denied', 'municipal', 'engineering', 'planning',
        'environmental', 'provincial', 'federal', 'agency', 'consultant',
        'architect', 'engineer', 'surveyor', 'contractor', 'developer'
    }
    
    for text in texts:
        # Extract meaningful terms, prioritizing domain-specific ones
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        for word in words:
            if len(word) >= 4 or word in domain_terms:
                tokens.append(word)
    
    counter = Counter(tokens)
    # Prioritize domain terms in results
    domain_results = [(term, count) for term, count in counter.most_common() if term in domain_terms]
    other_results = [(term, count) for term, count in counter.most_common() if term not in domain_terms]
    
    combined = domain_results + other_results
    return [term for term, _ in combined[:limit]]


def _common_phrases(texts: Iterable[str], limit: int = 5) -> List[str]:
    phrases: List[str] = []
    # Common domain phrases to look for
    domain_phrases = {
        'building permit', 'development permit', 'site plan', 'fire access',
        'stormwater management', 'tree protection', 'riparian assessment',
        'letters of assurance', 'inspection sequence', 'review comments',
        'coordination call', 'swept path', 'turning template', 'setback requirements',
        'zoning compliance', 'environmental review', 'municipal engineering',
        'planning department', 'permit intake', 'status update', 'timeline confirmation',
        'document upload', 'technical issue', 'account access', 'payment processing',
        'fee calculation', 'appeal process', 'variance application', 'covenant registration'
    }
    
    for text in texts:
        text_lower = text.lower()
        # First check for known domain phrases
        for phrase in domain_phrases:
            if phrase in text_lower:
                phrases.append(phrase)
        
        # Then extract other 2-3 word phrases
        words = re.findall(r"[a-zA-Z]{3,}", text_lower)
        for idx in range(len(words) - 1):
            two_word = " ".join(words[idx : idx + 2])
            if len(two_word) > 8:  # Avoid very short phrases
                phrases.append(two_word)
        for idx in range(len(words) - 2):
            three_word = " ".join(words[idx : idx + 3])
            if len(three_word) > 12:  # Avoid very short phrases
                phrases.append(three_word)
    
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
        # Work with original text for better pattern extraction
        theme = classify(text)
        per_theme.setdefault(theme, []).append(text)

    patterns: Dict[str, Dict[str, object]] = {}
    for theme, texts in per_theme.items():
        patterns[theme] = build_pattern(texts)
    return patterns
