import json
import random
import re
from typing import Dict, List, Optional

import requests

from .config import settings
from .pii import detect_pii, redact
from .storage import get_sanitized_texts


DISCLAIMER = "Synthetic / Exploratory — Not real user data"
MAX_SNIPPET_LENGTH = 320


def _ollama_generate(prompt: str) -> List[str]:
    response = requests.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("response", "")
    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    return [line for line in lines if len(line) > 5]


def _clean_terms(terms: List[str]) -> List[str]:
    # More targeted blocking - only block obvious PII tokens
    blocked = {
        "pii", "name", "email", "address", "phone", "confidential", "sensitive"
    }
    
    cleaned: List[str] = []
    for term in terms:
        normalized = re.sub(r"[^a-zA-Z\\s-]", "", term).strip()
        if not normalized or len(normalized) < 3:
            continue
            
        # Check if it's a blocked term
        tokens = normalized.lower().split()
        if any(token in blocked or token.startswith("pii") for token in tokens):
            continue
        if normalized.lower() in blocked:
            continue
            
        # Skip obvious redaction tokens
        if normalized.startswith('[') and normalized.endswith(']'):
            continue
            
        cleaned.append(normalized)
    
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for term in cleaned:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)
    
    return deduped


def _persona_defaults(theme: str) -> Dict[str, str]:
    defaults = {
        "Inspections & closeout": {
            "role": "site coordinator",
            "focus": "inspection sequence, letters of assurance, and final sign-off readiness",
        },
        "Permitting intake & status": {
            "role": "project administrator",
            "focus": "intake status, review milestones, and submission checklists",
        },
        "Site constraints & environmental review": {
            "role": "development planner",
            "focus": "environmental assessments, riparian reviews, and arborist dependencies",
        },
        "Design revisions & review comments": {
            "role": "architectural coordinator",
            "focus": "review comments, resubmission priorities, and plan coordination",
        },
        "Fire access & safety": {
            "role": "site planner",
            "focus": "emergency access widths, turning templates, and fire requirements",
        },
        "Expedite & financing pressure": {
            "role": "developer representative",
            "focus": "timeline risk, coordinated comments, and financing-driven milestones",
        },
        "Document upload & technical issues": {
            "role": "permit applicant",
            "focus": "portal uploads, file formats, and submission troubleshooting",
        },
        "Account access & authentication": {
            "role": "account user",
            "focus": "login recovery, access verification, and account support",
        },
        "Payment & fees": {
            "role": "applicant",
            "focus": "fee estimates, invoices, and payment confirmation",
        },
        "Appeals & reconsideration": {
            "role": "applicant",
            "focus": "appeal steps, reconsideration timelines, and required documentation",
        },
    }
    return defaults.get(
        theme,
        {
            "role": "project coordinator",
            "focus": "next steps, required documentation, and predictable timelines",
        },
    )


def _template_generate(kind: str, theme: str, pattern: Dict[str, object], n: int) -> List[str]:
    top_terms = _clean_terms(pattern.get("top_terms", []))
    phrases = _clean_terms(pattern.get("common_phrases", []))
    
    # Use actual extracted terms for better context
    focus_terms = top_terms[:3] if top_terms else []
    context_phrases = phrases[:2] if phrases else []
    
    if kind == "persona":
        defaults = _persona_defaults(theme)
        
        # Build more specific focus based on extracted terms
        if focus_terms:
            focus = f"{', '.join(focus_terms)} coordination and requirements"
        else:
            focus = defaults["focus"]
            
        if context_phrases:
            context = f"experience with {', '.join(context_phrases)}"
        else:
            context = "multi-agency coordination"
        
        base = [
            f"{theme} persona: A {defaults['role']} with {context}, seeking clarity on {focus} to streamline approvals.",
            f"{theme} persona: A {defaults['role']} coordinating {focus}, focused on avoiding delays and rework.",
            f"{theme} persona: A {defaults['role']} managing {context}, needing guidance on sequencing and dependencies.",
            f"{theme} persona: A {defaults['role']} balancing {focus} with timeline constraints and consultant coordination.",
            f"{theme} persona: A {defaults['role']} seeking consolidated feedback on {focus} to maintain project momentum."
        ]
    else:
        # Build more contextual enquiries
        if focus_terms and context_phrases:
            focus_text = f"{', '.join(focus_terms)} related to {', '.join(context_phrases)}"
        elif focus_terms:
            focus_text = f"{', '.join(focus_terms)} requirements and process"
        elif context_phrases:
            focus_text = f"guidance on {', '.join(context_phrases)}"
        else:
            focus_text = "process requirements and next steps"
            
        base = [
            f"{theme}: Requesting clarification on {focus_text} and expected timeline for resolution.",
            f"{theme}: Seeking guidance on {focus_text} to coordinate with consultants and avoid resubmission.",
            f"{theme}: Looking for consolidated feedback on {focus_text} and any blocking items.",
            f"{theme}: Asking about {focus_text} and how to sequence deliverables efficiently.",
            f"{theme}: Needing confirmation on {focus_text} and coordination with other review agencies."
        ]
    
    items = []
    for idx in range(n):
        items.append(base[idx % len(base)])
    return items


def _post_process(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    for item in items:
        findings = detect_pii(item)
        if findings:
            redacted, _ = redact(item)
            cleaned.append(redacted)
        else:
            cleaned.append(item)
    return cleaned


def _generate_with_guardrails(prompt: str, kind: str, theme: str, pattern: Dict[str, object], n: int) -> List[str]:
    try:
        if settings.llm_provider.lower() == "ollama":
            items = _ollama_generate(prompt)
        else:
            items = _template_generate(kind, theme, pattern, n)
    except requests.RequestException:
        items = _template_generate(kind, theme, pattern, n)

    if len(items) < n:
        items.extend(_template_generate(kind, theme, pattern, n - len(items)))

    items = items[:n]
    cleaned = []
    for item in items:
        findings = detect_pii(item)
        if findings:
            retry_items = _template_generate(kind, theme, pattern, 1)
            retry_item = retry_items[0]
            if detect_pii(retry_item):
                retry_item, _ = redact(retry_item)
            cleaned.append(retry_item)
        else:
            cleaned.append(item)
    return _post_process(cleaned)


def _split_into_snippets(text: str) -> List[str]:
    # First split on blank lines to detect paragraphs
    paragraphs = re.split(r"\n{2,}", text)
    candidates: List[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Further split into sentences or bullet-style fragments
        sentences = re.split(r"(?<=[.!?])\s+|[\n•\-]+", paragraph)
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip(" -•")
            if not normalized:
                continue
            if len(normalized) < 25:
                continue
            candidates.append(normalized[:MAX_SNIPPET_LENGTH].strip())
    return candidates


def _fetch_evidence_snippets(theme: str, limit: int = 30) -> List[str]:
    sanitized_texts = get_sanitized_texts(theme)
    if not sanitized_texts:
        return []
    snippets: List[str] = []
    seen: set[str] = set()
    for text in sanitized_texts:
        for snippet in _split_into_snippets(text):
            # Ensure snippets remain privacy-safe even if sanitized text missed something
            if detect_pii(snippet):
                snippet, _ = redact(snippet)
            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            snippets.append(snippet)
            if len(snippets) >= limit:
                return snippets
    return snippets


def _infer_tone(snippet: Optional[str]) -> str:
    if not snippet:
        return "measured but cautious"
    lowered = snippet.lower()
    tone_map = {
        "urgent": ["urgent", "asap", "immediate", "escalate", "rush", "financing"],
        "frustrated": ["frustrated", "confused", "again", "still waiting", "pushed back"],
        "concerned": ["concerned", "worried", "risk", "delay", "deadline"],
        "pressured": ["lender", "carry cost", "financing", "expedite", "penalty"],
        "clarifying": ["confirm", "clarify", "guidance", "direction"],
    }
    for tone, keywords in tone_map.items():
        if any(keyword in lowered for keyword in keywords):
            if tone == "urgent":
                return "urgent and escalating"
            if tone == "frustrated":
                return "frustrated but collaborative"
            if tone == "concerned":
                return "concerned about downstream risk"
            if tone == "pressured":
                return "under financing pressure"
            if tone == "clarifying":
                return "seeking specific clarification"
    return "measured but cautious"


def _infer_pressure(snippet: Optional[str], theme: str) -> str:
    focus = theme.lower()
    generic = f"keep the {focus} workstream on schedule"
    if not snippet:
        return generic
    lowered = snippet.lower()
    if any(keyword in lowered for keyword in ["deadline", "closing", "funding", "draw"]):
        return "protect financing milestones and avoid lender penalties"
    if any(keyword in lowered for keyword in ["inspection", "final", "occupancy"]):
        return "sequence inspections so occupancy isn't blocked"
    if any(keyword in lowered for keyword in ["intake", "submitted", "portal", "status"]):
        return "get a definitive read on intake status"
    if any(keyword in lowered for keyword in ["comments", "revision", "resubmit"]):
        return "close lingering review comments before resubmittal"
    if any(keyword in lowered for keyword in ["fire", "access", "swept"]):
        return "confirm fire access requirements before site work"
    return generic


def _focus_summary(pattern: Dict[str, object], defaults: Dict[str, str]) -> str:
    top_terms = _clean_terms(pattern.get("top_terms", []))
    phrases = _clean_terms(pattern.get("common_phrases", []))
    if top_terms and phrases:
        return f"{', '.join(top_terms[:2])} alignment around {phrases[0]}"
    if top_terms:
        return f"{', '.join(top_terms[:3])} coordination"
    if phrases:
        return f"{phrases[0]} guidance"
    return defaults["focus"]


def _build_enquiry_copy(
    theme: str,
    pattern: Dict[str, object],
    snippet: Optional[str],
    idx: int,
) -> str:
    defaults = _persona_defaults(theme)
    tone = _infer_tone(snippet)
    pressure = _infer_pressure(snippet, theme)
    focus = _focus_summary(pattern, defaults)
    snippet_clause = snippet or f"recent notes referencing {focus}"
    closing_actions = [
        "keep consultants aligned",
        "sequence submissions without rework",
        "unlock the next municipal checkpoint",
        "answer the lender's questions confidently",
        "plan the next coordination call with agencies",
    ]
    closing = closing_actions[idx % len(closing_actions)]
    return (
        f"A {defaults['role']} working through {theme.lower()} sounds {tone}. "
        f"They spell out that \"{snippet_clause}\" and they are trying to {pressure}. "
        f"They need concrete guidance on {focus} so they can {closing}."
    )


def _sample_evidence(snippets: List[str], anchor: Optional[str], per_item: int = 3) -> List[str]:
    if not snippets:
        return []
    pool = snippets.copy()
    random.shuffle(pool)
    evidence: List[str] = []
    if anchor and anchor in snippets:
        evidence.append(anchor)
    for snippet in pool:
        if anchor and snippet == anchor:
            continue
        evidence.append(snippet)
        if len(evidence) >= per_item:
            break
    return evidence[:per_item]


def generate_pseudo_enquiries(theme: str, pattern: Dict[str, object], n: int) -> List[Dict[str, object]]:
    snippets = _fetch_evidence_snippets(theme, limit=max(10, n * 3))
    items: List[Dict[str, object]] = []
    if snippets:
        for idx in range(n):
            snippet = snippets[idx % len(snippets)]
            text = _build_enquiry_copy(theme, pattern, snippet, idx)
            evidence = _sample_evidence(snippets, snippet)
            items.append(
                {
                    "text": text,
                    "evidence": evidence,
                    "tone": _infer_tone(snippet),
                    "pressure": _infer_pressure(snippet, theme),
                }
            )
        return items

    # Fall back to templated generation if no snippets are available yet
    prompt = (
        f"Generate {n} synthetic user enquiries based on this theme and pattern.\n"
        f"Theme: {theme}\n"
        f"Pattern JSON: {json.dumps(pattern)}\n"
        "Rules: No names, no IDs, no dates, no locations, synthetic only.\n"
        "Return as bullet points."
    )
    fallback_items = _generate_with_guardrails(prompt, "enquiry", theme, pattern, n)
    for entry in fallback_items:
        items.append(
            {
                "text": entry,
                "evidence": [],
                "tone": "measured but cautious",
                "pressure": f"keep the {theme.lower()} workstream on schedule",
                "evidence_note": "No sanitized snippets available yet for this theme.",
            }
        )
    return items


def generate_personas(theme: str, pattern: Dict[str, object], n: int) -> List[str]:
    prompt = (
        f"Generate {n} synthetic personas based on this theme and pattern.\n"
        f"Theme: {theme}\n"
        f"Pattern JSON: {json.dumps(pattern)}\n"
        "Rules: No names, no IDs, no dates, no locations, synthetic only.\n"
        "Return as bullet points."
    )
    return _generate_with_guardrails(prompt, "persona", theme, pattern, n)


def generate_scenarios(theme: str, pattern: Dict[str, object], n: int) -> List[str]:
    prompt = (
        f"Generate {n} synthetic scenarios based on this theme and pattern.\n"
        f"Theme: {theme}\n"
        f"Pattern JSON: {json.dumps(pattern)}\n"
        "Rules: No names, no IDs, no dates, no locations, synthetic only.\n"
        "Return as bullet points."
    )
    return _generate_with_guardrails(prompt, "scenario", theme, pattern, n)


def wrap_output(kind: str, items: List[object]) -> Dict[str, object]:
    return {
        "kind": kind,
        "items": items,
        "disclaimer": DISCLAIMER,
    }
