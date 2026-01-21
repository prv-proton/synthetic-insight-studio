import json
import re
from typing import Dict, List

import requests

from .config import settings
from .pii import detect_pii, redact


DISCLAIMER = "Synthetic / Exploratory — Not real user data"


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


def generate_pseudo_enquiries(theme: str, pattern: Dict[str, object], n: int) -> List[str]:
    prompt = (
        f"Generate {n} synthetic user enquiries based on this theme and pattern.\n"
        f"Theme: {theme}\n"
        f"Pattern JSON: {json.dumps(pattern)}\n"
        "Rules: No names, no IDs, no dates, no locations, synthetic only.\n"
        "Return as bullet points."
    )
    return _generate_with_guardrails(prompt, "enquiry", theme, pattern, n)


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


def wrap_output(kind: str, items: List[str]) -> Dict[str, object]:
    return {
        "kind": kind,
        "items": items,
        "disclaimer": DISCLAIMER,
    }
