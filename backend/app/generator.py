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
    blocked = {
        "pii",
        "name",
        "email",
        "address",
        "phone",
        "file",
        "date",
        "id",
        "parcel",
        "confidential",
        "legal",
        "attachment",
        "sensitive",
    }
    cleaned: List[str] = []
    for term in terms:
        normalized = re.sub(r"[^a-zA-Z\\s-]", "", term).strip()
        if not normalized:
            continue
        tokens = normalized.lower().split()
        if any(token in blocked or token.startswith("pii") for token in tokens):
            continue
        if normalized.lower() in blocked:
            continue
        cleaned.append(normalized)
    seen = set()
    deduped = []
    for term in cleaned:
        key = term.lower()
        if key in seen:
            continue
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
    top_terms_summary = ", ".join(top_terms[:4])
    phrases_summary = ", ".join(phrases[:2])
    if kind == "persona":
        defaults = _persona_defaults(theme)
        focus = top_terms_summary or defaults["focus"]
        signals = phrases_summary or "coordinated agency feedback"
        base = [
            (
                f"{theme} persona: A {defaults['role']} coordinating a multi-step submission, "
                f"seeking clarity on {focus} to avoid rework and keep timelines predictable."
            ),
            (
                f"{theme} persona: A {defaults['role']} balancing consultants and reviewers, "
                f"looking for guidance on {focus} and how to sequence deliverables."
            ),
            (
                f"{theme} persona: A {defaults['role']} focused on {focus}, "
                f"asking for consolidated comments and clear blocking items."
            ),
            (
                f"{theme} persona: A {defaults['role']} navigating {signals}, "
                f"needing a concise checklist for next steps and approvals."
            ),
            (
                f"{theme} persona: A {defaults['role']} seeking confirmation on {focus} "
                "and expected response windows to align internal schedules."
            ),
        ]
    else:
        base = [
            (
                f"{theme}: Requesting guidance on {top_terms_summary or 'process steps'} "
                f"with a focus on {phrases_summary or 'clear next actions'}."
            ),
            f"{theme}: Seeking clarification on required steps and expected timeline in a privacy-safe manner.",
            f"{theme}: Asking about next steps after recent submission and expected updates.",
            f"{theme}: Looking for help resolving a generic issue without sharing personal details.",
            f"{theme}: Requesting information on how to proceed with a standard case.",
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
