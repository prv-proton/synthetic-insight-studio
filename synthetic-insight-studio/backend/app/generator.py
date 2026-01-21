import json
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


def _template_generate(kind: str, theme: str, pattern: Dict[str, object], n: int) -> List[str]:
    top_terms = ", ".join(pattern.get("top_terms", []))
    phrases = ", ".join(pattern.get("common_phrases", []))
    base = [
        f"{theme}: Requesting guidance on {top_terms or 'process steps'} with a focus on {phrases or 'clear next actions'}.",
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
