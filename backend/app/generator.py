import json
import re
from typing import Any, Dict, List

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
    
    if kind != "enquiry":
        raise ValueError("Template generation only supports enquiry text.")

    if focus_terms and context_phrases:
        focus_text = f"{', '.join(focus_terms)} tied to {', '.join(context_phrases)}"
    elif focus_terms:
        focus_text = f"{', '.join(focus_terms)} requirements and process"
    elif context_phrases:
        focus_text = f"guidance on {', '.join(context_phrases)}"
    else:
        focus_text = "process requirements and next steps"

    base = [
        (
            f"Our project team is seeking clarity on {focus_text}. "
            "We need to align consultant tasks with agency review sequencing and avoid rework. "
            "We have already submitted current materials and want to confirm what remains outstanding. "
            "Please advise on expected timing and any dependencies that could delay approval."
        ),
        (
            f"We are requesting guidance on {focus_text}. "
            "The team is trying to coordinate with multiple reviewers while keeping schedules on track. "
            "We want to understand which items are blocking progress and what the next actionable steps are. "
            "Any detail on required updates or documentation would help us plan effectively."
        ),
        (
            f"Our enquiry focuses on {focus_text}. "
            "We are preparing a resubmission and need to avoid conflicting feedback from agencies. "
            "Please clarify the current status, the remaining checklist items, and likely turnaround time. "
            "We are ready to provide any additional supporting materials as needed."
        ),
        (
            f"We are looking for direction on {focus_text}. "
            "There is pressure to keep the project timeline intact while responding to review comments. "
            "We would like confirmation on the required sequence of deliverables and any approvals still pending. "
            "Guidance on priorities would help us reduce delays."
        ),
        (
            f"We need clarification on {focus_text}. "
            "Our team is coordinating several specialists and wants to ensure submissions meet expectations. "
            "Please confirm any missing items, key decision points, and anticipated review milestones. "
            "This will help us keep stakeholders aligned."
        ),
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
        "Each enquiry should be 2-6 sentences, plain paragraphs only.\n"
        "Do not format as an email and do not include subject lines or signatures.\n"
        "Return as bullet points."
    )
    return _generate_with_guardrails(prompt, "enquiry", theme, pattern, n)


def generate_personas(theme: str, pattern: Dict[str, object], n: int) -> List[Dict[str, object]]:
    defaults = _persona_defaults(theme)
    top_terms = _clean_terms(pattern.get("top_terms", []))
    phrases = _clean_terms(pattern.get("common_phrases", []))
    focus_terms = top_terms[:3] or ["review updates", "requirements", "next steps"]
    frustrations = phrases[:3] or ["unclear sequencing", "slow responses", "conflicting feedback"]
    tones = ["neutral", "anxious", "focused", "urgent", "collaborative"]
    levels = ["low", "medium", "high"]
    personas: List[Dict[str, object]] = []
    for idx in range(n):
        tone = tones[idx % len(tones)]
        experience_level = levels[idx % len(levels)]
        role = defaults["role"]
        primary_motivation = f"Keep {', '.join(focus_terms[:2])} moving without rework."
        persona_name = f"{role.title()} ({theme})"
        persona = {
            "persona_name": persona_name,
            "from_role": role,
            "experience_level": experience_level,
            "primary_motivation": primary_motivation,
            "frustrations": frustrations[:2],
            "needs": focus_terms[:3],
            "tone": tone,
        }
        personas.append(persona)
    return personas


def generate_scenarios(theme: str, pattern: Dict[str, object], n: int) -> List[Dict[str, object]]:
    top_terms = _clean_terms(pattern.get("top_terms", []))
    phrases = _clean_terms(pattern.get("common_phrases", []))
    goals = top_terms[:3] or ["clarify requirements", "keep schedule", "avoid resubmission"]
    blockers = phrases[:3] or ["missing checklist", "coordination gaps", "review delays"]
    scenarios: List[Dict[str, object]] = []
    for idx in range(n):
        scenario = {
            "title": f"{theme} scenario {idx + 1}",
            "narrative": (
                f"A project team is working through {theme.lower()} and needs clarity on {', '.join(goals[:2])}. "
                f"They are encountering blockers such as {', '.join(blockers[:2])} and want guidance on priorities. "
                "They are aiming to keep stakeholders aligned while minimizing rework."
            ),
            "goals": goals[:3],
            "blockers": blockers[:3],
        }
        scenarios.append(scenario)
    return scenarios
