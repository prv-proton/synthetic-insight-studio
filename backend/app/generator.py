import json
import re
from typing import Any, Dict, List

import requests

from .config import settings
from .email_thread import redact_uniform
from .llm_client import generate_json
from .pii import detect_pii, redact
from .prompts import (
    build_json_repair_prompt,
    build_pseudo_email_prompt,
    build_quality_improve_prompt,
)
from .quality import evaluate_pseudo_email
from .schemas import PseudoEmailModel


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
    context = _context_from_pattern(theme, pattern)
    emails = generate_pseudo_thread_first_email(context, style="permit_housing", n=n)
    rendered: List[str] = []
    for email in emails:
        subject = email.get("subject", "Subject: Permit enquiry")
        body = email.get("body", "")
        rendered.append(f"Subject: {subject}\n\n{body}")
    return rendered


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


PLACEHOLDER_TOKENS = [
    "[NAME]",
    "[EMAIL]",
    "[PHONE]",
    "[ADDRESS]",
    "[PARCEL_ID]",
    "[FILE_NO]",
    "[ATTACHMENT]",
    "[DATE]",
]


def generate_pseudo_thread_first_email(
    context: Dict[str, Any],
    style: str = "permit_housing",
    n: int = 1,
) -> List[Dict[str, Any]]:
    prompt = _build_pseudo_email_prompt(context, style=style, n=n)
    try:
        if settings.llm_provider.lower() == "ollama":
            response = _ollama_generate_json(prompt)
            emails = _coerce_email_list(response, n)
        else:
            emails = _template_pseudo_emails(context, n)
    except requests.RequestException:
        emails = _template_pseudo_emails(context, n)

    processed: List[Dict[str, Any]] = []
    for email in emails:
        cleaned = _sanitize_pseudo_email(email)
        processed.append(cleaned)
    return processed


def generate_high_fidelity_pseudo_email(
    context_json: Dict[str, Any],
    style: str = "permit_housing",
    enhanced: bool = True,
) -> Dict[str, Any]:
    prompt = build_pseudo_email_prompt(context_json, style=style)
    draft, used_repair = _generate_pseudo_email_json(prompt)
    validated = _validate_pseudo_email(draft)
    if validated is None:
        validated = _template_pseudo_emails(context_json, 1)[0]
    issues = evaluate_pseudo_email(validated)
    used_improve = False
    if enhanced and issues:
        improved_prompt = build_quality_improve_prompt(validated, context_json)
        improved, used_repair_2 = _generate_pseudo_email_json(improved_prompt)
        improved_validated = _validate_pseudo_email(improved)
        if improved_validated:
            validated = improved_validated
            used_repair = used_repair or used_repair_2
            used_improve = True
            issues = evaluate_pseudo_email(validated)
    cleaned = _sanitize_pseudo_email(validated)
    cleaned["quality_signals"] = {
        "issues": issues,
        "used_repair": used_repair,
        "used_improve": used_improve,
    }
    return cleaned


def _ollama_generate_json(prompt: str) -> Dict[str, Any]:
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
    raw = payload.get("response", "").strip()
    return _parse_json_payload(raw) or {}


def _parse_json_payload(raw: str) -> Dict[str, Any] | None:
    if not raw:
        return None
    
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _generate_pseudo_email_json(prompt: str) -> tuple[Dict[str, Any], bool]:
    used_repair = False
    payload = generate_json(prompt, temperature=0.7, top_p=0.9, num_predict=1000)
    raw = payload.get("response", "").strip()
    parsed = _parse_json_payload(raw) or {}
    if parsed:
        return parsed, used_repair
    repair_prompt = build_json_repair_prompt(raw, _pseudo_email_schema_hint())
    payload = generate_json(repair_prompt, temperature=0.4, top_p=0.9, num_predict=800)
    raw = payload.get("response", "").strip()
    parsed = _parse_json_payload(raw) or {}
    used_repair = True
    return parsed, used_repair


def _validate_pseudo_email(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if not payload:
        return None
    try:
        validated = PseudoEmailModel.model_validate(payload)
        return validated.model_dump()
    except ValueError:
        return None


def _coerce_email_list(data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    emails = []
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
    elif isinstance(data, dict):
        items = [data]
    else:
        items = []
    for item in items[:n]:
        emails.append(_coerce_email_item(item))
    if len(emails) < n:
        emails.extend([_coerce_email_item({}) for _ in range(n - len(emails))])
    return emails


def _coerce_email_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject": str(item.get("subject", "Permit enquiry update")).strip(),
        "from_role": str(item.get("from_role", "unknown")).strip(),
        "tone": str(item.get("tone", "neutral")).strip(),
        "body": str(item.get("body", "")).strip(),
        "attachments_mentioned": item.get("attachments_mentioned", []) or [],
        "placeholders_used": item.get("placeholders_used", []) or [],
    }


def _sanitize_pseudo_email(email: Dict[str, Any]) -> Dict[str, Any]:
    subject = str(email.get("subject", "")).strip()
    body = str(email.get("body", "")).strip()
    redacted_subject, _ = redact_uniform(subject)
    redacted_body, _ = redact_uniform(body)
    placeholders_used = _extract_placeholders(redacted_subject + " " + redacted_body)
    attachments = email.get("attachments_mentioned", []) or []
    if "[ATTACHMENT]" in redacted_body and "[ATTACHMENT]" not in attachments:
        attachments = attachments + ["[ATTACHMENT]"]
    from_role = email.get("from_role", "unknown")
    tone = email.get("tone", "neutral")
    if detect_pii(redacted_body):
        redacted_body, _ = redact(redacted_body)
    return {
        "subject": redacted_subject or "Permit enquiry update",
        "from_role": from_role,
        "tone": tone,
        "body": redacted_body,
        "attachments_mentioned": attachments,
        "placeholders_used": placeholders_used,
    }


def _extract_placeholders(text: str) -> List[str]:
    return [token for token in PLACEHOLDER_TOKENS if token in text]


def _build_pseudo_email_prompt(context: Dict[str, Any], style: str, n: int) -> str:
    context_json = json.dumps(context, ensure_ascii=False)
    return (
        f"Generate {n} synthetic first-email messages for a permitting thread.\n"
        f"Style: {style}\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Never include real names/emails/phones/addresses.\n"
        "- Use placeholders: [NAME], [EMAIL], [PHONE], [ADDRESS], [PARCEL_ID], "
        "[FILE_NO], [ATTACHMENT], [DATE].\n"
        "- Include motivations, constraints, and decision points in narrative form.\n"
        "Schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "subject": str,\n'
        '      "from_role": "homeowner|developer|consultant|unknown",\n'
        '      "tone": "urgent|anxious|frustrated|neutral",\n'
        '      "body": str,\n'
        '      "attachments_mentioned": [str],\n'
        '      "placeholders_used": [str]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Context JSON:\n{context_json}\n"
        "Return JSON only."
    )


def _template_pseudo_emails(context: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    tco = context.get("tco", {})
    persona = context.get("persona", {})
    goals = tco.get("goals", []) or ["Permit enquiry update"]
    subject = goals[0] if goals else "Permit enquiry update"
    from_role = tco.get("actor_role") or persona.get("from_role", "unknown")
    tone = tco.get("tone") or persona.get("tone", "neutral")
    constraints = tco.get("constraints", []) or ["Timeline pressure."]
    blockers = tco.get("blockers", []) or []
    decision_points = tco.get("decision_points", []) or []
    tried = tco.get("what_they_tried", []) or ["Submitted the latest materials."]
    asking = tco.get("what_they_are_asking", []) or ["Confirm next steps and remaining items."]
    attachments = tco.get("attachments_mentioned", []) or ["[ATTACHMENT]"]

    base_body = (
        "Hello,\n\n"
        f"I'm reaching out regarding a permitting request tied to [ADDRESS] and file [FILE_NO]. "
        f"Our team is trying to move forward but we're facing constraints like {', '.join(constraints[:2])}. "
        f"Primary goals include {', '.join(goals[:2])}.\n\n"
        f"What we've tried so far: {', '.join(tried[:2])}. "
        f"Pending items include {', '.join(blockers[:2]) or 'review feedback and approval timing'}. "
        f"Key decision points are {', '.join(decision_points[:2]) or 'confirming requirements and sequencing'}.\n\n"
        f"What we're asking: {', '.join(asking[:2])}. "
        "We can share updated materials if needed.\n\n"
        f"Attachments mentioned: {', '.join(attachments[:2])}\n\n"
        f"Thanks,\nA {from_role} applicant"
    )

    emails = []
    for _ in range(n):
        emails.append(
            {
                "subject": subject,
                "from_role": from_role,
                "tone": tone,
                "body": base_body,
                "attachments_mentioned": attachments,
                "placeholders_used": _extract_placeholders(base_body + subject),
            }
        )
    return emails


def _context_from_pattern(theme: str, pattern: Dict[str, Any]) -> Dict[str, Any]:
    top_terms = _clean_terms(pattern.get("top_terms", []))
    phrases = _clean_terms(pattern.get("common_phrases", []))
    tco = {
        "stage": "in_review",
        "actor_role": "unknown",
        "tone": "neutral",
        "goals": top_terms[:3] or ["Clarify review requirements."],
        "constraints": phrases[:2] or ["Timeline pressure."],
        "blockers": phrases[2:4] or [],
        "decision_points": ["Confirm requirements", "Agree on next steps"],
        "what_they_tried": ["Submitted initial materials."],
        "what_they_are_asking": ["Confirm remaining items and timelines."],
        "attachments_mentioned": ["[ATTACHMENT]"],
        "agencies_or_roles": ["Planning"],
        "timeline_signals": ["[DATE]"],
    }
    return {
        "tco": tco,
        "persona": {
            "persona_name": "Focused applicant",
            "from_role": "unknown",
            "experience_level": "medium",
            "primary_motivation": "Move the permit forward",
            "frustrations": [],
            "needs": tco["goals"],
            "tone": "neutral",
        },
    }


def _pseudo_email_schema_hint() -> str:
    return (
        "{\n"
        '  "subject": str,\n'
        '  "from_role": "homeowner|developer|consultant|unknown",\n'
        '  "tone": "urgent|anxious|frustrated|neutral|unknown",\n'
        '  "body": str,\n'
        '  "attachments_mentioned": [str],\n'
        '  "motivations": [str],\n'
        '  "decision_points": [str],\n'
        '  "assumptions": [str]\n'
        "}"
    )
