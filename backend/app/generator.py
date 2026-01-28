import json
import random
import re
from typing import Dict, List, Optional

import requests

from .config import settings
from .email_thread import redact_uniform
from .llm_client import generate_json
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
