from __future__ import annotations

import json
from typing import Dict


def build_context_prompt(thread_redacted: str, baseline_context: Dict[str, object]) -> str:
    example_thread = (
        "Subject: Permit update for [ADDRESS] [FILE_NO]\n"
        "Hi team,\n\n"
        "We submitted [ATTACHMENT] last week and are waiting on planning feedback. "
        "Our financing milestone is [DATE], so we need clarity on remaining items.\n\n"
        "Can you confirm if engineering has signed off and whether a variance is still required?\n"
    )
    example_context = {
        "stage": "in_review",
        "actor_role": "developer",
        "tone": "urgent",
        "goals": ["Confirm remaining items", "Maintain review timeline"],
        "constraints": ["Financing milestone at [DATE]", "Carry costs"],
        "blockers": ["Planning feedback pending", "Engineering sign-off"],
        "decision_points": ["Variance requirement"],
        "what_they_tried": ["Submitted [ATTACHMENT]"],
        "what_they_are_asking": ["Confirm remaining items", "Status of sign-offs"],
        "attachments_mentioned": ["[ATTACHMENT]"],
        "agencies_or_roles": ["Planning", "Engineering"],
        "timeline_signals": ["[DATE]"],
    }
    baseline_json = json.dumps(baseline_context, ensure_ascii=False)
    example_context_json = json.dumps(example_context, ensure_ascii=False)
    return (
        "You are a UX research synthesis assistant.\n"
        "Return STRICT JSON only. Do not add markdown or commentary.\n"
        "Never output real names/emails/phones/addresses; use placeholders like [ADDRESS], [FILE_NO].\n"
        "Schema:\n"
        "{\n"
        '  "stage": "early_inquiry|in_review|conflict_resolution|closeout|expedite|unknown",\n'
        '  "actor_role": "homeowner|developer|consultant|unknown",\n'
        '  "tone": "urgent|anxious|frustrated|neutral|unknown",\n'
        '  "goals": [str],\n'
        '  "constraints": [str],\n'
        '  "blockers": [str],\n'
        '  "decision_points": [str],\n'
        '  "what_they_tried": [str],\n'
        '  "what_they_are_asking": [str],\n'
        '  "attachments_mentioned": [str],\n'
        '  "agencies_or_roles": [str],\n'
        '  "timeline_signals": [str]\n'
        "}\n\n"
        "Example thread (redacted):\n"
        f"{example_thread}\n"
        "Example context JSON:\n"
        f"{example_context_json}\n\n"
        "Now analyze this redacted thread and refine the baseline context.\n"
        f"Baseline context JSON:\n{baseline_json}\n\n"
        "Thread (redacted):\n"
        f"{thread_redacted}\n\n"
        "Return JSON only."
    )


def build_pseudo_email_prompt(context_json: Dict[str, object], style: str = "permit_housing") -> str:
    example_context = {
        "stage": "in_review",
        "actor_role": "consultant",
        "tone": "anxious",
        "goals": ["Confirm review status", "Avoid resubmission"],
        "constraints": ["Consultant schedule", "[DATE] deadline"],
        "blockers": ["Awaiting checklist"],
        "decision_points": ["Whether revised plan is required"],
        "what_they_tried": ["Uploaded [ATTACHMENT]"],
        "what_they_are_asking": ["Confirm remaining items", "Clarify next steps"],
        "attachments_mentioned": ["[ATTACHMENT]"],
        "agencies_or_roles": ["Planning"],
        "timeline_signals": ["[DATE]"],
    }
    example_email = {
        "subject": "Status check for [ADDRESS] ([FILE_NO])",
        "from_role": "consultant",
        "tone": "anxious",
        "body": (
            "Hello Permitting Navigator,\n\n"
            "We are following up on [FILE_NO] for [ADDRESS]. We uploaded [ATTACHMENT] last week "
            "and are trying to keep our [DATE] milestone. We need clarity on the remaining items "
            "and whether a revised plan is required.\n\n"
            "So far we've coordinated with our consultants and provided the updated drawings. "
            "Could you confirm the current status and any outstanding agency sign-offs?\n\n"
            "Thanks,\nA consultant team"
        ),
        "attachments_mentioned": ["[ATTACHMENT]"],
        "motivations": ["Maintain schedule", "Avoid resubmission"],
        "decision_points": ["Need for revised plan"],
        "assumptions": ["Checklist is pending"],
    }
    context_payload = json.dumps(context_json, ensure_ascii=False)
    example_context_json = json.dumps(example_context, ensure_ascii=False)
    example_email_json = json.dumps(example_email, ensure_ascii=False)
    return (
        "You are drafting a realistic first email enquiry to a permitting navigator.\n"
        "Output STRICT JSON only. No markdown.\n"
        "Use placeholders only: [ADDRESS], [FILE_NO], [PARCEL_ID], [ATTACHMENT], [DATE], [EMAIL], [PHONE].\n"
        "Style: "
        f"{style}.\n"
        "Schema:\n"
        "{\n"
        '  "subject": str,\n'
        '  "from_role": "homeowner|developer|consultant|unknown",\n'
        '  "tone": "urgent|anxious|frustrated|neutral|unknown",\n'
        '  "body": str,\n'
        '  "attachments_mentioned": [str],\n'
        '  "motivations": [str],\n'
        '  "decision_points": [str],\n'
        '  "assumptions": [str]\n'
        "}\n\n"
        "Example context JSON:\n"
        f"{example_context_json}\n"
        "Example email JSON:\n"
        f"{example_email_json}\n\n"
        "Now write the email based on this context JSON:\n"
        f"{context_payload}\n\n"
        "Return JSON only."
    )


def build_json_repair_prompt(bad_output: str, schema_hint: str) -> str:
    return (
        "You are repairing invalid JSON output.\n"
        "Return STRICT JSON only, no markdown.\n"
        "Never include personal identifiers; use placeholders like [ADDRESS], [FILE_NO].\n"
        f"Schema hint:\n{schema_hint}\n\n"
        f"Bad output:\n{bad_output}\n\n"
        "Return corrected JSON only."
    )


def build_quality_improve_prompt(draft_json: Dict[str, object], context_json: Dict[str, object]) -> str:
    draft = json.dumps(draft_json, ensure_ascii=False)
    context = json.dumps(context_json, ensure_ascii=False)
    return (
        "Improve the draft JSON for clarity, specificity, and realism while keeping placeholders only.\n"
        "Return STRICT JSON only.\n"
        "Do not add any personal identifiers.\n"
        f"Context JSON:\n{context}\n\n"
        f"Draft JSON:\n{draft}\n\n"
        "Improve missing motivations, decision points, and narrative flow without adding PII."
    )
