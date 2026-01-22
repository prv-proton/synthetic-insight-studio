import json
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests

from . import patterns as pattern_lib
from .audit import record
from .config import settings
from .generator import (
    DISCLAIMER,
    generate_personas,
    generate_pseudo_enquiries,
    generate_scenarios,
    generate_pseudo_thread_first_email,
    wrap_output,
)
from .ingestion import infer_source_type, trim_excess_whitespace
from .schemas import (
    GenerateRequest,
    IngestRequest,
    OllamaStatus,
    ResetResponse,
    SettingsResponse,
    ThreadContext,
    ThemeSummary,
)
from .storage import (
    clear_all,
    get_audit_recent,
    get_generated,
    get_pattern,
    get_sanitized_texts,
    get_texts_for_patterns,
    get_theme_count,
    get_theme_counts,
    init_db,
    insert_enquiries,
    insert_generated,
    upsert_theme_counts,
    upsert_patterns,
)
from .email_thread import (
    infer_email_like,
    normalize_email_text,
    redact_uniform,
    split_thread_into_turns,
)


app = FastAPI(title="Synthetic Insight Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.post("/ingest")
async def ingest(
    request: Request,
    files: List[UploadFile] | None = File(default=None),
) -> Dict[str, object]:
    if files:
        return await _ingest_files(files)
    payload = IngestRequest(**await request.json())
    return _ingest_payload(payload)


@app.post("/sanitize")
async def sanitize(
    request: Request,
    files: List[UploadFile] | None = File(default=None),
) -> Dict[str, object]:
    if files:
        mode = request.query_params.get("mode", "mask_only")
        items = await _sanitize_files(files, mode)
        record(
            "sanitize",
            {"count": len(items), "items": [item["audit"] for item in items]},
        )
        return {"items": [item["response"] for item in items]}
    payload = await request.json()
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Text is required")
    source_type = payload.get("source_type", "auto")
    mode = payload.get("mode", "mask_only") if isinstance(payload, dict) else "mask_only"
    response, audit = _sanitize_text(text, source_type, mode, filename=None)
    record("sanitize", {"count": 1, "items": [audit]})
    return response


@app.post("/thread/analyze")
async def analyze_thread(
    request: Request,
    files: List[UploadFile] | None = File(default=None),
) -> Dict[str, object]:
    text, source_type = await _get_thread_payload(request, files)
    tco, persona, next_questions, redacted_latest, meta, stats, turns_count = _analyze_thread_text(
        text,
        source_type,
    )

    record(
        "thread_analyze",
        {
            "source_type": source_type,
            "meta": meta,
            "turns_count": turns_count,
            "redaction_stats": stats,
        },
    )

    return {
        "latest_message_redacted": redacted_latest,
        "tco": tco,
        "persona": persona,
        "next_questions": next_questions,
        "meta": meta,
        "turns_redacted_count": turns_count,
    }


@app.get("/evidence/library")
def evidence_library(topic: str = "permit_housing", limit: int = 5) -> Dict[str, object]:
    snippets = _load_evidence_library(topic, limit)
    return {"topic": topic, "items": snippets}


@app.post("/generate/pseudo_email")
async def generate_pseudo_email(request: Request) -> Dict[str, object]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    source = payload.get("source")
    thread_text = payload.get("thread_text")
    theme = payload.get("theme")
    evidence_source = payload.get("evidence_source", "none")
    n_snippets = int(payload.get("n_snippets", 5))
    style = payload.get("style", "permit_housing")

    analysis = None
    context: Dict[str, object] = {}
    if source == "thread_analyze":
        if not isinstance(thread_text, str) or not thread_text.strip():
            raise HTTPException(status_code=400, detail="thread_text is required")
        tco, persona, next_questions, _, _, _, _ = _analyze_thread_text(thread_text, "auto")
        analysis = {"tco": tco, "persona": persona, "next_questions": next_questions}
        context = {"tco": tco, "persona": persona, "next_questions": next_questions}
    elif source == "tco":
        tco = payload.get("tco")
        if not isinstance(tco, dict):
            raise HTTPException(status_code=400, detail="tco is required")
        validated = ThreadContext.model_validate(tco)
        context = {
            "tco": validated.model_dump(),
            "persona": payload.get("persona") or {},
            "next_questions": payload.get("next_questions") or {},
        }
    elif source == "patterns":
        if not theme:
            raise HTTPException(status_code=400, detail="theme is required")
        pattern_entry = get_pattern(theme)
        if not pattern_entry:
            raise HTTPException(status_code=404, detail="Theme not found")
        pattern = pattern_entry["pattern"]
        context = _context_from_pattern(theme, pattern)
    else:
        raise HTTPException(status_code=400, detail="source must be thread_analyze, tco, or patterns")

    pseudo_emails = generate_pseudo_thread_first_email(context, style=style, n=1)
    pseudo_email = pseudo_emails[0] if pseudo_emails else {}

    topic = payload.get("topic", "permit_housing")
    evidence_snippets, evidence_method = _resolve_evidence_snippets(
        evidence_source,
        thread_text,
        n_snippets,
        topic,
    )
    inspired_by = _build_inspired_by(context)

    record(
        "generate_pseudo_email",
        {
            "source": source,
            "theme": theme,
            "evidence_method": evidence_method,
            "snippets_count": len(evidence_snippets),
        },
    )

    response: Dict[str, object] = {
        "pseudo_email": pseudo_email,
        "evidence_snippets": evidence_snippets,
        "evidence_method": evidence_method,
        "inspired_by": inspired_by,
    }
    if analysis:
        response["analysis"] = analysis
    return response


def _ingest_payload(request: IngestRequest) -> Dict[str, object]:
    sanitized: List[Dict[str, str]] = []
    raw_items: List[str] = []
    theme_counts: List[Dict[str, int]] = []
    for item in request.enquiries:
        redacted, stats = redact_uniform(item.text)
        theme = pattern_lib.classify(redacted)
        sanitized.append({"text_sanitized": redacted, "theme": theme})
        raw_items.append(item.text)
        record("pii_redaction", {"theme": theme, "stats": stats})
        theme_counts.append({"theme": theme, "count_total": 1})
    upsert_theme_counts(theme_counts)
    insert_enquiries(sanitized, raw_items)
    _rebuild_patterns()
    record("ingest", {"count": len(request.enquiries)})
    return {"status": "ok", "ingested": len(request.enquiries)}


async def _ingest_files(files: List[UploadFile]) -> Dict[str, object]:
    sanitized: List[Dict[str, str]] = []
    responses: List[Dict[str, object]] = []
    theme_counts: List[Dict[str, int]] = []

    for file in files:
        filename = file.filename or "uploaded"
        extension = Path(filename).suffix.lower()
        if extension not in {".txt", ".eml"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        try:
            raw_bytes = await file.read()
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} must be valid UTF-8 text",
            ) from exc

        text = trim_excess_whitespace(text)
        if len(text) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} must contain at least 10 characters",
            )

        # Split threaded content by separator
        text_segments = [segment.strip() for segment in text.split('---') if segment.strip()]
        if len(text_segments) > 1:
            # Process each segment as separate enquiry
            for segment in text_segments:
                if len(segment) < 10:
                    continue
                _process_text_segment(segment, filename, extension, sanitized, theme_counts, responses)
        else:
            # Process as single text
            _process_text_segment(text, filename, extension, sanitized, theme_counts, responses)

    upsert_theme_counts(theme_counts)
    if sanitized:
        insert_enquiries(sanitized, raw=None)
    _rebuild_patterns()
    record("ingest", {"count": len(responses)})
    return {"status": "ok", "ingested": len(responses), "items": responses}


def _process_text_segment(
    text: str,
    filename: str,
    extension: str,
    sanitized: List[Dict[str, str]],
    theme_counts: List[Dict[str, int]],
    responses: List[Dict[str, object]],
) -> None:
    source_file_type = extension.lstrip(".")
    inferred_source_type = (
        "email_like" if extension == ".eml" else infer_source_type(text)
    )
    normalization_applied = False
    normalization_meta: Dict[str, object] | None = None
    working_text = text

    if inferred_source_type == "email_like":
        normalized = normalize_email_text(text)
        working_text = str(normalized["cleaned_full"])
        normalization_meta = normalized["meta"]
        normalization_applied = True

    redacted, redaction_stats = redact_uniform(working_text)
    storage_decision = "sanitized_text_stored"

    theme = pattern_lib.classify(redacted)
    sanitized.append({"text_sanitized": redacted, "theme": theme})

    theme_counts.append(
        {
            "theme": theme,
            "count_total": 1,
        }
    )

    record(
        "ingest_txt",
        {
            "filename": filename,
            "source_file_type": source_file_type,
            "inferred_source_type": inferred_source_type,
            "normalization_applied": normalization_applied,
            "normalization_meta": normalization_meta,
            "redaction_counts": redaction_stats,
            "storage_decision": storage_decision,
        },
    )

    responses.append(
        {
            "filename": filename,
            "source_file_type": source_file_type,
            "inferred_source_type": inferred_source_type,
            "normalization_applied": normalization_applied,
            "redaction_counts": redaction_stats,
            "storage_decision": storage_decision,
            "sanitized_text": redacted,
        }
    )


async def _sanitize_files(
    files: List[UploadFile],
    mode: str,
) -> List[Dict[str, Dict[str, object]]]:
    items: List[Dict[str, Dict[str, object]]] = []
    for file in files:
        filename = file.filename or "uploaded"
        extension = Path(filename).suffix.lower()
        if extension not in {".txt", ".eml", ".jsonl"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        raw_bytes = await file.read()
        if len(raw_bytes) > 1_048_576:
            raise HTTPException(
                status_code=413, detail=f"{filename} exceeds 1MB size limit"
            )
        text = raw_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            raise HTTPException(
                status_code=400, detail=f"{filename} must not be empty"
            )
        source_type = "email" if extension == ".eml" else "auto"
        response, audit = _sanitize_text(text, source_type, mode, filename=filename)
        items.append({"response": response, "audit": audit})
    return items


def _sanitize_text(
    text: str,
    source_type: str,
    mode: str,
    filename: str | None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    if source_type not in {"auto", "plain", "email"}:
        raise HTTPException(
            status_code=400,
            detail="source_type must be one of: plain, email, auto",
        )
    inferred_source_type = _infer_sanitize_source_type(text, source_type)
    normalization_meta: Dict[str, object] | None = None
    working_text = text
    turns_count = None
    if inferred_source_type in {"email", "email_like"}:
        normalized = normalize_email_text(text)
        working_text = str(normalized["cleaned_full"])
        normalization_meta = normalized["meta"]
        turns_count = len(split_thread_into_turns(working_text))
    if mode not in {"mask_only", "mask_and_extract_evidence"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: mask_only, mask_and_extract_evidence",
        )
    sanitized_text, redaction_stats = redact_uniform(working_text)
    response: Dict[str, object] = {
        "sanitized_text": sanitized_text,
        "redaction_stats": redaction_stats,
        "inferred_source_type": inferred_source_type,
        "normalization_meta": normalization_meta,
        "notes": [
            "Sanitize-only mode: no classification, no storage, no pattern extraction."
        ],
    }
    if turns_count is not None:
        response["turns_count"] = turns_count
    if mode == "mask_and_extract_evidence":
        snippets = _extract_evidence_snippets(sanitized_text)
        sanitized_snippets: List[str] = []
        for snippet in snippets:
            snippet_redacted, _ = redact_uniform(snippet)
            sanitized_snippets.append(snippet_redacted)
        response["evidence_snippets"] = sanitized_snippets
    if filename:
        response["filename"] = filename
    audit = {
        "filename": filename,
        "inferred_source_type": inferred_source_type,
        "redaction_stats": redaction_stats,
        "normalized": normalization_meta is not None,
        "turns_count": turns_count,
    }
    return response, audit


async def _get_thread_payload(
    request: Request,
    files: List[UploadFile] | None,
) -> tuple[str, str]:
    if files:
        if len(files) != 1:
            raise HTTPException(status_code=400, detail="Upload exactly one thread file")
        file = files[0]
        filename = file.filename or "uploaded"
        extension = Path(filename).suffix.lower()
        if extension not in {".txt", ".eml"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        raw_bytes = await file.read()
        text = raw_bytes.decode("utf-8", errors="replace")
        source_type = "email" if extension == ".eml" else "auto"
        return text, source_type
    payload = await request.json()
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Text is required")
    source_type = payload.get("source_type", "auto")
    return text, source_type


def _analyze_thread_text(
    text: str,
    source_type: str,
) -> tuple[
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
    str,
    Dict[str, object],
    int,
]:
    normalized = _normalize_thread_input(text, source_type)
    cleaned_full = normalized["cleaned_full"]
    latest_message = normalized["latest_message"]
    meta = normalized["meta"]

    redacted_full, full_stats = redact_uniform(cleaned_full)
    redacted_latest, latest_stats = redact_uniform(latest_message)
    turns = split_thread_into_turns(cleaned_full)
    redacted_turns = _redact_turns(turns)

    tco_heuristic = _build_tco_heuristic(
        redacted_latest,
        redacted_full,
        meta,
        redacted_turns,
    )
    tco = _refine_tco_with_ollama(tco_heuristic, redacted_latest, redacted_turns)
    tco = _sanitize_tco_output(tco)
    persona, next_questions = _build_persona_and_questions(tco, redacted_latest)
    stats = {"latest": latest_stats, "full": full_stats}
    return tco, persona, next_questions, redacted_latest[:400], meta, stats, len(redacted_turns)


def _infer_sanitize_source_type(text: str, source_type: str) -> str:
    if source_type == "auto":
        return "email_like" if infer_email_like(text) else "plain_text"
    if source_type == "plain":
        return "plain_text"
    return "email"


def _data_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "data" / filename


def _load_evidence_library(topic: str, limit: int) -> List[Dict[str, object]]:
    path = _data_path("approved_anonymized_snippets.jsonl")
    if not path.exists():
        return []
    items: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_topic = entry.get("topic")
            entry_topics = entry.get("topics") or []
            if entry_topic != topic and topic not in entry_topics:
                continue
            items.append(
                {
                    "text": entry.get("text", ""),
                    "tags": entry.get("tags", []),
                    "stage": entry.get("stage"),
                }
            )
            if len(items) >= limit:
                break
    return items


def _resolve_evidence_snippets(
    evidence_source: str,
    thread_text: str | None,
    n_snippets: int,
    topic: str,
) -> tuple[List[str], str]:
    if evidence_source == "library":
        items = _load_evidence_library(topic, n_snippets)
        return [item["text"] for item in items], "library"
    if evidence_source == "uploaded" and thread_text:
        response, _ = _sanitize_text(thread_text, "auto", "mask_and_extract_evidence", None)
        snippets = response.get("evidence_snippets", [])[:n_snippets]
        return snippets, "uploaded_sanitize" if snippets else "patterns_only"
    return [], "patterns_only"


def _build_inspired_by(context: Dict[str, object]) -> str:
    tco = context.get("tco", {}) if isinstance(context, dict) else {}
    blockers = tco.get("blockers", []) or []
    constraints = tco.get("constraints", []) or []
    signals = ", ".join(blockers[:2] + constraints[:2])
    return f"Inspired by common patterns: {signals or 'standard permitting guidance.'}"


def _context_from_pattern(theme: str, pattern: Dict[str, object]) -> Dict[str, object]:
    top_terms = pattern.get("top_terms", []) or []
    phrases = pattern.get("common_phrases", []) or []
    tco = {
        "stage": "in_review",
        "actor_role": "unknown",
        "tone": "neutral",
        "goals": top_terms[:3] or ["Clarify requirements."],
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
        "next_questions": {
            "to_clarify": ["What steps remain?"],
            "to_unblock": ["Is any document missing?"],
            "risks_if_ignored": ["Timeline may slip."],
        },
    }


def _normalize_thread_input(text: str, source_type: str) -> Dict[str, object]:
    if source_type not in {"auto", "plain", "email"}:
        raise HTTPException(
            status_code=400,
            detail="source_type must be one of: plain, email, auto",
        )
    if source_type == "plain":
        cleaned = trim_excess_whitespace(text)
        meta = {
            "removed_headers": False,
            "removed_signatures": False,
            "removed_disclaimers": False,
            "removed_mime": False,
            "removed_quotes": False,
            "subject": None,
        }
        return {
            "cleaned_full": cleaned,
            "latest_message": cleaned,
            "meta": meta,
        }
    is_email = source_type == "email" or infer_email_like(text)
    if not is_email:
        cleaned = trim_excess_whitespace(text)
        meta = {
            "removed_headers": False,
            "removed_signatures": False,
            "removed_disclaimers": False,
            "removed_mime": False,
            "removed_quotes": False,
            "subject": None,
        }
        return {
            "cleaned_full": cleaned,
            "latest_message": cleaned,
            "meta": meta,
        }
    normalized = normalize_email_text(text)
    return {
        "cleaned_full": normalized["cleaned_full"],
        "latest_message": normalized["latest_message"],
        "meta": normalized["meta"],
    }


def _redact_turns(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    redacted_turns = []
    for turn in turns:
        body = turn.get("body", "")
        redacted_body, _ = redact_uniform(body)
        redacted_turns.append(
            {
                "speaker_role": turn.get("speaker_role", "unknown"),
                "speaker_hint": turn.get("speaker_hint", ""),
                "body": redacted_body,
            }
        )
    return redacted_turns


def _build_tco_heuristic(
    latest_message: str,
    full_text: str,
    meta: Dict[str, object],
    turns: List[Dict[str, str]],
) -> Dict[str, object]:
    base_text = latest_message or full_text

    goals = _extract_sentences(base_text, ["need", "request", "looking for", "seeking", "want"])
    constraints = _extract_sentences(
        base_text,
        ["must", "require", "constraint", "deadline", "cannot", "can't", "[DATE]"],
    )
    blockers = _extract_sentences(
        base_text,
        ["blocked", "waiting", "pending", "missing", "conflicting", "issue", "unable"],
    )
    decision_points = _extract_sentences(
        base_text,
        ["decide", "approval", "determine", "confirm", "sign off"],
    )
    what_they_tried = _extract_sentences(
        base_text,
        ["submitted", "uploaded", "shared", "provided", "tried", "completed"],
    )
    what_they_are_asking = _extract_sentences(
        base_text,
        ["can you", "could you", "please", "confirm", "clarify", "advise", "request"],
    )

    attachments = []
    if "[ATTACHMENT]" in full_text or "attach" in full_text.lower():
        attachments.append("[ATTACHMENT]")

    agencies_or_roles = _extract_agencies(full_text)
    timeline_signals = _extract_timeline_signals(full_text)
    stage = _infer_stage(full_text)
    actor_role = _infer_role(full_text)
    tone = _infer_tone(base_text)

    return {
        "stage": stage,
        "actor_role": actor_role,
        "tone": tone,
        "goals": goals,
        "constraints": constraints,
        "blockers": blockers,
        "decision_points": decision_points,
        "what_they_tried": what_they_tried,
        "what_they_are_asking": what_they_are_asking,
        "attachments_mentioned": attachments,
        "agencies_or_roles": agencies_or_roles,
        "timeline_signals": timeline_signals,
    }


def _build_persona_and_questions(
    tco: Dict[str, object],
    base_text: str,
) -> tuple[Dict[str, object], Dict[str, object]]:
    goals = tco.get("goals", []) if isinstance(tco, dict) else []
    constraints = tco.get("constraints", []) if isinstance(tco, dict) else []
    blockers = tco.get("blockers", []) if isinstance(tco, dict) else []
    agencies = tco.get("agencies_or_roles", []) if isinstance(tco, dict) else []
    attachments = tco.get("attachments_mentioned", []) if isinstance(tco, dict) else []
    timeline_signals = tco.get("timeline_signals", []) if isinstance(tco, dict) else []
    from_role = tco.get("actor_role", "unknown")
    tone = tco.get("tone", "neutral")
    experience_level = _infer_experience(base_text)
    primary_motivation = goals[0] if goals else "Clarity on requirements and next steps."
    frustrations = blockers[:3] if blockers else _extract_sentences(
        base_text, ["frustrated", "confusing", "unclear", "delay"]
    )
    needs = goals[:4] if goals else ["Guidance on requirements and sequencing."]
    persona = {
        "persona_name": _build_persona_name(from_role, tone),
        "from_role": from_role,
        "experience_level": experience_level,
        "primary_motivation": primary_motivation,
        "frustrations": frustrations,
        "needs": needs,
        "tone": tone,
    }
    next_questions = {
        "to_clarify": _build_next_questions(goals, constraints, agencies),
        "to_unblock": _build_unblock_questions(blockers, attachments),
        "risks_if_ignored": _build_risk_questions(blockers, timeline_signals),
    }
    return persona, next_questions


def _refine_tco_with_ollama(
    heuristic: Dict[str, object],
    latest_message: str,
    turns: List[Dict[str, str]],
) -> Dict[str, object]:
    prompt = _build_tco_prompt(heuristic, latest_message, turns)
    try:
        if settings.llm_provider.lower() != "ollama":
            return heuristic
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("response", "").strip()
        data = _parse_json_payload(raw)
        if data is None:
            return heuristic
        validated = ThreadContext.model_validate(data)
        return validated.model_dump()
    except requests.RequestException:
        return heuristic
    except ValueError:
        return heuristic


def _sanitize_tco_output(tco: Dict[str, object]) -> Dict[str, object]:
    sanitized = _redact_object_strings(tco)
    coerced = _coerce_tco_schema(sanitized)
    validated = ThreadContext.model_validate(coerced)
    return validated.model_dump()


def _extract_evidence_snippets(text: str) -> List[str]:
    keywords = [
        "questions",
        "blocking",
        "blocker",
        "conflicting",
        "remaining",
        "expedite",
        "status summary",
        "status",
        "decision",
        "next steps",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: List[str] = []
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            candidates.append(line)
    if len(candidates) < 3:
        for line in lines:
            if line in candidates:
                continue
            if "?" in line or "status" in line.lower():
                candidates.append(line)
            if len(candidates) >= 5:
                break
    if len(candidates) < 3:
        for line in lines:
            if line not in candidates:
                candidates.append(line)
            if len(candidates) >= 3:
                break
    return [candidate[:240].strip() for candidate in candidates[:5]]


def _extract_sentences(text: str, keywords: List[str]) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results: List[str] = []
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            results.append(line)
    if not results:
        for line in lines:
            if "?" in line:
                results.append(line)
            if len(results) >= 3:
                break
    return _dedupe_list(results)[:5]


def _extract_agencies(text: str) -> List[str]:
    agency_terms = [
        "planning",
        "engineering",
        "fire",
        "heritage",
        "environment",
        "utilities",
        "transportation",
        "parks",
        "building",
        "zoning",
        "public works",
        "navigator",
        "agency",
    ]
    found = [term.title() for term in agency_terms if term in text.lower()]
    return _dedupe_list(found)


def _extract_timeline_signals(text: str) -> List[str]:
    tokens = []
    lower = text.lower()
    month_terms = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    for term in month_terms:
        if term in lower:
            tokens.append(term.title())
    for term in ["week", "weeks", "month", "months", "day", "days", "deadline", "asap"]:
        if term in lower:
            tokens.append(term)
    if "[DATE]" in text:
        tokens.append("[DATE]")
    return _dedupe_list(tokens)


def _infer_stage(text: str) -> str:
    lower = text.lower()
    if "expedite" in lower or "rush" in lower:
        return "expedite"
    if "appeal" in lower or "conflict" in lower or "dispute" in lower:
        return "conflict_resolution"
    if "inspection" in lower or "closeout" in lower or "final" in lower:
        return "closeout"
    if "review" in lower or "comment" in lower or "resubmit" in lower:
        return "in_review"
    if "intake" in lower or "submit" in lower or "inquiry" in lower:
        return "early_inquiry"
    return "unknown"


def _infer_role(text: str) -> str:
    lower = text.lower()
    if "developer" in lower or "builder" in lower:
        return "developer"
    if "consultant" in lower or "architect" in lower or "engineer" in lower:
        return "consultant"
    if "homeowner" in lower or "resident" in lower:
        return "homeowner"
    return "unknown"


def _infer_experience(text: str) -> str:
    lower = text.lower()
    if "first time" in lower or "new to" in lower:
        return "low"
    if "experienced" in lower or "repeat" in lower or "multiple" in lower:
        return "high"
    return "medium"


def _infer_tone(text: str) -> str:
    lower = text.lower()
    if "urgent" in lower or "asap" in lower or "immediately" in lower:
        return "urgent"
    if "frustrated" in lower or "confusing" in lower or "unacceptable" in lower:
        return "frustrated"
    if "anxious" in lower or "worried" in lower:
        return "anxious"
    return "neutral"


def _one_sentence_summary(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:180]
    return "Thread summary derived from the latest request."


def _build_persona_name(role: str, tone: str) -> str:
    role_label = {
        "homeowner": "homeowner",
        "developer": "developer",
        "consultant": "consultant",
        "unknown": "applicant",
    }.get(role, "applicant")
    tone_prefix = {
        "urgent": "Time-pressed",
        "frustrated": "Frustrated",
        "anxious": "Anxious",
        "neutral": "Focused",
    }.get(tone, "Focused")
    return f"{tone_prefix} {role_label}"


def _build_next_questions(
    goals: List[str],
    constraints: List[str],
    agencies: List[str],
) -> List[str]:
    questions = []
    if not goals:
        questions.append("What outcome is the requester trying to achieve?")
    if not constraints:
        questions.append("Are there specific constraints or policies affecting the request?")
    if not agencies:
        questions.append("Which agency or reviewer is the primary point of contact?")
    if not questions:
        questions.append("What details should be confirmed before the next step?")
    return questions[:3]


def _build_unblock_questions(blockers: List[str], attachments: List[str]) -> List[str]:
    questions = []
    if blockers:
        questions.append("Which outstanding items are blocking progress right now?")
    if not attachments:
        questions.append("Are there missing attachments or plans that should be shared?")
    if not questions:
        questions.append("What information is needed to move the review forward?")
    return questions[:3]


def _build_risk_questions(blockers: List[str], timeline: List[str]) -> List[str]:
    questions = []
    if blockers:
        questions.append("Delays may continue if blocker details remain unresolved.")
    if timeline:
        questions.append("Timeline commitments could slip without clarified milestones.")
    if not questions:
        questions.append("Unclear requirements could lead to rework or delay.")
    return questions[:3]


def _build_tco_prompt(
    heuristic: Dict[str, object],
    latest_message: str,
    turns: List[Dict[str, str]],
) -> str:
    turns_preview = json.dumps(turns[:6], ensure_ascii=False)
    heuristic_json = json.dumps(heuristic, ensure_ascii=False)
    return (
        "You are refining a redacted email thread context object.\n"
        "Rules: NEVER output real names/emails/phones/addresses. Keep tokens like "
        "[ADDRESS], [FILE_NO], [PARCEL_ID], [ATTACHMENT], [DATE].\n"
        "Return STRICT JSON only, matching this schema:\n"
        "{\n"
        '  "stage": "early_inquiry|in_review|conflict_resolution|closeout|expedite|unknown",\n'
        '  "actor_role": "homeowner|developer|consultant|unknown",\n'
        '  "tone": "anxious|frustrated|neutral|urgent|unknown",\n'
        '  "goals": [str],\n'
        '  "constraints": [str],\n'
        '  "blockers": [str],\n'
        '  "decision_points": [str],\n'
        '  "what_they_tried": [str],\n'
        '  "what_they_are_asking": [str],\n'
        '  "attachments_mentioned": [str],\n'
        '  "agencies_or_roles": [str],\n'
        '  "timeline_signals": [str]\n'
        "}\n"
        f"Latest redacted message:\n{latest_message}\n"
        f"Thread turns (redacted preview):\n{turns_preview}\n"
        f"Baseline heuristic JSON:\n{heuristic_json}\n"
        "Return only JSON."
    )


def _parse_json_payload(raw: str) -> Dict[str, object] | None:
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


def _redact_object_strings(value: object) -> object:
    if isinstance(value, str):
        redacted, _ = redact_uniform(value)
        return redacted
    if isinstance(value, list):
        return [_redact_object_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_object_strings(val) for key, val in value.items()}
    return value


def _coerce_tco_schema(data: Dict[str, object]) -> Dict[str, object]:
    stage = _coerce_enum(data.get("stage"), {
        "early_inquiry",
        "in_review",
        "conflict_resolution",
        "closeout",
        "expedite",
        "unknown",
    })
    actor_role = _coerce_enum(
        data.get("actor_role"),
        {"homeowner", "developer", "consultant", "unknown"},
    )
    tone = _coerce_enum(
        data.get("tone"),
        {"anxious", "frustrated", "neutral", "urgent", "unknown"},
    )
    return {
        "stage": stage,
        "actor_role": actor_role,
        "tone": tone,
        "goals": _coerce_list(data.get("goals")),
        "constraints": _coerce_list(data.get("constraints")),
        "blockers": _coerce_list(data.get("blockers")),
        "decision_points": _coerce_list(data.get("decision_points")),
        "what_they_tried": _coerce_list(data.get("what_they_tried")),
        "what_they_are_asking": _coerce_list(data.get("what_they_are_asking")),
        "attachments_mentioned": _coerce_list(data.get("attachments_mentioned")),
        "agencies_or_roles": _coerce_list(data.get("agencies_or_roles")),
        "timeline_signals": _coerce_list(data.get("timeline_signals")),
    }


def _coerce_enum(value: object, allowed: set[str]) -> str:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in allowed:
            return lowered
    return "unknown" if "unknown" in allowed else next(iter(allowed))


def _coerce_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_text(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Unknown"


def _dedupe_list(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


@app.post("/patterns/rebuild")
def rebuild_patterns() -> Dict[str, object]:
    themes = _rebuild_patterns()
    return {"status": "ok", "themes": themes}


def _rebuild_patterns() -> List[str]:
    theme_counts = get_theme_counts()
    themes: List[str] = []
    audit_payload: List[Dict[str, object]] = []
    for entry in theme_counts:
        theme = entry["theme"]
        # Use raw texts for pattern extraction when available, sanitized otherwise
        texts = get_texts_for_patterns(theme)
        text_available_count = len(texts)
        pattern = pattern_lib.build_pattern(texts)
        pattern.update(
            {
                "count": entry["count_total"],
                "count_total": entry["count_total"],
                "text_available_count": text_available_count,
                "patterns_available": text_available_count > 0,
            }
        )
        upsert_patterns(theme, pattern)
        themes.append(theme)
        audit_payload.append(
            {
                "theme": theme,
                "count_total": entry["count_total"],
                "patterns_available": text_available_count > 0,
            }
        )
    record("patterns_rebuild", {"themes": audit_payload})
    return themes


def _build_evidence_cards(
    theme: str,
    pattern: Dict[str, object],
    confidence: str,
) -> List[Dict[str, str]]:
    top_terms = pattern.get("top_terms") or []
    common_phrases = pattern.get("common_phrases") or []
    sentiment = pattern.get("sentiment_proxy") or {}
    return [
        {
            "title": "Common decision points",
            "detail": (
                "Recurring terms: "
                + (", ".join(top_terms[:5]) if top_terms else "Limited signals.")
            ),
            "confidence": confidence,
        },
        {
            "title": "Likely blockers",
            "detail": (
                "Frequent phrases: "
                + (", ".join(common_phrases[:3]) if common_phrases else "Limited signals.")
            ),
            "confidence": confidence,
        },
        {
            "title": "Typical questions to ask agency",
            "detail": (
                "Sentiment proxy: "
                f"negative={sentiment.get('negative', 0)}, positive={sentiment.get('positive', 0)}."
            ),
            "confidence": confidence,
        },
    ]


def _confidence_level(count: int) -> str:
    if count <= 0:
        return "LOW"
    if count < settings.k_threshold:
        return "MEDIUM"
    return "HIGH"


@app.get("/themes", response_model=List[ThemeSummary])
def themes() -> List[ThemeSummary]:
    summary = get_theme_counts()
    response: List[ThemeSummary] = []
    for entry in summary:
        count_total = entry["count_total"]
        response.append(
            ThemeSummary(
                theme=entry["theme"],
                count=count_total,
                count_total=count_total,
                meets_k=count_total >= settings.k_threshold,
            )
        )
    return response


@app.get("/evidence/cards")
def evidence_cards(theme: str) -> Dict[str, object]:
    pattern_entry = get_pattern(theme)
    if not pattern_entry:
        raise HTTPException(status_code=404, detail="Theme not found")
    pattern = pattern_entry["pattern"]
    theme_counts = get_theme_count(theme)
    count_total = (
        theme_counts["count_total"]
        if theme_counts
        else int(pattern.get("count_total") or pattern.get("count", 0))
    )
    text_available_count = len(get_texts_for_patterns(theme))
    confidence = _confidence_level(text_available_count)
    cards = _build_evidence_cards(theme, pattern, confidence)
    return {
        "theme": theme,
        "count_total": count_total,
        "evidence_cards": cards,
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> Dict[str, object]:
    pattern_entry = get_pattern(request.theme)
    if not pattern_entry:
        raise HTTPException(status_code=404, detail="Theme not found")
    pattern = pattern_entry["pattern"]
    theme_counts = get_theme_count(request.theme)
    count_total = (
        theme_counts["count_total"]
        if theme_counts
        else int(pattern.get("count_total") or pattern.get("count", 0))
    )
    if (
        count_total < settings.k_threshold
        and not request.allow_below_threshold
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Theme does not meet k-threshold of {settings.k_threshold}",
        )

    if request.kind == "enquiry":
        items = generate_pseudo_enquiries(request.theme, pattern, request.count)
    elif request.kind == "persona":
        items = generate_personas(request.theme, pattern, request.count)
    elif request.kind == "scenario":
        items = generate_scenarios(request.theme, pattern, request.count)
    else:
        raise HTTPException(status_code=400, detail="Unsupported generation kind")

    output = wrap_output(request.kind, items)
    output["theme"] = request.theme
    output["disclaimer"] = DISCLAIMER
    output["confidence"] = _confidence_level(count_total)
    output_id = insert_generated(request.theme, request.kind, output)
    record("generated", {"id": output_id, "theme": request.theme, "kind": request.kind})
    return {"id": output_id, **output}


@app.get("/generated")
def generated(theme: str | None = None) -> Dict[str, object]:
    records = get_generated(theme)
    return {"items": records}


@app.get("/audit/recent")
def audit_recent() -> Dict[str, object]:
    return {"items": get_audit_recent()}


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    clear_all()
    record("reset", {"status": "cleared"})
    return ResetResponse(status="ok", cleared=True)


@app.get("/ollama/status", response_model=OllamaStatus)
def ollama_status() -> OllamaStatus:
    try:
        import requests

        response = requests.get(settings.ollama_url, timeout=3)
        return OllamaStatus(available=response.status_code < 500)
    except Exception as exc:  # noqa: BLE001
        return OllamaStatus(available=False, detail=str(exc))


@app.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    return SettingsResponse(k_threshold=settings.k_threshold)
