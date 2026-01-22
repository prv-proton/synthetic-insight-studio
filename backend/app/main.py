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
)
from .ingestion import infer_source_type, trim_excess_whitespace
from .schemas import (
    GenerateRequest,
    IngestRequest,
    OllamaStatus,
    ResetResponse,
    SettingsResponse,
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


@app.get("/evidence/library")
def evidence_library(topic: str = "permit_housing", limit: int = 5) -> Dict[str, object]:
    snippets = _load_evidence_library(topic, limit)
    return {"topic": topic, "items": snippets}


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

    pseudo_enquiries = generate_pseudo_enquiries(request.theme, pattern, request.count)
    personas = generate_personas(request.theme, pattern, request.count)
    scenarios = generate_scenarios(request.theme, pattern, request.count)
    evidence = {
        "theme_count": count_total,
        "top_terms": pattern.get("top_terms", [])[:8],
        "common_blockers": pattern.get("common_phrases", [])[:5],
    }
    response = {
        "theme": request.theme,
        "pseudo_enquiries": pseudo_enquiries,
        "personas": personas,
        "scenarios": scenarios,
        "evidence": evidence,
    }
    output_id = insert_generated(request.theme, "context_pack", response)
    record(
        "generated_context_pack",
        {"id": output_id, "theme": request.theme, "kind": "context_pack"},
    )
    return {"id": output_id, **response}


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
