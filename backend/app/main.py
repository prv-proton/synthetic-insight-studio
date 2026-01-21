from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import patterns as pattern_lib
from .audit import record
from .config import settings
from .email_normalize import normalize_email
from .generator import (
    DISCLAIMER,
    generate_personas,
    generate_pseudo_enquiries,
    generate_scenarios,
    wrap_output,
)
from .ingestion import (
    assess_risk,
    detect_attachment_hints,
    detect_quoted_thread,
    infer_source_type,
    should_store_sanitized,
    trim_excess_whitespace,
)
from .pii import detect_pii, redact
from .tag_redact import redact_tagged
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


def _ingest_payload(request: IngestRequest) -> Dict[str, object]:
    sanitized: List[Dict[str, str]] = []
    raw_items: List[str] = []
    theme_counts: List[Dict[str, int]] = []
    for item in request.enquiries:
        redacted, stats = redact(item.text)
        theme = pattern_lib.classify(redacted)
        inferred_source_type = infer_source_type(item.text)
        post_redaction_findings = detect_pii(redacted)
        quoted_thread = detect_quoted_thread(item.text)
        attachment_hints = detect_attachment_hints(item.text)
        risk_level, _ = assess_risk(
            stats,
            post_redaction_findings,
            inferred_source_type == "email_like",
            quoted_thread,
            attachment_hints,
        )
        if should_store_sanitized(risk_level):
            sanitized.append({"text_sanitized": redacted, "theme": theme})
        raw_items.append(item.text)
        record("pii_redaction", {"theme": theme, "stats": stats})
        theme_counts.append(
            {
                "theme": theme,
                "count_low": 1 if risk_level == "LOW" else 0,
                "count_medium": 1 if risk_level == "MEDIUM" else 0,
                "count_high": 1 if risk_level == "HIGH" else 0,
            }
        )
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
        normalized = normalize_email(text)
        working_text = normalized.normalized_text
        normalization_meta = normalized.meta
        normalization_applied = True

    redacted, redaction_stats = redact(working_text)
    post_redaction_findings = detect_pii(redacted)
    quoted_thread = detect_quoted_thread(working_text)
    attachment_hints = detect_attachment_hints(working_text)
    risk_level, risk_reasons = assess_risk(
        redaction_stats,
        post_redaction_findings,
        inferred_source_type == "email_like",
        quoted_thread,
        attachment_hints,
    )
    storage_decision = (
        "sanitized_text_stored"
        if should_store_sanitized(risk_level)
        else "aggregates_only"
    )

    theme = pattern_lib.classify(redacted)
    if should_store_sanitized(risk_level):
        sanitized.append({"text_sanitized": redacted, "theme": theme})

    theme_counts.append(
        {
            "theme": theme,
            "count_low": 1 if risk_level == "LOW" else 0,
            "count_medium": 1 if risk_level == "MEDIUM" else 0,
            "count_high": 1 if risk_level == "HIGH" else 0,
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
            "post_redaction_findings": len(post_redaction_findings),
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
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
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
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
    normalized_text = text
    normalization_meta: Dict[str, object] | None = None
    if source_type not in {"auto", "plain", "email"}:
        raise HTTPException(
            status_code=400,
            detail="source_type must be one of: plain, email, auto",
        )
    if source_type == "auto":
        inferred_source_type = infer_source_type(text)
    elif source_type == "plain":
        inferred_source_type = "plain_text"
    else:
        inferred_source_type = "email"
    if inferred_source_type in {"email", "email_like"}:
        normalized = normalize_email(text)
        normalized_text = normalized.normalized_text
        normalization_meta = normalized.meta
    if mode not in {"mask_only", "mask_and_extract_evidence"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: mask_only, mask_and_extract_evidence",
        )
    tagged_text, tag_redaction_stats = redact_tagged(normalized_text)
    sanitized_text, pii_redaction_stats = redact(tagged_text)
    response: Dict[str, object] = {
        "sanitized_text": sanitized_text,
        "tag_redaction_stats": tag_redaction_stats,
        "pii_redaction_stats": pii_redaction_stats,
        "inferred_source_type": inferred_source_type,
        "normalization_meta": normalization_meta,
        "notes": [
            "Sanitize-only mode: no classification, no storage, no pattern extraction."
        ],
    }
    if mode == "mask_and_extract_evidence":
        snippets = _extract_evidence_snippets(sanitized_text)
        sanitized_snippets: List[str] = []
        for snippet in snippets:
            snippet_tagged, _ = redact_tagged(snippet)
            snippet_redacted, _ = redact(snippet_tagged)
            sanitized_snippets.append(snippet_redacted)
        response["evidence_snippets"] = sanitized_snippets
    if filename:
        response["filename"] = filename
    audit = {
        "filename": filename,
        "inferred_source_type": inferred_source_type,
        "tag_redaction_stats": tag_redaction_stats,
        "pii_redaction_stats": pii_redaction_stats,
        "normalized": normalization_meta is not None,
    }
    return response, audit


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
                "count_low": entry["count_low"],
                "count_medium": entry["count_medium"],
                "count_high": entry["count_high"],
                "text_available_count": text_available_count,
                "patterns_available": text_available_count > 0,
                "counts_only": text_available_count == 0,
            }
        )
        upsert_patterns(theme, pattern)
        themes.append(theme)
        insight_quality = _insight_quality(text_available_count)
        audit_payload.append(
            {
                "theme": theme,
                "count_total": entry["count_total"],
                "count_high": entry["count_high"],
                "insight_quality": insight_quality,
                "patterns_available": text_available_count > 0,
            }
        )
    record("patterns_rebuild", {"themes": audit_payload})
    return themes


def _insight_quality(text_available_count: int) -> str:
    threshold = max(5, settings.k_threshold / 2)
    if text_available_count == 0:
        return "COUNTS_ONLY"
    if text_available_count < threshold:
        return "LIMITED"
    return "STRONG"


def _high_ratio(count_high: int, count_total: int) -> float:
    denominator = max(count_total, 1)
    return count_high / denominator


def _evidence_confidence(insight_quality: str) -> str:
    if insight_quality == "LIMITED":
        return "MEDIUM"
    return "HIGH"


def _build_evidence_cards(
    theme: str,
    pattern: Dict[str, object],
    insight_quality: str,
) -> List[Dict[str, str]]:
    if insight_quality == "COUNTS_ONLY":
        return [
            {
                "title": "Common decision points",
                "detail": f"Aggregate counts for {theme} only; no text patterns retained.",
                "confidence": "LOW",
            },
            {
                "title": "Likely blockers",
                "detail": "Generic blockers inferred from count-only aggregates.",
                "confidence": "LOW",
            },
            {
                "title": "Typical questions to ask agency",
                "detail": "Use standard permitting checklists; no text-derived signals.",
                "confidence": "LOW",
            },
        ]
    top_terms = pattern.get("top_terms") or []
    common_phrases = pattern.get("common_phrases") or []
    sentiment = pattern.get("sentiment_proxy") or {}
    confidence = _evidence_confidence(insight_quality)
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


@app.get("/themes", response_model=List[ThemeSummary])
def themes() -> List[ThemeSummary]:
    summary = get_theme_counts()
    response: List[ThemeSummary] = []
    for entry in summary:
        count_total = entry["count_total"]
        count_high = entry["count_high"]
        count_low = entry["count_low"]
        count_medium = entry["count_medium"]
        text_available_count = count_low + count_medium + count_high
        high_ratio = _high_ratio(count_high, count_total)
        high_dominant = count_high >= 1 and high_ratio >= settings.high_dominant_ratio
        insight_quality = _insight_quality(text_available_count)
        patterns_available = text_available_count > 0
        message = None
        if insight_quality == "COUNTS_ONLY":
            message = (
                "Only aggregate counts available; no sanitized text retained for pattern extraction."
            )
        elif high_dominant:
            message = (
                "Theme has sufficient volume but is mostly high-risk; only aggregate signals retained."
            )
        response.append(
            ThemeSummary(
                theme=entry["theme"],
                count=count_total,
                count_total=count_total,
                count_low=count_low,
                count_medium=count_medium,
                count_high=count_high,
                meets_k=count_total >= settings.k_threshold,
                high_ratio=high_ratio,
                high_dominant=high_dominant,
                text_available_count=text_available_count,
                patterns_available=patterns_available,
                insight_quality=insight_quality,
                message=message,
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
    count_low = theme_counts["count_low"] if theme_counts else int(pattern.get("count_low", 0))
    count_medium = (
        theme_counts["count_medium"] if theme_counts else int(pattern.get("count_medium", 0))
    )
    count_high = theme_counts["count_high"] if theme_counts else int(pattern.get("count_high", 0))
    text_available_count = count_low + count_medium + count_high
    insight_quality = _insight_quality(text_available_count)
    cards = _build_evidence_cards(theme, pattern, insight_quality)
    return {
        "theme": theme,
        "count_total": count_total,
        "insight_quality": insight_quality,
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
    count_low = theme_counts["count_low"] if theme_counts else int(pattern.get("count_low", 0))
    count_medium = (
        theme_counts["count_medium"] if theme_counts else int(pattern.get("count_medium", 0))
    )
    count_high = theme_counts["count_high"] if theme_counts else int(pattern.get("count_high", 0))
    text_available_count = count_low + count_medium + count_high
    insight_quality = _insight_quality(text_available_count)
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
    if insight_quality == "COUNTS_ONLY":
        output["confidence"] = "LOW"
        output["note"] = (
            "Generated from aggregate counts only; no text patterns available."
        )
    elif insight_quality == "LIMITED":
        output["confidence"] = "MEDIUM"
    else:
        output["confidence"] = "HIGH"
    output["insight_quality"] = insight_quality
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
