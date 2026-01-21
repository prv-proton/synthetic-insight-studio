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
                "sanitized_text": redacted if risk_level != "HIGH" else None,
            }
        )

    upsert_theme_counts(theme_counts)
    if sanitized:
        insert_enquiries(sanitized, raw=None)
    _rebuild_patterns()
    record("ingest", {"count": len(responses)})
    return {"status": "ok", "ingested": len(responses), "items": responses}


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
        texts = get_sanitized_texts(theme)
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


@app.get("/themes", response_model=List[ThemeSummary])
def themes() -> List[ThemeSummary]:
    summary = get_theme_counts()
    response: List[ThemeSummary] = []
    for entry in summary:
        count_total = entry["count_total"]
        count_high = entry["count_high"]
        count_low = entry["count_low"]
        count_medium = entry["count_medium"]
        text_available_count = count_low + count_medium
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
    text_available_count = count_low + count_medium
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
