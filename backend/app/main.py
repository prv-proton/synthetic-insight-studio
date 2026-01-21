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
    get_themes_summary,
    init_db,
    insert_enquiries,
    insert_generated,
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
    for item in request.enquiries:
        redacted, stats = redact(item.text)
        theme = pattern_lib.classify(redacted)
        sanitized.append({"text_sanitized": redacted, "theme": theme})
        raw_items.append(item.text)
        record("pii_redaction", {"theme": theme, "stats": stats})
    insert_enquiries(sanitized, raw_items)
    _rebuild_patterns()
    record("ingest", {"count": len(sanitized)})
    return {"status": "ok", "ingested": len(sanitized)}


async def _ingest_files(files: List[UploadFile]) -> Dict[str, object]:
    sanitized: List[Dict[str, str]] = []
    responses: List[Dict[str, object]] = []

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

        if should_store_sanitized(risk_level):
            theme = pattern_lib.classify(redacted)
            sanitized.append({"text_sanitized": redacted, "theme": theme})

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
    enquiries = get_sanitized_texts()
    pattern_map = pattern_lib.extract_patterns(enquiries)
    for theme, pattern in pattern_map.items():
        upsert_patterns(theme, pattern)
    themes = list(pattern_map.keys())
    record("patterns_rebuild", {"themes": themes})
    return themes


@app.get("/themes", response_model=List[ThemeSummary])
def themes() -> List[ThemeSummary]:
    summary = get_themes_summary()
    return [ThemeSummary(**row) for row in summary]


@app.post("/generate")
def generate(request: GenerateRequest) -> Dict[str, object]:
    pattern_entry = get_pattern(request.theme)
    if not pattern_entry:
        raise HTTPException(status_code=404, detail="Theme not found")
    pattern = pattern_entry["pattern"]
    if pattern.get("count", 0) < settings.k_threshold:
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
