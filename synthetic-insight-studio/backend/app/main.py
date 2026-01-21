from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import patterns as pattern_lib
from .audit import record
from .config import settings
from .generator import (
    DISCLAIMER,
    generate_personas,
    generate_pseudo_enquiries,
    generate_scenarios,
    wrap_output,
)
from .pii import redact
from .schemas import (
    GenerateRequest,
    IngestRequest,
    OllamaStatus,
    ResetResponse,
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
def ingest(request: IngestRequest) -> Dict[str, object]:
    sanitized: List[Dict[str, str]] = []
    raw_items: List[str] = []
    for item in request.enquiries:
        redacted, stats = redact(item.text)
        theme = pattern_lib.classify(redacted)
        sanitized.append({"text_sanitized": redacted, "theme": theme})
        raw_items.append(item.text)
        record("pii_redaction", {"theme": theme, "stats": stats})
    insert_enquiries(sanitized, raw_items)
    record("ingest", {"count": len(sanitized)})
    return {"status": "ok", "ingested": len(sanitized)}


@app.post("/patterns/rebuild")
def rebuild_patterns() -> Dict[str, object]:
    enquiries = get_sanitized_texts()
    pattern_map = pattern_lib.extract_patterns(enquiries)
    for theme, pattern in pattern_map.items():
        upsert_patterns(theme, pattern)
    record("patterns_rebuild", {"themes": list(pattern_map.keys())})
    return {"status": "ok", "themes": list(pattern_map.keys())}


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
