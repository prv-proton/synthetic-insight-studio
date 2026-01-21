# Synthetic Insight Studio

Synthetic Insight Studio is a local demo that showcases a privacy-first pipeline:

**Ingestion → PII redaction → Pattern extraction/store → Synthetic generation → UI export**

All outputs are synthetic and privacy-safe. The system never exposes raw enquiries in the UI, and raw storage is disabled by default.

## Architecture

- **Backend:** FastAPI + SQLite
- **UI:** Streamlit
- **LLM:** Ollama (default, with graceful template fallback)
- **Guardrails:** k-threshold enforcement, post-generation PII scan, redaction

## Quick start (Docker Compose)

```bash
cd synthetic-insight-studio
docker-compose up --build
```

Open:
- Backend: http://localhost:8000/docs
- UI: http://localhost:8501

The stack will start Ollama and pull `qwen2:1.5b` by default.

## Demo script

1. Open the UI.
2. **Load data** → click **Load seed dataset**.
3. **Pattern overview** → click **Rebuild patterns**.
4. **Generate context pack** → pick a theme and generate enquiries/personas/scenarios.
5. **Export markdown** → download the synthetic export.

## Environment variables

Backend (`backend/app/config.py` defaults):

| Variable | Default |
| --- | --- |
| DB_PATH | `/data/app.db` |
| STORE_RAW | `false` |
| K_THRESHOLD | `10` |
| LLM_PROVIDER | `ollama` |
| OLLAMA_URL | `http://ollama:11434` |
| OLLAMA_MODEL | `qwen2:1.5b` |

## Security & privacy notes

- **No raw enquiry text is shown in the UI.**
- **Raw enquiries are not stored unless `STORE_RAW=true`.**
- Post-generation PII scans are enforced, with regeneration and redaction fallback.
- A **k-threshold** guardrail prevents theme generation with too few records.
- Every output is labeled: _“Synthetic / Exploratory — Not real user data.”_

## Supported ingestion formats

- **`.jsonl`** — multiple enquiries with `{ "text": "..." }` per line.
- **`.eml`** — email messages (treated as email-like text).
- **`.txt`** — raw text or email-like content (auto-detected and handled conservatively).

The prototype auto-detects structure for `.txt` uploads and always runs PII redaction. Raw file content is never displayed in the UI.

## OpenShift deployment

Manifests are in `openshift/`. Apply them in your OpenShift cluster:

```bash
oc apply -f openshift/
```

For Ollama, configure `OLLAMA_URL` to point at a platform-provided service.

## Model change

To switch models:

1. Update `OLLAMA_MODEL` in `docker-compose.yml` or in your deployment env.
2. Restart the stack. Ollama will pull the model on startup.
