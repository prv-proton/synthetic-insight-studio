import json
import os
from urllib.parse import quote
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Synthetic Insight Studio", layout="wide")

st.title("Synthetic Insight Studio")
st.info("Synthetic / Exploratory — Not real user data")

backend_url_default = os.getenv("BACKEND_URL", "http://backend:8000")
backend_url = st.sidebar.text_input("Backend URL", value=backend_url_default)


def call_backend(
    method: str,
    path: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    url = f"{backend_url}{path}"
    response = requests.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def call_backend_files(
    path: str,
    files: List[st.runtime.uploaded_file_manager.UploadedFile],
    params: Dict[str, Any] | None = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    url = f"{backend_url}{path}"
    payload = [
        ("files", (file.name, file.getvalue(), file.type or "text/plain"))
        for file in files
    ]
    response = requests.post(url, files=payload, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def format_backend_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        data = response.json()
    except ValueError:
        data = response.text
    if isinstance(data, dict):
        detail = data.get("detail")
        if detail:
            return f"{response.status_code} {response.reason}: {detail}"
    if isinstance(data, str) and data.strip():
        return f"{response.status_code} {response.reason}: {data}"
    return f"{response.status_code} {response.reason}"


if st.sidebar.button("Clear stored data"):
    try:
        result = call_backend("POST", "/reset")
        if result.get("cleared"):
            st.sidebar.success("Stored data cleared.")
        else:
            st.sidebar.warning("Clear request completed with warnings.")
    except requests.RequestException as exc:
        st.sidebar.error(f"Clear failed: {format_backend_error(exc)}")


def get_ollama_status() -> str:
    try:
        status = call_backend("GET", "/ollama/status")
        return "Available" if status.get("available") else "Unavailable"
    except requests.RequestException:
        return "Unavailable"


def get_k_threshold() -> int | None:
    try:
        settings = call_backend("GET", "/settings")
        return settings.get("k_threshold")
    except requests.RequestException:
        return None


st.sidebar.markdown("### LLM Status")
st.sidebar.write(get_ollama_status())

tabs = st.tabs(
    [
        "Load data",
        "Pattern overview",
        "Generate context pack",
        "Sanitize (Preview)",
        "Audit log",
    ]
)

with tabs[0]:
    st.subheader("Load synthetic data")
    st.caption("Raw enquiries are never displayed; only sanitized patterns are used.")
    load_seed = st.button("Load seed dataset")
    if load_seed:
        try:
            with open("/data/enquiries_seed.jsonl", "r", encoding="utf-8") as handle:
                enquiries = [{"text": json.loads(line)["text"]} for line in handle if line.strip()]
            payload = {"enquiries": enquiries}
            result = call_backend("POST", "/ingest", payload)
            st.success(f"Ingested {result.get('ingested')} synthetic enquiries.")
        except FileNotFoundError:
            st.error("Seed data not found in container.")
        except requests.RequestException as exc:
            st.error(f"Failed to ingest data: {format_backend_error(exc)}")

    st.markdown("---")
    st.subheader("Upload synthetic dataset")
    file = st.file_uploader("Upload JSONL with {\"text\": \"...\"} entries", type=["jsonl"])
    if file:
        try:
            entries = []
            for line in file.getvalue().decode("utf-8").splitlines():
                if line.strip():
                    entries.append({"text": json.loads(line)["text"]})
            result = call_backend("POST", "/ingest", {"enquiries": entries})
            st.success(f"Ingested {result.get('ingested')} synthetic enquiries.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")

    st.markdown("---")
    st.subheader("Upload text or email files")
    st.caption("Raw file content is not stored or displayed.")
    upload_files = st.file_uploader(
        "Upload .txt or .eml files",
        type=["txt", "eml"],
        accept_multiple_files=True,
    )
    if upload_files and st.button("Ingest text files"):
        try:
            result = call_backend_files("/ingest", upload_files)
            st.success(f"Ingested {result.get('ingested')} text files.")
            items = result.get("items", [])
            for index, item in enumerate(items):
                with st.expander(f"{item.get('filename', 'upload')} details", expanded=True):
                    st.write(f"Inferred content type: {item.get('inferred_source_type')}")
                    st.write(f"Redaction summary: {item.get('redaction_counts', {})}")
                    toggle_key = f"preview_{index}"
                    if st.toggle("Preview sanitized snippet", value=False, key=toggle_key):
                        sanitized_text = item.get("sanitized_text") or ""
                        st.code(sanitized_text[:300])
        except requests.RequestException as exc:
            st.error(f"Upload failed: {format_backend_error(exc)}")

with tabs[1]:
    st.subheader("Pattern overview")
    st.caption("Themes are shown from aggregate counts; no raw text is displayed.")
    if st.button("Rebuild patterns"):
        try:
            result = call_backend("POST", "/patterns/rebuild")
            st.success(f"Patterns rebuilt for {len(result.get('themes', []))} themes.")
        except requests.RequestException as exc:
            st.error(f"Failed to rebuild patterns: {format_backend_error(exc)}")

    try:
        summary = call_backend("GET", "/themes")
    except requests.RequestException:
        summary = []
    if summary:
        table_rows = []
        for item in summary:
            table_rows.append(
                {
                    "theme": item.get("theme"),
                    "total": item.get("count_total", item.get("count", 0)),
                    "meets_k": item.get("meets_k", False),
                }
            )
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No themes available yet. Load data to see counts.")

with tabs[2]:
    st.subheader("Generate synthetic context pack")
    st.caption("Outputs are synthetic and privacy-safe.")
    try:
        themes = call_backend("GET", "/themes")
    except requests.RequestException as exc:
        themes = []
        st.error(f"Failed to load themes: {format_backend_error(exc)}")
    theme_details = {
        item.get("theme"): item for item in themes if isinstance(item, dict)
    }
    count_by_theme = {
        item.get("theme"): item.get("count_total", item.get("count", 0))
        for item in themes
        if isinstance(item, dict)
    }
    theme_options = list(count_by_theme.keys()) if count_by_theme else []
    theme = st.selectbox(
        "Theme",
        options=theme_options if theme_options else ["No themes"],
        format_func=lambda name: (
            f"{name} ({count_by_theme.get(name, 0)} records)"
            if name != "No themes"
            else "No themes"
        ),
        key="generate_theme",
    )
    count = st.slider("Count", min_value=1, max_value=10, value=5)
    k_threshold = get_k_threshold()
    if k_threshold is not None:
        st.info(f"Minimum records per theme: {k_threshold}.")
    selected_count = count_by_theme.get(theme, 0) if theme != "No themes" else 0
    is_below_threshold = (
        k_threshold is not None and theme != "No themes" and selected_count < k_threshold
    )
    if st.button(
        "Generate",
        disabled=theme == "No themes",
    ):
        if theme == "No themes":
            st.warning("Load data and rebuild patterns first.")
        else:
            try:
                payload = {
                    "theme": theme,
                    "count": count,
                    "allow_below_threshold": True,
                }
                result = call_backend("POST", "/generate", payload, timeout=60)
                st.success("Generated synthetic outputs.")
                st.markdown("### Pseudo enquiries")
                for enquiry in result.get("pseudo_enquiries", []):
                    st.write(f"- {enquiry}")
                st.markdown("### Personas")
                for persona in result.get("personas", []):
                    st.json(persona)
                st.markdown("### Scenarios")
                for scenario in result.get("scenarios", []):
                    st.json(scenario)
                evidence = result.get("evidence", {})
                if evidence:
                    st.markdown("### Evidence")
                    st.write(f"Theme count: {evidence.get('theme_count')}")
                    st.write(f"Top terms: {evidence.get('top_terms', [])}")
                    st.write(f"Common blockers: {evidence.get('common_blockers', [])}")
            except requests.RequestException as exc:
                st.error(f"Generation failed: {format_backend_error(exc)}")

    st.markdown("---")
    st.subheader("Evidence grounding")
    st.caption("Evidence is derived from masked text or aggregate patterns only.")
    evidence_mode = st.radio(
        "Evidence source",
        options=["Aggregated evidence cards", "Upload sample for masked snippets"],
        horizontal=True,
    )
    if evidence_mode == "Aggregated evidence cards":
        if theme == "No themes":
            st.info("Select a theme to load evidence cards.")
        elif st.button("Load evidence cards"):
            try:
                cards_response = call_backend(
                    "GET", f"/evidence/cards?theme={quote(theme)}"
                )
                cards = cards_response.get("evidence_cards", [])
                if cards:
                    for card in cards:
                        st.markdown(f"**{card.get('title')}**")
                        st.write(card.get("detail"))
                        st.caption(f"Confidence: {card.get('confidence')}")
                else:
                    st.info("No evidence cards available.")
            except requests.RequestException as exc:
                st.error(f"Failed to load evidence cards: {format_backend_error(exc)}")
    else:
        sample_file = st.file_uploader(
            "Upload a sample .txt or .eml file for masked snippets",
            type=["txt", "eml"],
        )
        if sample_file and st.button("Extract masked snippets"):
            try:
                result = call_backend_files(
                    "/sanitize",
                    [sample_file],
                    params={"mode": "mask_and_extract_evidence"},
                )
                items = result.get("items", [])
                evidence_snippets = items[0].get("evidence_snippets") if items else []
                if evidence_snippets:
                    st.write("Masked inspiration snippets:")
                    for snippet in evidence_snippets:
                        st.code(snippet)
                else:
                    st.info("No snippets extracted from the sample.")
                st.caption(
                    "Snippets are extracted only from masked text; nothing is stored."
                )
            except requests.RequestException as exc:
                st.error(f"Snippet extraction failed: {format_backend_error(exc)}")

with tabs[3]:
    st.subheader("Sanitize (Preview)")
    st.caption("Preview masked text without storing or classifying it.")
    mode = st.selectbox(
        "Sanitization mode",
        options=["mask_only", "mask_and_extract_evidence"],
        help="mask_only returns sanitized text; mask_and_extract_evidence also returns snippet candidates.",
    )
    source_type = st.selectbox(
        "Source type",
        options=["auto", "plain", "email"],
        help="Email detection is used only to strip headers and quoted content.",
    )
    text_input = st.text_area("Paste enquiry text", height=200)
    if st.button("Sanitize text"):
        if not text_input.strip():
            st.warning("Provide text to sanitize.")
        else:
            try:
                result = call_backend(
                    "POST",
                    "/sanitize",
                    {"text": text_input, "source_type": source_type, "mode": mode},
                )
                st.markdown("### Sanitized text")
                st.code(result.get("sanitized_text", ""))
                st.write(f"Redaction stats: {result.get('redaction_stats', {})}")
                if mode == "mask_and_extract_evidence":
                    st.markdown("### Evidence snippets")
                    st.write(result.get("evidence_snippets", []))
            except requests.RequestException as exc:
                st.error(f"Sanitize failed: {format_backend_error(exc)}")

    st.markdown("---")
    st.subheader("Sanitize files")
    sanitize_files = st.file_uploader(
        "Upload .txt, .eml, or .jsonl files",
        type=["txt", "eml", "jsonl"],
        accept_multiple_files=True,
    )
    if sanitize_files and st.button("Sanitize files"):
        try:
            result = call_backend_files("/sanitize", sanitize_files, params={"mode": mode})
            for item in result.get("items", []):
                filename = item.get("filename", "upload")
                st.markdown(f"**{filename}**")
                st.code(item.get("sanitized_text", "")[:400])
                st.write(f"Redaction stats: {item.get('redaction_stats', {})}")
                if mode == "mask_and_extract_evidence":
                    st.write(item.get("evidence_snippets", []))
        except requests.RequestException as exc:
            st.error(f"Sanitize failed: {format_backend_error(exc)}")

with tabs[4]:
    st.subheader("Audit log")
    try:
        audit = call_backend("GET", "/audit/recent").get("items", [])
        st.table(audit)
    except requests.RequestException as exc:
        st.error(f"Failed to load audit log: {format_backend_error(exc)}")
