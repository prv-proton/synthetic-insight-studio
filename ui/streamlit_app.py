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
        "Sanitize (Preview)",
        "Single Thread Analyzer",
        "Pattern overview",
        "Generate context pack",
        "Export markdown",
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
    st.subheader("Sanitize (Preview)")
    st.info(
        "This does not classify or store content. It only masks sensitive data for review."
    )
    extract_snippets = st.checkbox("Extract inspiration snippets", value=False)
    sanitize_source_type = st.selectbox(
        "Source type",
        options=["auto", "plain", "email"],
        help="Choose how to interpret pasted text for normalization.",
    )
    sanitize_text = st.text_area(
        "Paste text or email content",
        height=200,
    )
    sanitize_files = st.file_uploader(
        "Upload .txt, .eml, or .jsonl files",
        type=["txt", "eml", "jsonl"],
        accept_multiple_files=True,
    )
    if st.button("Mask sensitive info"):
        try:
            if sanitize_files:
                mode = "mask_and_extract_evidence" if extract_snippets else "mask_only"
                result = call_backend_files("/sanitize", sanitize_files, params={"mode": mode})
            elif sanitize_text.strip():
                mode = "mask_and_extract_evidence" if extract_snippets else "mask_only"
                result = call_backend(
                    "POST",
                    "/sanitize",
                    {
                        "text": sanitize_text,
                        "source_type": sanitize_source_type,
                        "mode": mode,
                    },
                )
            else:
                st.warning("Provide text or upload a file to sanitize.")
                result = None
            if result:
                items = result.get("items")
                if items:
                    for item in items:
                        with st.expander(item.get("filename", "Sanitized file"), expanded=True):
                            st.write(
                                f"Inferred source type: {item.get('inferred_source_type')}"
                            )
                            st.write("Normalization metadata:")
                            st.json(item.get("normalization_meta"))
                            st.write("Redaction stats:")
                            st.table([item.get("redaction_stats", {})])
                            st.text_area(
                                "Sanitized text",
                                value=item.get("sanitized_text", ""),
                                height=300,
                                disabled=True,
                            )
                            evidence_snippets = item.get("evidence_snippets")
                            if evidence_snippets:
                                st.write("Inspiration snippets:")
                                for snippet in evidence_snippets:
                                    st.code(snippet)
                                st.caption(
                                    "Snippets are extracted only from masked text; nothing is stored."
                                )
                else:
                    st.write(
                        f"Inferred source type: {result.get('inferred_source_type')}"
                    )
                    st.write("Normalization metadata:")
                    st.json(result.get("normalization_meta"))
                    st.write("Redaction stats:")
                    st.table([result.get("redaction_stats", {})])
                    st.text_area(
                        "Sanitized text",
                        value=result.get("sanitized_text", ""),
                        height=300,
                        disabled=True,
                    )
                    evidence_snippets = result.get("evidence_snippets")
                    if evidence_snippets:
                        st.write("Inspiration snippets:")
                        for snippet in evidence_snippets:
                            st.code(snippet)
                        st.caption(
                            "Snippets are extracted only from masked text; nothing is stored."
                        )
        except requests.RequestException as exc:
            st.error(f"Sanitize failed: {format_backend_error(exc)}")

with tabs[2]:
    st.subheader("Single Thread Analyzer")
    st.info("All outputs are synthetic/derived and redacted. Raw thread is not stored.")
    analyzer_source_type = st.selectbox(
        "Source type",
        options=["auto", "plain", "email"],
        help="Use email for .eml uploads or email-style pasted text.",
    )
    analyzer_text = st.text_area(
        "Paste a single email thread",
        height=200,
    )
    analyzer_file = st.file_uploader(
        "Upload a single .txt or .eml thread",
        type=["txt", "eml"],
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Mask Only"):
            try:
                if analyzer_file:
                    result = call_backend_files("/sanitize", [analyzer_file])
                    item = result.get("items", [{}])[0]
                elif analyzer_text.strip():
                    result = call_backend(
                        "POST",
                        "/sanitize",
                        {
                            "text": analyzer_text,
                            "source_type": analyzer_source_type,
                            "mode": "mask_only",
                        },
                    )
                    item = result
                else:
                    st.warning("Provide a thread to mask.")
                    item = None
                if item:
                    st.write("Redacted preview:")
                    sanitized_value = item.get("sanitized_text", "") or ""
                    st.text_area(
                        "Masked thread",
                        value=sanitized_value,
                        height=300,
                        disabled=True,
                    )
                    st.write("Redaction stats:")
                    st.table([item.get("redaction_stats", {})])
            except requests.RequestException as exc:
                st.error(f"Mask failed: {format_backend_error(exc)}")
    with col2:
        if st.button("Analyze Thread"):
            try:
                if analyzer_file:
                    result = call_backend_files("/thread/analyze", [analyzer_file], timeout=30)
                elif analyzer_text.strip():
                    result = call_backend(
                        "POST",
                        "/thread/analyze",
                        {
                            "text": analyzer_text,
                            "source_type": analyzer_source_type,
                        },
                        timeout=30,
                    )
                else:
                    st.warning("Provide a thread to analyze.")
                    result = None
                if result:
                    full_thread = result.get("full_thread_redacted", "")
                    if full_thread:
                        st.write("Full redacted thread:")
                        st.text_area(
                            "Full thread (masked)",
                            value=full_thread,
                            height=300,
                            disabled=True,
                        )
                    st.write("Latest redacted message (preview):")
                    latest_preview = result.get("latest_message_redacted", "")
                    st.text_area(
                        "Latest message (masked)",
                        value=latest_preview,
                        height=200,
                        disabled=True,
                    )
                    analysis = result.get("analysis", {})
                    st.markdown("### Thread Context Summary")
                    st.json(analysis.get("thread_summary", {}))
                    st.markdown("### Persona")
                    persona = analysis.get("persona", {})
                    st.write(
                        f"**{persona.get('persona_name', 'Persona')}** — "
                        f"{persona.get('role', 'unknown')} / "
                        f"{persona.get('experience_level', 'unknown')}"
                    )
                    st.write(f"Motivation: {persona.get('primary_motivation', '')}")
                    st.write(f"Tone: {persona.get('tone', '')}")
                    st.markdown("### Next Questions")
                    next_questions = analysis.get("next_questions", {})
                    st.write("To clarify:")
                    st.write(next_questions.get("to_clarify", []))
                    st.write("To unblock:")
                    st.write(next_questions.get("to_unblock", []))
                    st.write("Risks if ignored:")
                    st.write(next_questions.get("risks_if_ignored", []))
            except requests.RequestException as exc:
                st.error(f"Analyze failed: {format_backend_error(exc)}")

with tabs[3]:
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

with tabs[4]:
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
    )
    kind = st.selectbox("Kind", options=["enquiry", "persona", "scenario"])
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
                    "kind": kind,
                    "count": count,
                    "allow_below_threshold": True,
                }
                result = call_backend("POST", "/generate", payload)
                st.success("Generated synthetic outputs.")
                st.write(result.get("disclaimer"))
                if result.get("confidence"):
                    st.write(f"Confidence: {result.get('confidence')}")
                if result.get("note"):
                    st.info(result.get("note"))
                items = result.get("items", [])
                if kind == "enquiry" and items:
                    for idx, item in enumerate(items, start=1):
                        st.markdown(f"**Pseudo enquiry {idx}**")
                        if isinstance(item, dict):
                            tone = item.get("tone")
                            pressure = item.get("pressure")
                            if tone or pressure:
                                st.caption(f"Tone: {tone or 'n/a'} · Pressure: {pressure or 'n/a'}")
                            st.write(item.get("text", ""))
                            evidence = item.get("evidence", [])
                            if evidence:
                                st.caption("Evidence snippets (sanitized & anonymized)")
                                for snippet in evidence:
                                    st.code(snippet)
                            elif item.get("evidence_note"):
                                st.caption(item.get("evidence_note"))
                        else:
                            st.write(item)
                else:
                    st.write(items)
            except requests.RequestException as exc:
                st.error(f"Generation failed: {format_backend_error(exc)}")

    st.markdown("---")

with tabs[5]:
    st.subheader("Export markdown")
    try:
        generated = call_backend("GET", "/generated").get("items", [])
    except requests.RequestException:
        generated = []
    if generated:
        export_lines: List[str] = ["# Synthetic Insight Studio Export", "", "Synthetic / Exploratory — Not real user data", ""]
        for item in generated:
            export_lines.append(f"## {item['theme']} — {item['kind']}")
            export_lines.append("")
            content = item["content"]
            export_lines.append(content.get("disclaimer", ""))
            export_lines.append("")
            for entry in content.get("items", []):
                if isinstance(entry, dict):
                    export_lines.append(f"- {entry.get('text', '')}")
                    evidence_list = entry.get("evidence") or []
                    if evidence_list:
                        for snippet in evidence_list:
                            export_lines.append(f"  - evidence: {snippet}")
                    elif entry.get("evidence_note"):
                        export_lines.append(f"  - evidence: {entry.get('evidence_note')}")
                else:
                    export_lines.append(f"- {entry}")
            export_lines.append("")
        markdown = "\n".join(export_lines)
        st.download_button("Download markdown", data=markdown, file_name="synthetic_export.md")
        st.text_area("Preview", markdown, height=400)
    else:
        st.info("No generated items available yet.")

with tabs[6]:
    st.subheader("Audit log")
    try:
        audit = call_backend("GET", "/audit/recent").get("items", [])
        st.table(audit)
    except requests.RequestException as exc:
        st.error(f"Failed to load audit log: {format_backend_error(exc)}")
