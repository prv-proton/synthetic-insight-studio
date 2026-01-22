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
        "Generate realistic pseudo enquiry",
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
    st.subheader("Generate realistic pseudo enquiry")
    st.info("Synthetic / Exploratory — Not real user data")
    st.caption("Evidence snippets are anonymized inspiration (not verbatim).")
    pseudo_enhanced = st.toggle(
        "Enhanced LLM synthesis (more context-aware, slower)",
        value=True,
        key="pseudo_enhanced",
    )
    mode = st.radio(
        "Mode",
        options=["From a single thread"],
        horizontal=True,
    )
    if mode == "From a single thread":
        pseudo_source_type = st.selectbox(
            "Thread source type",
            options=["auto", "plain", "email"],
            key="pseudo_source_type",
        )
        pseudo_text = st.text_area(
            "Paste a single email thread",
            height=200,
            key="pseudo_text",
        )
        pseudo_file = st.file_uploader(
            "Upload a single .txt or .eml thread",
            type=["txt", "eml"],
            key="pseudo_file",
        )
        evidence_source = st.selectbox(
            "Evidence snippet source",
            options=["library", "uploaded", "none"],
            help="Uploaded uses sanitized snippets from this thread.",
            key="evidence_source",
        )
        if st.button("Generate pseudo enquiry", key="generate_pseudo_enquiry_thread"):
            try:
                thread_text = None
                if pseudo_file:
                    thread_text = pseudo_file.getvalue().decode("utf-8", errors="replace")
                elif pseudo_text.strip():
                    thread_text = pseudo_text
                if not thread_text:
                    st.warning("Provide a thread to generate from.")
                else:
                    sanitize_result = call_backend(
                        "POST",
                        "/sanitize",
                        {
                            "text": thread_text,
                            "source_type": pseudo_source_type,
                            "mode": "mask_only",
                        },
                    )
                    st.write("Sanitized preview:")
                    st.code((sanitize_result.get("sanitized_text", "") or "")[:400])
                    result = call_backend(
                        "POST",
                        "/generate/pseudo_email",
                        {
                            "source": "thread_analyze",
                            "thread_text": thread_text,
                            "evidence_source": evidence_source,
                            "n_snippets": 5,
                            "enhanced": pseudo_enhanced,
                        },
                        timeout=30,
                    )
                    pseudo_email = result.get("pseudo_email", {})
                    st.markdown("### Pseudo enquiry")
                    st.write(f"**Subject:** {pseudo_email.get('subject', '')}")
                    st.write(f"**From role:** {pseudo_email.get('from_role', '')}")
                    st.write(f"**Tone:** {pseudo_email.get('tone', '')}")
                    st.text_area(
                        "Email body",
                        value=pseudo_email.get("body", ""),
                        height=260,
                        disabled=True,
                    )
                    st.write(
                        f"Attachments mentioned: {pseudo_email.get('attachments_mentioned', [])}"
                    )
                    st.write(
                        f"Placeholders used: {pseudo_email.get('placeholders_used', [])}"
                    )
                    analysis = result.get("analysis", {})
                    if analysis:
                        st.markdown("### Persona")
                        persona = analysis.get("persona", {})
                        st.write(
                            f"**{persona.get('persona_name', 'Persona')}** — "
                            f"{persona.get('from_role', 'unknown')} / "
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
                    st.markdown("### Evidence grounding")
                    st.write(result.get("inspired_by", ""))
                    st.caption(
                        f"Evidence method: {result.get('evidence_method', 'patterns_only')}"
                    )
                    st.write(result.get("evidence_snippets", []))
                    email_quality = result.get("quality_signals", {})
                    if email_quality:
                        st.markdown("### Quality signals")
                        st.write(f"Used repair: {email_quality.get('used_repair', False)}")
                        st.write(f"Used improve: {email_quality.get('used_improve', False)}")
                        st.write(f"Issues: {email_quality.get('issues', [])}")
            except requests.RequestException as exc:
                st.error(f"Generate failed: {format_backend_error(exc)}")
    else:
        try:
            theme_items = call_backend("GET", "/themes")
        except requests.RequestException as exc:
            theme_items = []
            st.error(f"Failed to load themes: {format_backend_error(exc)}")
        theme_options = [
            item.get("theme") for item in theme_items if isinstance(item, dict)
        ]
        theme_choice = st.selectbox(
            "Theme",
            options=theme_options if theme_options else ["No themes"],
            key="theme_choice",
        )
        if st.button(
            "Generate pseudo enquiry from theme",
            disabled=theme_choice == "No themes",
            key="generate_pseudo_enquiry_theme",
        ):
            try:
                result = call_backend(
                    "POST",
                    "/generate/pseudo_email",
                    {
                        "source": "patterns",
                        "theme": theme_choice,
                        "evidence_source": "library",
                        "n_snippets": 5,
                        "enhanced": pseudo_enhanced,
                    },
                    timeout=30,
                )
                pseudo_email = result.get("pseudo_email", {})
                st.markdown("### Pseudo enquiry")
                st.write(f"**Subject:** {pseudo_email.get('subject', '')}")
                st.write(f"**From role:** {pseudo_email.get('from_role', '')}")
                st.write(f"**Tone:** {pseudo_email.get('tone', '')}")
                st.text_area(
                    "Email body",
                    value=pseudo_email.get("body", ""),
                    height=260,
                    disabled=True,
                )
                st.markdown("### Evidence grounding")
                st.write(result.get("inspired_by", ""))
                st.caption(
                    f"Evidence method: {result.get('evidence_method', 'patterns_only')}"
                )
                st.write(result.get("evidence_snippets", []))
                email_quality = result.get("quality_signals", {})
                if email_quality:
                    st.markdown("### Quality signals")
                    st.write(f"Used repair: {email_quality.get('used_repair', False)}")
                    st.write(f"Used improve: {email_quality.get('used_improve', False)}")
                    st.write(f"Issues: {email_quality.get('issues', [])}")
            except requests.RequestException as exc:
                st.error(f"Generate failed: {format_backend_error(exc)}")

with tabs[2]:
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

with tabs[3]:
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
    kind = st.selectbox("Kind", options=["enquiry", "persona", "scenario"], key="generate_kind")
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
                result = call_backend("POST", "/generate", payload, timeout=60)
                st.success("Generated synthetic outputs.")
                st.write(result.get("disclaimer"))
                if result.get("confidence"):
                    st.write(f"Confidence: {result.get('confidence')}")
                if result.get("note"):
                    st.info(result.get("note"))
                st.write(result.get("items", []))
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

with tabs[4]:
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
                export_lines.append(f"- {entry}")
            export_lines.append("")
        markdown = "\n".join(export_lines)
        st.download_button("Download markdown", data=markdown, file_name="synthetic_export.md")
        st.text_area("Preview", markdown, height=400)
    else:
        st.info("No generated items available yet.")

with tabs[5]:
    st.subheader("Audit log")
    try:
        audit = call_backend("GET", "/audit/recent").get("items", [])
        st.table(audit)
    except requests.RequestException as exc:
        st.error(f"Failed to load audit log: {format_backend_error(exc)}")