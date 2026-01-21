import json
import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Synthetic Insight Studio", layout="wide")

st.title("Synthetic Insight Studio")
st.info("Synthetic / Exploratory — Not real user data")

backend_url_default = os.getenv("BACKEND_URL", "http://backend:8000")
backend_url = st.sidebar.text_input("Backend URL", value=backend_url_default)


def call_backend(method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{backend_url}{path}"
    response = requests.request(method, url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def call_backend_files(path: str, files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Dict[str, Any]:
    url = f"{backend_url}{path}"
    payload = [
        ("files", (file.name, file.getvalue(), file.type or "text/plain"))
        for file in files
    ]
    response = requests.post(url, files=payload, timeout=10)
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
                    st.write(f"Risk level: {item.get('risk_level')}")
                    st.write("Risk explanation:")
                    st.write(", ".join(item.get("risk_reasons", [])))
                    st.write(f"Redaction summary: {item.get('redaction_counts', {})}")
                    if item.get("risk_level") == "HIGH":
                        st.info("Preview disabled for HIGH risk content.")
                    else:
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
                result = call_backend_files("/sanitize", sanitize_files)
            elif sanitize_text.strip():
                result = call_backend(
                    "POST",
                    "/sanitize",
                    {"text": sanitize_text, "source_type": sanitize_source_type},
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
        except requests.RequestException as exc:
            st.error(f"Sanitize failed: {format_backend_error(exc)}")

with tabs[2]:
    st.subheader("Pattern overview")
    st.caption("Themes are shown from aggregate counts; no raw text is displayed.")
    st.info(
        "High-risk uploads are parsed and counted, but their text is not stored. "
        "Themes remain visible to support early discovery."
    )
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
            labels = []
            if item.get("high_dominant"):
                labels.append("HIGH-RISK DOMINANT")
            if item.get("insight_quality") == "COUNTS_ONLY":
                labels.append("COUNTS ONLY")
            label_text = f" ({' / '.join(labels)})" if labels else ""
            theme_label = f"⚠️ {item.get('theme')}{label_text}" if labels else item.get("theme")
            table_rows.append(
                {
                    "theme": theme_label,
                    "total": item.get("count_total", item.get("count", 0)),
                    "low": item.get("count_low", 0),
                    "medium": item.get("count_medium", 0),
                    "high": item.get("count_high", 0),
                    "meets_k": item.get("meets_k", False),
                    "high_ratio": f"{item.get('high_ratio', 0):.2f}",
                    "insight_quality": item.get("insight_quality", "UNKNOWN"),
                }
            )
        df = pd.DataFrame(table_rows)

        def highlight_row(row: pd.Series) -> List[str]:
            if "HIGH-RISK DOMINANT" in str(row["theme"]):
                return ["background-color: #fff3cd"] * len(row)
            if "COUNTS ONLY" in str(row["theme"]):
                return ["background-color: #f8d7da"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(highlight_row, axis=1),
            use_container_width=True,
        )
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
    selected_theme = theme_details.get(theme)
    if selected_theme and (
        selected_theme.get("insight_quality") == "COUNTS_ONLY"
        or selected_theme.get("high_dominant")
    ):
        st.warning(
            "This theme is mostly high-risk. Generation will be generic and labeled lower confidence."
        )
    allow_below_threshold = True
    if is_below_threshold:
        st.info(f"{selected_count} record(s) available for this theme.")
        enforce_threshold = st.checkbox(
            "Enforce minimum record threshold",
            value=False,
        )
        allow_below_threshold = not enforce_threshold
    if st.button(
        "Generate",
        disabled=theme == "No themes",
    ):
        if theme == "No themes":
            st.warning("Load data and rebuild patterns first.")
        elif is_below_threshold:
            st.warning("Select a theme that meets the minimum record threshold.")
        else:
            try:
                payload = {
                    "theme": theme,
                    "kind": kind,
                    "count": count,
                    "allow_below_threshold": allow_below_threshold,
                }
                result = call_backend("POST", "/generate", payload)
                st.success("Generated synthetic outputs.")
                st.write(result.get("disclaimer"))
                if result.get("confidence"):
                    st.write(f"Confidence: {result.get('confidence')}")
                if result.get("note"):
                    st.info(result.get("note"))
                st.write(result.get("items", []))
            except requests.RequestException as exc:
                st.error(f"Generation failed: {format_backend_error(exc)}")

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
