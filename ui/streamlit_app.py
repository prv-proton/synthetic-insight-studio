import json
import os
from typing import Any, Dict, List

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


def get_ollama_status() -> str:
    try:
        status = call_backend("GET", "/ollama/status")
        return "Available" if status.get("available") else "Unavailable"
    except requests.RequestException:
        return "Unavailable"


st.sidebar.markdown("### LLM Status")
st.sidebar.write(get_ollama_status())

tabs = st.tabs(
    [
        "Load data",
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
            st.error(f"Failed to ingest data: {exc}")

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

with tabs[1]:
    st.subheader("Pattern overview")
    st.caption("Themes only appear after rebuild; no raw text is shown.")
    if st.button("Rebuild patterns"):
        try:
            result = call_backend("POST", "/patterns/rebuild")
            st.success(f"Patterns rebuilt for {len(result.get('themes', []))} themes.")
        except requests.RequestException as exc:
            st.error(f"Failed to rebuild patterns: {exc}")

    try:
        summary = call_backend("GET", "/themes")
    except requests.RequestException:
        summary = []
    st.table(summary)

with tabs[2]:
    st.subheader("Generate synthetic context pack")
    st.caption("Outputs are synthetic and privacy-safe, with k-threshold enforcement.")
    try:
        themes = call_backend("GET", "/themes")
    except requests.RequestException as exc:
        themes = []
        st.error(f"Failed to load themes: {exc}")
    theme_options = [item["theme"] for item in themes] if isinstance(themes, list) else []
    theme = st.selectbox("Theme", options=theme_options if theme_options else ["No themes"])
    kind = st.selectbox("Kind", options=["enquiry", "persona", "scenario"])
    count = st.slider("Count", min_value=1, max_value=10, value=5)
    if st.button("Generate"):
        if theme == "No themes":
            st.warning("Load data and rebuild patterns first.")
        else:
            try:
                payload = {"theme": theme, "kind": kind, "count": count}
                result = call_backend("POST", "/generate", payload)
                st.success("Generated synthetic outputs.")
                st.write(result.get("disclaimer"))
                st.write(result.get("items", []))
            except requests.RequestException as exc:
                st.error(f"Generation failed: {exc}")

with tabs[3]:
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

with tabs[4]:
    st.subheader("Audit log")
    try:
        audit = call_backend("GET", "/audit/recent").get("items", [])
        st.table(audit)
    except requests.RequestException as exc:
        st.error(f"Failed to load audit log: {exc}")
