from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from ingestion_agent.agent import build_graph, initial_state
from query_agent.agent import build_query_graph, initial_query_state
from query_agent.constants import DEFAULT_TOP_K

_LOCAL_MODEL = str(_REPO_ROOT / "models" / "e5-large-v2")
_EMBEDDING_MODEL = _LOCAL_MODEL if Path(_LOCAL_MODEL).is_dir() else "intfloat/e5-large-v2"
_CHROMA_PATH = str(_REPO_ROOT / "chroma_db")
_HISTORY_FILE = Path(__file__).parent / "history.json"
_PASSAGE_PREFIX = "passage: "


# ── cached query graph ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_query_graph():
    graph, lm_available, resolved_model = build_query_graph(
        chroma_path=_CHROMA_PATH,
        embedding_model=_EMBEDDING_MODEL,
    )
    return graph, lm_available, resolved_model


# ── backend wrappers ──────────────────────────────────────────────────────────

def search_documents(query: str, top_k: int = DEFAULT_TOP_K):
    graph, lm_available, lm_model = _get_query_graph()
    state = initial_query_state(
        query,
        lm_studio_available=lm_available,
        top_k=top_k,
        lm_studio_model=lm_model,
    )
    result = graph.invoke(state)
    return (
        result.get("ranked_documents") or [],
        result.get("summary", ""),
        result.get("reformulated_query", ""),
    )


def ingest_documents(folder_path: str, progress_slot) -> dict:
    graph = build_graph(chroma_path=_CHROMA_PATH, embedding_model=_EMBEDDING_MODEL)
    run_state = initial_state([folder_path])
    config = {"recursion_limit": 100_000}
    counts = {"processed": 0, "skipped": 0, "errors": 0}
    for event in graph.stream(run_state, config=config, stream_mode="values"):
        counts["processed"] = len(event.get("indexed_files") or [])
        counts["skipped"] = len(event.get("skipped_files") or [])
        counts["errors"] = len(event.get("error_log") or [])
        progress_slot.caption(
            f"processed {counts['processed']}  •  skipped {counts['skipped']}"
            f"  •  errors {counts['errors']}  —  {event.get('status', '')}"
        )
    return counts


# ── history helpers ───────────────────────────────────────────────────────────

def load_history() -> list:
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text())
    except Exception:
        return []


def save_history(history: list) -> None:
    _HISTORY_FILE.write_text(json.dumps(history, indent=2))


def add_history(query: str, results: list, summary: str, reformulated: str) -> None:
    history = load_history()
    history = [h for h in history if h["query"] != query]
    history.insert(0, {
        "query": query,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": len(results),
        "summary": summary,
        "reformulated": reformulated,
        "ranked_documents": results,
    })
    save_history(history)


# ── result renderer ───────────────────────────────────────────────────────────

def _mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg",
            "jpeg": "image/jpeg", "tiff": "image/tiff"}.get(ext.lstrip("."), "application/octet-stream")


def render_results(display: dict) -> None:
    results = display.get("results") or []
    summary = display.get("summary", "")
    reformulated = display.get("reformulated", "")
    query = display.get("query", "")

    if reformulated and reformulated != query:
        st.caption(f"Search: {reformulated}")

    if not results:
        st.warning("No matching documents found.")
        return

    if summary:
        st.info(summary)

    st.subheader("Top Results")
    for idx, r in enumerate(results, start=1):
        source_name = r.get("source_name", "Unknown")
        score_pct = int(r.get("score", 0) * 100)
        page = r.get("page", "?")
        source_path = r.get("source", "")
        snippet = r.get("document", "")
        if snippet.startswith(_PASSAGE_PREFIX):
            snippet = snippet[len(_PASSAGE_PREFIX):]

        with st.expander(f"#{idx}  {source_name}  ({score_pct}%)  —  p.{page}", expanded=(idx == 1)):
            st.caption(source_path)
            st.text(snippet[:600] + ("…" if len(snippet) > 600 else ""))
            if source_path and Path(source_path).is_file():
                st.download_button(
                    label="Open / Download",
                    data=Path(source_path).read_bytes(),
                    file_name=source_name,
                    mime=_mime(source_path),
                    key=f"dl_{idx}_{id(display)}",
                )


# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Knowledge Catalyst", page_icon="🔎", layout="wide")

_logo = Path(__file__).parent / "modelicon_logo.png"
if _logo.exists():
    st.image(str(_logo), width=180)

for key, default in [("selected_query", ""), ("display_result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Ingest Documents")

folder_path = st.sidebar.text_input("Folder to ingest", placeholder="/path/to/your/docs")
ingest_btn = st.sidebar.button("Ingest")

if ingest_btn:
    if not folder_path.strip():
        st.sidebar.warning("Enter a folder path")
    elif not Path(folder_path).exists():
        st.sidebar.error("Folder not found")
    else:
        progress_slot = st.sidebar.empty()
        with st.spinner("Ingesting documents…"):
            counts = ingest_documents(folder_path, progress_slot)
        progress_slot.empty()
        st.sidebar.success(
            f"Done — {counts['processed']} indexed, {counts['skipped']} skipped, {counts['errors']} errors"
        )

st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Search History")

history = load_history()
for i, item in enumerate(history):
    col1, col2 = st.sidebar.columns([4, 1])
    if col1.button(item["query"], key=f"hist_{i}"):
        st.session_state.selected_query = item["query"]
        # Restore cached output if available; otherwise force a fresh search
        if "ranked_documents" in item:
            st.session_state.display_result = {
                "query": item["query"],
                "results": item["ranked_documents"],
                "summary": item.get("summary", ""),
                "reformulated": item.get("reformulated", item["query"]),
            }
        else:
            st.session_state.display_result = None
        st.rerun()
    if col2.button("🗑", key=f"del_{i}"):
        history.pop(i)
        save_history(history)
        st.rerun()
    st.sidebar.caption(f"{item['time']} • {item.get('results', 0)} results")

# ── main area ─────────────────────────────────────────────────────────────────

st.title("Knowledge Catalyst")
st.markdown("Semantic search over your indexed documents")

with st.form("search_form"):
    query = st.text_input(
        "Search query",
        value=st.session_state.selected_query,
        placeholder="e.g. convergence detection in simulations",
    )
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching…"):
            results, summary, reformulated = search_documents(query)
        display = {"query": query, "results": results, "summary": summary, "reformulated": reformulated}
        st.session_state.display_result = display
        st.session_state.selected_query = ""
        add_history(query, results, summary, reformulated)

if st.session_state.display_result:
    render_results(st.session_state.display_result)
