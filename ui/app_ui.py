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
from ingestion_agent.db.chroma_client import get_chroma_collection
from query_agent import ollama
from query_agent.agent import build_query_graph, initial_query_state
from query_agent.constants import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, DEFAULT_TOP_K

_LOCAL_MODEL = str(_REPO_ROOT / "models" / "e5-large-v2")
_EMBEDDING_MODEL = _LOCAL_MODEL if Path(_LOCAL_MODEL).is_dir() else "intfloat/e5-large-v2"
_CHROMA_PATH = str(_REPO_ROOT / "chroma_db")
_HISTORY_FILE = Path(__file__).parent / "history.json"
_PASSAGE_PREFIX = "passage: "


# ── cached query graph ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_query_graph():
    graph, _ollama_available, _resolved_model = build_query_graph(
        chroma_path=_CHROMA_PATH,
        embedding_model=_EMBEDDING_MODEL,
    )
    return graph


# ── indexed-folder stats ──────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_ingested_folder_stats() -> dict[str, dict]:
    """Return per-folder stats derived from Chroma metadata.

    Returns a dict mapping folder path → {"files": [str, …], "latest_mtime": float}.
    """
    try:
        collection = get_chroma_collection(persist_path=_CHROMA_PATH)
        result = collection.get(include=["metadatas"])
    except Exception:
        return {}

    folders: dict[str, dict] = {}
    for meta in result.get("metadatas") or []:
        source = meta.get("source", "")
        if not source:
            continue
        folder = str(Path(source).parent)
        mtime = float(meta.get("file_last_modified") or 0)
        entry = folders.setdefault(folder, {"files": [], "latest_mtime": 0.0})
        if source not in entry["files"]:
            entry["files"].append(source)
        if mtime > entry["latest_mtime"]:
            entry["latest_mtime"] = mtime

    return dict(sorted(folders.items()))


# ── backend wrappers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_ollama_models() -> list[str]:
    return ollama.get_all_models(DEFAULT_OLLAMA_BASE_URL)


def search_documents(query: str, top_k: int = DEFAULT_TOP_K, model: str | None = None):
    graph = _get_query_graph()
    ollama_available = ollama.check_available(DEFAULT_OLLAMA_BASE_URL)
    ollama_model = (
        model or ollama.get_first_model(DEFAULT_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_MODEL
        if ollama_available else DEFAULT_OLLAMA_MODEL
    )
    state = initial_query_state(
        query,
        ollama_available=ollama_available,
        top_k=top_k,
        ollama_model=ollama_model,
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
    counts = {"newly_indexed": 0, "up_to_date": 0, "skipped": 0, "errors": 0}
    for event in graph.stream(run_state, config=config, stream_mode="values"):
        counts["newly_indexed"] = len(event.get("indexed_files") or [])
        counts["up_to_date"] = len(event.get("pre_indexed_files") or [])
        counts["skipped"] = len(event.get("skipped_files") or [])
        counts["errors"] = len(event.get("error_log") or [])
        progress_slot.caption(
            f"new {counts['newly_indexed']}  •  up to date {counts['up_to_date']}"
            f"  •  skipped {counts['skipped']}  •  errors {counts['errors']}"
            f"  —  {event.get('status', '')}"
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

for key, default in [("selected_query", ""), ("display_result", None), ("selected_model", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("LLM")
_available_models = _fetch_ollama_models()
if _available_models:
    _model_options = ["Auto (best available)"] + _available_models
    _current_idx = (
        _model_options.index(st.session_state.selected_model)
        if st.session_state.selected_model in _model_options else 0
    )
    _chosen = st.sidebar.selectbox("Model", _model_options, index=_current_idx)
    st.session_state.selected_model = None if _chosen == "Auto (best available)" else _chosen
else:
    st.sidebar.caption("Ollama not connected — running in semantic search mode")
    st.session_state.selected_model = None

st.sidebar.markdown("---")
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
            f"Done — {counts['newly_indexed']} newly indexed, "
            f"{counts['up_to_date']} already up to date, "
            f"{counts['skipped']} skipped, {counts['errors']} errors"
        )
        get_ingested_folder_stats.clear()

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Indexed Folders")

_folder_stats = get_ingested_folder_stats()
if not _folder_stats:
    st.sidebar.caption("No folders indexed yet")
else:
    _total_files = sum(len(v["files"]) for v in _folder_stats.values())
    st.sidebar.caption(f"{len(_folder_stats)} folder{'s' if len(_folder_stats) != 1 else ''}  •  {_total_files} file{'s' if _total_files != 1 else ''}")
    for _folder, _info in _folder_stats.items():
        _label = Path(_folder).name or _folder
        _count = len(_info["files"])
        _mtime = datetime.fromtimestamp(_info["latest_mtime"]).strftime("%Y-%m-%d") if _info["latest_mtime"] else ""
        with st.sidebar.expander(f"{_label}  ({_count})", expanded=False):
            if _mtime:
                st.caption(f"Last indexed: {_mtime}")
            st.caption(_folder)
            for _f in sorted(_info["files"]):
                st.text(Path(_f).name)

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
    top_k = st.number_input(
        "Results to return (top-k)",
        min_value=1, max_value=50,
        value=DEFAULT_TOP_K, step=1,
    )
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching…"):
            results, summary, reformulated = search_documents(query, top_k=int(top_k), model=st.session_state.selected_model)
        display = {"query": query, "results": results, "summary": summary, "reformulated": reformulated}
        st.session_state.display_result = display
        st.session_state.selected_query = ""
        add_history(query, results, summary, reformulated)

if st.session_state.display_result:
    render_results(st.session_state.display_result)
