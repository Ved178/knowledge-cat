"""LangGraph StateGraph definition for the Knowledge Catalyst ingestion layer."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ingestion_agent.constants import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_LOG_DB_PATH
from ingestion_agent.db.chroma_client import get_chroma_collection
from ingestion_agent.models.embedder import get_embedding_model
from ingestion_agent.nodes.chunk import chunk_text
from ingestion_agent.nodes.classify import classify_file
from ingestion_agent.nodes.embed import build_embed_chunks_node
from ingestion_agent.nodes.extract import extract_text
from ingestion_agent.nodes.scan import build_scan_drive_node
from ingestion_agent.nodes.store import (
    build_log_error_node,
    build_log_skip_node,
    build_store_to_chroma_node,
    ensure_log_schema,
    retry,
)


class IngestionState(TypedDict, total=False):
    """Mutable state carried through the ingestion graph."""

    file_queue: list[str]
    current_file: str
    extracted_text: str
    chunks: list[str]
    embeddings: list[list[float]]
    retry_count: int
    skipped_files: list[str]
    error_log: list[dict[str, Any]]
    indexed_files: list[str]
    status: str
    root_paths: list[str]
    file_type: str
    chunk_metadatas: list[dict[str, Any]]
    scanned_roots: bool
    skip_reason: str
    last_error: str


def update_status(state: IngestionState) -> IngestionState:
    """Update a compact status summary after a terminal per-file node."""
    queued = len(state.get("file_queue") or [])
    processed = len(state.get("indexed_files") or [])
    skipped = len(state.get("skipped_files") or [])
    errors = len(state.get("error_log") or [])
    return {
        **state,
        "status": (
            f"queued={queued} processed={processed} skipped={skipped} errors={errors}"
        ),
    }


def route_after_scan(state: IngestionState) -> str:
    """Route to classification when a file is selected, otherwise finish."""
    return "classify_file" if state.get("current_file") else END


def route_after_classify(state: IngestionState) -> str:
    """Route unsupported files to skip logging and supported files to extraction."""
    return "log_skip" if state.get("file_type") == "unsupported" else "extract_text"


def route_after_extract(state: IngestionState) -> str:
    """Route successful extraction to chunking, failures to retry or error logging."""
    if not state.get("last_error"):
        return "chunk_text"
    if int(state.get("retry_count") or 0) < 2:
        return "retry"
    return "log_error"


def route_after_store(state: IngestionState) -> str:
    """Route storage failures to error logging and successful writes to status updates."""
    return "log_error" if state.get("last_error") else "update_status"


def build_graph(
    *,
    checkpointer: Any | None = None,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    log_db_path: str = DEFAULT_LOG_DB_PATH,
    embedding_model: str = "intfloat/e5-large-v2",
) -> CompiledStateGraph:
    """Build and compile the LangGraph ingestion graph."""
    ensure_log_schema(log_db_path)
    collection = get_chroma_collection(chroma_path, collection_name)
    model = get_embedding_model(embedding_model)

    graph = StateGraph(IngestionState)
    graph.add_node("scan_drive", build_scan_drive_node(collection))
    graph.add_node("classify_file", classify_file)
    graph.add_node("extract_text", extract_text)
    graph.add_node("chunk_text", chunk_text)
    graph.add_node("embed_chunks", build_embed_chunks_node(model))
    graph.add_node("store_to_chroma", build_store_to_chroma_node(collection))
    graph.add_node("update_status", update_status)
    graph.add_node("log_skip", build_log_skip_node(log_db_path))
    graph.add_node("log_error", build_log_error_node(log_db_path))
    graph.add_node("retry", retry)

    graph.set_entry_point("scan_drive")
    graph.add_conditional_edges("scan_drive", route_after_scan)
    graph.add_conditional_edges("classify_file", route_after_classify)
    graph.add_conditional_edges("extract_text", route_after_extract)
    graph.add_edge("retry", "extract_text")
    graph.add_edge("chunk_text", "embed_chunks")
    graph.add_edge("embed_chunks", "store_to_chroma")
    graph.add_conditional_edges("store_to_chroma", route_after_store)
    graph.add_edge("log_skip", "update_status")
    graph.add_edge("log_error", "update_status")
    graph.add_edge("update_status", "scan_drive")

    return graph.compile(checkpointer=checkpointer)


def initial_state(root_paths: list[str]) -> IngestionState:
    """Create a clean initial graph state for a new ingestion run."""
    return {
        "root_paths": root_paths,
        "file_queue": [],
        "current_file": "",
        "extracted_text": "",
        "chunks": [],
        "chunk_metadatas": [],
        "embeddings": [],
        "retry_count": 0,
        "skipped_files": [],
        "error_log": [],
        "indexed_files": [],
        "status": "initialized",
        "scanned_roots": False,
        "skip_reason": "",
        "last_error": "",
    }
