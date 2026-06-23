"""LangGraph StateGraph definition for the Knowledge Catalyst query layer."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ingestion_agent.constants import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME
from ingestion_agent.db.chroma_client import get_chroma_collection
from ingestion_agent.models.embedder import get_embedding_model
from query_agent import ollama
from query_agent.constants import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, DEFAULT_TOP_K
from query_agent.nodes.rank import rank_documents
from query_agent.nodes.reformulate import reformulate_query
from query_agent.nodes.retrieve import build_retrieve_node
from query_agent.nodes.summarize import summarize


class QueryState(TypedDict, total=False):
    raw_query: str
    reformulated_query: str
    query_embedding: list[float]
    raw_results: list[dict[str, Any]]
    ranked_documents: list[dict[str, Any]]
    summary: str
    ollama_available: bool
    top_k: int
    ollama_model: str
    ollama_base_url: str


def build_query_graph(
    *,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = "intfloat/e5-large-v2",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> tuple[CompiledStateGraph, bool, str]:
    """Build and compile the query graph; returns (graph, ollama_available, resolved_model)."""
    collection = get_chroma_collection(chroma_path, collection_name)
    model = get_embedding_model(embedding_model)
    ollama_available = ollama.check_available(ollama_base_url)
    if ollama_available and not ollama_model:
        ollama_model = ollama.get_first_model(ollama_base_url) or ollama_model

    graph = StateGraph(QueryState)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("retrieve", build_retrieve_node(collection, model))
    graph.add_node("rank_documents", rank_documents)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("reformulate_query")
    graph.add_edge("reformulate_query", "retrieve")
    graph.add_edge("retrieve", "rank_documents")
    graph.add_edge("rank_documents", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile(), ollama_available, ollama_model


def initial_query_state(
    raw_query: str,
    *,
    ollama_available: bool,
    top_k: int = DEFAULT_TOP_K,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> QueryState:
    """Create a clean initial state for one query invocation."""
    return {
        "raw_query": raw_query,
        "reformulated_query": "",
        "query_embedding": [],
        "raw_results": [],
        "ranked_documents": [],
        "summary": "",
        "ollama_available": ollama_available,
        "top_k": top_k,
        "ollama_model": ollama_model,
        "ollama_base_url": ollama_base_url,
    }
