"""Chroma retrieval node — embeds the query and fetches the nearest chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from ingestion_agent.models.embedder import encode_query
from query_agent.constants import DEFAULT_TOP_K

# Over-fetch chunks so document-level deduplication still leaves enough results.
_OVERFETCH_MULTIPLIER = 4


def build_retrieve_node(collection: Collection, model: SentenceTransformer):
    """Build a retrieve node bound to the given Chroma collection and embedding model."""

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("reformulated_query") or state.get("raw_query", "")
        top_k = int(state.get("top_k") or DEFAULT_TOP_K)

        query_embedding = encode_query(model, query)

        total = collection.count()
        if total == 0:
            return {**state, "raw_results": [], "query_embedding": query_embedding}

        n_results = min(top_k * _OVERFETCH_MULTIPLIER, total)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        rows: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 2.0
            # ChromaDB defaults to L2; embeddings are normalized so L2 ∈ [0, 2].
            # Map to a [0, 1] similarity score: score = 1 - distance / 2.
            score = max(0.0, 1.0 - distance / 2.0)
            rows.append(
                {
                    "id": chunk_id,
                    "document": documents[i] if i < len(documents) else "",
                    "source": metadata.get("source", ""),
                    "source_name": Path(metadata.get("source", "unknown")).name,
                    "page": metadata.get("page", ""),
                    "chunk_index": metadata.get("chunk_index", ""),
                    "file_type": metadata.get("file_type", ""),
                    "distance": distance,
                    "score": score,
                }
            )

        return {**state, "raw_results": rows, "query_embedding": query_embedding}

    return retrieve
