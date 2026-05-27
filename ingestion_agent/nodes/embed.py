"""Embedding node for passage chunks."""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from ingestion_agent.models.embedder import encode_passages


def build_embed_chunks_node(model: SentenceTransformer):
    """Build an embed_chunks node bound to an already-loaded model."""

    def embed_chunks(state: dict[str, Any]) -> dict[str, Any]:
        """Encode text chunks into normalized embedding vectors."""
        chunks = list(state.get("chunks") or [])
        embeddings = encode_passages(model, chunks) if chunks else []
        return {
            **state,
            "embeddings": embeddings,
            "status": f"Embedded {len(embeddings)} chunks",
        }

    return embed_chunks

