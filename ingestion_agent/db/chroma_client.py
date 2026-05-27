"""ChromaDB persistent client setup for Knowledge Catalyst."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from ingestion_agent.constants import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME


def get_chroma_collection(
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Return the persistent local ChromaDB collection used by ingestion."""
    Path(persist_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(name=collection_name)

