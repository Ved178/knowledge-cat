"""Singleton embedding model loader for intfloat/e5-large-v2."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from huggingface_hub.errors import LocalEntryNotFoundError
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/e5-large-v2"


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model for offline use."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path = str(Path(model_name).expanduser()) if _looks_like_path(model_name) else model_name
    try:
        return SentenceTransformer(model_path, local_files_only=True)
    except (OSError, LocalEntryNotFoundError) as exc:
        raise RuntimeError(
            "Embedding model is not available offline. Pre-download "
            "'intfloat/e5-large-v2' into the local Hugging Face cache, or pass "
            "--embedding-model /absolute/path/to/local/e5-large-v2. The ingestion "
            "pipeline does not download models at runtime."
        ) from exc


def _looks_like_path(value: str) -> bool:
    """Return True when a model argument appears to be a local filesystem path."""
    return value.startswith((".", "~", "/")) or "\\" in value or Path(value).exists()


def encode_query(model: SentenceTransformer, query: str) -> list[float]:
    """Encode a query string into a normalized embedding vector with the E5 query prefix."""
    from ingestion_agent.constants import QUERY_PREFIX

    vector = model.encode(
        [QUERY_PREFIX + query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector[0].tolist()


def encode_passages(model: SentenceTransformer, passages: Iterable[str]) -> list[list[float]]:
    """Encode already-prefixed passage texts into normalized embedding vectors."""
    vectors = model.encode(
        list(passages),
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()
