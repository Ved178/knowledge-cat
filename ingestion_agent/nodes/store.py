"""Chroma storage and SQLite logging nodes."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from ingestion_agent.constants import DEFAULT_LOG_DB_PATH


def ensure_log_schema(log_db_path: str = DEFAULT_LOG_DB_PATH) -> None:
    """Create the ingestion SQLite log schema if it does not already exist."""
    with sqlite3.connect(log_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_log (
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                reason TEXT NOT NULL,
                error_type TEXT NOT NULL,
                retry_count INTEGER NOT NULL
            )
            """
        )
        connection.commit()


def _write_log(
    file_path: str,
    reason: str,
    error_type: str,
    retry_count: int,
    log_db_path: str,
) -> None:
    """Write one structured ingestion log row."""
    ensure_log_schema(log_db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(log_db_path) as connection:
        connection.execute(
            """
            INSERT INTO ingestion_log (timestamp, file_path, reason, error_type, retry_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, file_path, reason, error_type, retry_count),
        )
        connection.commit()


def build_store_to_chroma_node(collection: Collection):
    """Build a store_to_chroma node bound to the local Chroma collection."""

    def store_to_chroma(state: dict[str, Any]) -> dict[str, Any]:
        """Upsert chunks, embeddings, and metadata into ChromaDB."""
        file_path = state.get("current_file", "")
        chunks = list(state.get("chunks") or [])
        embeddings = list(state.get("embeddings") or [])
        metadatas = list(state.get("chunk_metadatas") or [])
        indexed_files = list(state.get("indexed_files") or [])

        try:
            if not chunks:
                raise ValueError(f"No chunks available for {file_path}")
            if len(chunks) != len(embeddings):
                raise ValueError("Chunk and embedding counts do not match")

            last_modified = os.path.getmtime(file_path)
            file_type = state.get("file_type", "")
            enriched_metadatas = [
                {
                    **metadata,
                    "file_last_modified": float(last_modified),
                    "file_type": file_type,
                }
                for metadata in metadatas
            ]
            ids = [f"{file_path}::chunk_{index}" for index in range(len(chunks))]

            collection.delete(where={"source": file_path})
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=enriched_metadatas,
            )

            indexed_files.append(file_path)
            return {
                **state,
                "indexed_files": indexed_files,
                "last_error": "",
                "status": f"Stored {len(chunks)} chunks for {Path(file_path).name}",
            }
        except Exception as exc:
            return {
                **state,
                "last_error": str(exc),
                "status": f"Storage failed for {file_path}: {exc}",
            }

    return store_to_chroma


def build_log_skip_node(log_db_path: str = DEFAULT_LOG_DB_PATH):
    """Build a log_skip node bound to a local SQLite database path."""

    def log_skip(state: dict[str, Any]) -> dict[str, Any]:
        """Log an unsupported or intentionally skipped file."""
        file_path = state.get("current_file", "")
        reason = state.get("skip_reason", "unsupported")
        retry_count = int(state.get("retry_count") or 0)
        skipped_files = list(state.get("skipped_files") or [])
        skipped_files.append(f"{file_path} | {reason}")
        _write_log(file_path, reason, "skip", retry_count, log_db_path)
        return {**state, "skipped_files": skipped_files, "status": f"Skipped {file_path}"}

    return log_skip


def build_log_error_node(log_db_path: str = DEFAULT_LOG_DB_PATH):
    """Build a log_error node bound to a local SQLite database path."""

    def log_error(state: dict[str, Any]) -> dict[str, Any]:
        """Log a file failure after all extraction retries have been exhausted."""
        file_path = state.get("current_file", "")
        reason = state.get("last_error", "unknown error")
        retry_count = int(state.get("retry_count") or 0)
        error_log = list(state.get("error_log") or [])
        entry = {
            "file_path": file_path,
            "reason": reason,
            "error_type": "extraction",
            "retry_count": retry_count,
        }
        error_log.append(entry)
        _write_log(file_path, reason, "extraction", retry_count, log_db_path)
        return {**state, "error_log": error_log, "status": f"Failed {file_path}"}

    return log_error


def retry(state: dict[str, Any]) -> dict[str, Any]:
    """Increment the retry counter before another extraction attempt."""
    retry_count = int(state.get("retry_count") or 0) + 1
    return {**state, "retry_count": retry_count, "status": f"Retry {retry_count}"}
