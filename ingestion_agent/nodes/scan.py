"""Drive scanning node for the ingestion graph."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from ingestion_agent.constants import SUPPORTED_EXTENSIONS


def _already_indexed(collection: Collection, file_path: str, last_modified: float) -> bool:
    """Return True when Chroma already contains chunks for this file version."""
    try:
        results = collection.get(
            where={"source": file_path},
            include=["metadatas"],
            limit=1,
        )
    except Exception:
        return False

    for metadata in results.get("metadatas") or []:
        stored_mtime = metadata.get("file_last_modified")
        if stored_mtime is not None and float(stored_mtime) == float(last_modified):
            return True
    return False


def build_scan_drive_node(collection: Collection):
    """Build a scan_drive node bound to the local Chroma collection."""

    def scan_drive(state: dict[str, Any]) -> dict[str, Any]:
        """Populate the file queue once, then select the next file for processing."""
        file_queue = list(state.get("file_queue") or [])
        scanned_roots = bool(state.get("scanned_roots"))
        indexed_files = list(state.get("indexed_files") or [])
        skipped_files = list(state.get("skipped_files") or [])

        if not scanned_roots:
            roots = [Path(path).expanduser().resolve() for path in state.get("root_paths", [])]
            discovered: list[str] = []

            for root in roots:
                if not root.exists():
                    skipped_files.append(f"{root} | path does not exist")
                    continue
                if root.is_file():
                    candidates = [root]
                else:
                    candidates = [path for path in root.rglob("*") if path.is_file()]

                for path in candidates:
                    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    file_path = str(path.resolve())
                    last_modified = os.path.getmtime(file_path)
                    if _already_indexed(collection, file_path, last_modified):
                        indexed_files.append(file_path)
                        continue
                    discovered.append(file_path)

            file_queue = discovered
            scanned_roots = True

        if file_queue:
            current_file = file_queue.pop(0)
            status = f"Processing {current_file}"
            retry_count = 0
        else:
            current_file = ""
            status = "complete"
            retry_count = int(state.get("retry_count") or 0)

        return {
            **state,
            "file_queue": file_queue,
            "current_file": current_file,
            "extracted_text": "",
            "chunks": [],
            "chunk_metadatas": [],
            "embeddings": [],
            "retry_count": retry_count,
            "skipped_files": skipped_files,
            "indexed_files": indexed_files,
            "status": status,
            "scanned_roots": scanned_roots,
            "last_error": "",
        }

    return scan_drive

