"""Sliding-window text chunking node."""

from __future__ import annotations

import re
from typing import Any

from ingestion_agent.constants import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, PASSAGE_PREFIX

PAGE_PATTERN = re.compile(r"\[PAGE\s+(\d+)\]\s*", re.IGNORECASE)


def _page_sections(text: str) -> list[tuple[int, str]]:
    """Split extracted text into page-numbered sections."""
    matches = list(PAGE_PATTERN.finditer(text))
    if not matches:
        return [(1, text)]

    sections: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((page_number, text[start:end].strip()))
    return sections


def chunk_text(state: dict[str, Any]) -> dict[str, Any]:
    """Create overlapping passage-prefixed chunks with per-chunk metadata."""
    file_path = state.get("current_file", "")
    extracted_text = state.get("extracted_text", "")
    chunks: list[str] = []
    metadatas: list[dict[str, Any]] = []
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    for page_number, page_text in _page_sections(extracted_text):
        words = page_text.split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + CHUNK_SIZE_TOKENS]
            if not window:
                continue
            chunk_index = len(chunks)
            chunks.append(PASSAGE_PREFIX + " ".join(window))
            metadatas.append(
                {
                    "source": file_path,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
            )
            if start + CHUNK_SIZE_TOKENS >= len(words):
                break

    return {
        **state,
        "chunks": chunks,
        "chunk_metadatas": metadatas,
        "status": f"Chunked {file_path} into {len(chunks)} chunks",
    }

