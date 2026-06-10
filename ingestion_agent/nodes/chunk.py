"""Clean, section-aware text chunking node."""

from __future__ import annotations

import math
import re
from typing import Any

from ingestion_agent.constants import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    MIN_CHUNK_TOKENS,
    PASSAGE_PREFIX,
)

PAGE_PATTERN = re.compile(r"\[PAGE\s+(\d+)\]\s*", re.IGNORECASE)
REFERENCE_HEADING_PATTERN = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE,
)
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*((\d+(\.\d+)*)\s+)?[A-Z][A-Za-z0-9,()\- /]{2,90}\s*$"
)
PAGE_FOOTER_PATTERN = re.compile(
    r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$|^\s*\d+\s*\|\s*.+$",
    re.IGNORECASE,
)
INLINE_PAGE_HEADER_PATTERN = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?[\s\-:|]+\|?\s*$")


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


def _normalize_line(line: str) -> str:
    """Normalize whitespace in one extracted text line."""
    return re.sub(r"\s+", " ", line).strip()


def _line_key(line: str) -> str:
    """Return a comparison key for repeated boilerplate detection."""
    return re.sub(r"\d+", "#", line.lower()).strip()


def _recurring_line_keys(page_lines: list[tuple[int, list[str]]]) -> set[str]:
    """Identify short lines that recur across pages and are likely headers or footers."""
    line_pages: dict[str, set[int]] = {}
    for page_number, lines in page_lines:
        for line in lines:
            key = _line_key(line)
            if 4 <= len(key) <= 100:
                line_pages.setdefault(key, set()).add(page_number)

    page_count = max(1, len(page_lines))
    threshold = max(2, math.ceil(page_count * 0.25))
    return {key for key, pages in line_pages.items() if len(pages) >= threshold}


def _clean_page_lines(page_lines: list[tuple[int, list[str]]]) -> list[tuple[int, list[str]]]:
    """Remove repeated headers, footers, page numbers, and empty table separators."""
    recurring_keys = _recurring_line_keys(page_lines)
    cleaned_pages: list[tuple[int, list[str]]] = []

    for page_number, lines in page_lines:
        cleaned_lines: list[str] = []
        for line in lines:
            normalized = _normalize_line(line)
            if not normalized:
                continue
            if TABLE_SEPARATOR_PATTERN.match(normalized):
                continue
            if PAGE_FOOTER_PATTERN.match(normalized):
                continue
            if INLINE_PAGE_HEADER_PATTERN.search(normalized) and len(normalized.split()) <= 24:
                continue
            key = _line_key(normalized)
            word_count = len(normalized.split())
            if key in recurring_keys and word_count <= 12:
                continue
            cleaned_lines.append(normalized)
        cleaned_pages.append((page_number, cleaned_lines))

    return cleaned_pages


def _prepare_pages(text: str) -> list[tuple[int, list[str]]]:
    """Split text into pages and cleaned line lists."""
    page_lines = [
        (page_number, [line for line in page_text.splitlines()])
        for page_number, page_text in _page_sections(text)
    ]
    return _clean_page_lines(page_lines)


def _is_heading(line: str) -> bool:
    """Return True when a line looks like a section heading."""
    if len(line.split()) > 12:
        return False
    if REFERENCE_HEADING_PATTERN.match(line):
        return True
    if SECTION_HEADING_PATTERN.match(line) and not line.endswith("."):
        return True
    return line.isupper() and 1 <= len(line.split()) <= 10


def _paragraph_blocks(lines: list[str], skip_references: bool) -> list[str]:
    """Group cleaned lines into paragraph and heading-aware text blocks."""
    blocks: list[str] = []
    current: list[str] = []
    in_references = False

    for line in lines:
        if REFERENCE_HEADING_PATTERN.match(line):
            in_references = skip_references
            if current:
                blocks.append(" ".join(current))
                current = []
            if not skip_references:
                blocks.append(line)
            continue
        if in_references:
            continue

        if _is_heading(line):
            if current:
                blocks.append(" ".join(current))
                current = []
            blocks.append(line)
            continue

        if line.startswith("|") and line.endswith("|"):
            if current:
                blocks.append(" ".join(current))
                current = []
            blocks.append(line)
            continue

        current.append(line)

    if current:
        blocks.append(" ".join(current))
    return [block for block in blocks if block.strip()]


def _window_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Split oversized text into overlapping word windows."""
    if len(words) <= chunk_size:
        return [" ".join(words)]

    windows: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            continue
        windows.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return windows


def _page_chunks(
    page_number: int,
    lines: list[str],
    *,
    chunk_size: int,
    overlap: int,
    skip_references: bool,
) -> list[tuple[int, str]]:
    """Create clean chunks for one page while preserving page metadata."""
    blocks = _paragraph_blocks(lines, skip_references)
    chunks: list[tuple[int, str]] = []
    current_words: list[str] = []

    for block in blocks:
        block_words = block.split()
        if not block_words:
            continue

        if len(block_words) > chunk_size:
            if current_words:
                chunks.append((page_number, " ".join(current_words)))
                current_words = []
            for window in _window_words(block_words, chunk_size, overlap):
                chunks.append((page_number, window))
            continue

        if current_words and len(current_words) + len(block_words) > chunk_size:
            chunks.append((page_number, " ".join(current_words)))
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(block_words)

    if current_words:
        chunks.append((page_number, " ".join(current_words)))

    return chunks


def _tabular_candidate_chunks(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[int, str]]:
    """Split tabular content into overlapping word windows without prose cleaning.

    Bypasses _prepare_pages so that numeric-leading rows (e.g. spreadsheet data)
    are never dropped by the PDF-tuned PAGE_FOOTER_PATTERN.
    """
    words = text.split()
    if not words:
        return []
    windows = _window_words(words, chunk_size, overlap)
    return [(1, w) for w in windows]


def chunk_text(state: dict[str, Any]) -> dict[str, Any]:
    """Create overlapping passage-prefixed chunks with per-chunk metadata."""
    file_path = state.get("current_file", "")
    extracted_text = state.get("extracted_text", "")
    file_type = state.get("file_type", "")
    chunk_size = int(state.get("chunk_size_tokens") or CHUNK_SIZE_TOKENS)
    overlap = int(state.get("chunk_overlap_tokens") or CHUNK_OVERLAP_TOKENS)
    min_chunk_size = int(state.get("min_chunk_tokens") or MIN_CHUNK_TOKENS)
    skip_references = bool(state.get("skip_reference_chunks", True))
    chunks: list[str] = []
    metadatas: list[dict[str, Any]] = []
    candidate_chunks: list[tuple[int, str]] = []

    if chunk_size <= 0:
        raise ValueError("chunk_size_tokens must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap_tokens must be non-negative and smaller than chunk_size_tokens")
    if min_chunk_size < 0:
        raise ValueError("min_chunk_tokens must be non-negative")

    if file_type == "office_tabular":
        candidate_chunks = _tabular_candidate_chunks(extracted_text, chunk_size, overlap)
    else:
        for page_number, lines in _prepare_pages(extracted_text):
            candidate_chunks.extend(_page_chunks(
                page_number,
                lines,
                chunk_size=chunk_size,
                overlap=overlap,
                skip_references=skip_references,
            ))

    filtered_chunks = [
        (page_number, chunk_body)
        for page_number, chunk_body in candidate_chunks
        if len(chunk_body.split()) >= min_chunk_size
    ]
    if not filtered_chunks and candidate_chunks:
        filtered_chunks = candidate_chunks

    for page_number, chunk_body in filtered_chunks:
        chunk_index = len(chunks)
        chunks.append(PASSAGE_PREFIX + chunk_body)
        metadatas.append(
            {
                "source": file_path,
                "page": page_number,
                "chunk_index": chunk_index,
                "chunk_word_count": len(chunk_body.split()),
            }
        )

    return {
        **state,
        "chunks": chunks,
        "chunk_metadatas": metadatas,
        "status": f"Chunked {file_path} into {len(chunks)} chunks",
    }
