"""Summary generation node — synthesizes a short answer from the ranked chunks."""

from __future__ import annotations

from typing import Any

from query_agent import ollama
from query_agent.constants import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    SUMMARY_PROMPT,
)

_MAX_EXCERPT_CHARS = 400
_MAX_DOCS_FOR_SUMMARY = 5


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a 2-3 sentence answer with inline citations from the top ranked chunks."""
    if not state.get("ollama_available"):
        return {**state, "summary": ""}

    ranked: list[dict[str, Any]] = list(state.get("ranked_documents") or [])
    raw_query = state.get("raw_query", "")
    model = state.get("ollama_model", DEFAULT_OLLAMA_MODEL)
    base_url = state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)

    if not ranked:
        return {**state, "summary": ""}

    excerpts = "\n\n".join(
        f"[{row['source_name']} p.{row['page']}]: {row['document'][:_MAX_EXCERPT_CHARS]}"
        for row in ranked[:_MAX_DOCS_FOR_SUMMARY]
    )

    try:
        prompt = SUMMARY_PROMPT.format(query=raw_query, excerpts=excerpts)
        summary = ollama.generate(prompt, model=model, base_url=base_url, timeout=180.0)
        return {**state, "summary": summary}
    except Exception:
        return {**state, "summary": ""}
