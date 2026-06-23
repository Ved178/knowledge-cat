"""Query reformulation node — rewrites the raw query for better semantic coverage."""

from __future__ import annotations

from typing import Any

from query_agent import ollama
from query_agent.constants import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    REFORMULATION_PROMPT,
)


def reformulate_query(state: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the raw query with an LLM; pass it through when Ollama is unavailable."""
    raw_query = state.get("raw_query", "")
    if not state.get("ollama_available"):
        return {**state, "reformulated_query": raw_query}

    model = state.get("ollama_model", DEFAULT_OLLAMA_MODEL)
    base_url = state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
    try:
        prompt = REFORMULATION_PROMPT.format(query=raw_query)
        reformulated = ollama.generate(prompt, model=model, base_url=base_url, timeout=30.0)
        return {**state, "reformulated_query": reformulated or raw_query}
    except Exception:
        return {**state, "reformulated_query": raw_query}
