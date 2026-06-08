"""Document ranking node — rolls up chunk hits to one result per source file."""

from __future__ import annotations

from typing import Any

from query_agent.constants import DEFAULT_TOP_K


def rank_documents(state: dict[str, Any]) -> dict[str, Any]:
    """Keep the best-scoring chunk per file, then return the top-k files by score."""
    raw_results: list[dict[str, Any]] = list(state.get("raw_results") or [])
    top_k = int(state.get("top_k") or DEFAULT_TOP_K)

    best_by_source: dict[str, dict[str, Any]] = {}
    for row in raw_results:
        source = row["source"]
        if source not in best_by_source or row["score"] > best_by_source[source]["score"]:
            best_by_source[source] = row

    ranked = sorted(best_by_source.values(), key=lambda r: r["score"], reverse=True)[:top_k]
    return {**state, "ranked_documents": ranked}
