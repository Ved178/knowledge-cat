"""Interactive REPL for the Knowledge Catalyst query layer."""

from __future__ import annotations

import argparse
import sys

from ingestion_agent.constants import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME
from query_agent.agent import build_query_graph, initial_query_state
from query_agent.constants import (
    DEFAULT_MIN_SCORE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
)

_PASSAGE_PREFIX = "passage: "


def _format_results(state: dict) -> str:
    """Format ranked documents and optional summary for terminal output."""
    lines: list[str] = []
    ranked: list[dict] = state.get("ranked_documents") or []
    summary: str = state.get("summary", "")
    raw_query: str = state.get("raw_query", "")
    reformulated: str = state.get("reformulated_query", "")

    if reformulated and reformulated != raw_query:
        lines.append(f"\n  search: {reformulated}")

    if not ranked:
        if state.get("raw_results"):
            lines.append("\n  No sufficiently relevant documents found for this query.")
        else:
            lines.append("\n  No results found.")
        return "\n".join(lines)

    if summary:
        lines.append(f"\n{summary}\n")

    lines.append("")
    for i, doc in enumerate(ranked, 1):
        score_pct = int(doc.get("score", 0) * 100)
        source_name = doc.get("source_name", "unknown")
        page = doc.get("page", "?")
        snippet: str = doc.get("document", "")
        if snippet.startswith(_PASSAGE_PREFIX):
            snippet = snippet[len(_PASSAGE_PREFIX):]
        snippet_display = snippet[:200].replace("\n", " ")
        if len(snippet) > 200:
            snippet_display += "…"
        lines.append(f"  {i}. {source_name}  ({score_pct}%)")
        lines.append(f'     p.{page} — "{snippet_display}"')
        lines.append("")

    return "\n".join(lines)


def run_repl(
    *,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = "intfloat/e5-large-v2",
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> None:
    print("Loading query layer...", end=" ", flush=True)
    try:
        graph, ollama_available, resolved_model = build_query_graph(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
        )
    except RuntimeError as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    ollama_model = resolved_model
    if ollama_available:
        print(f"ready  (Ollama: {ollama_model})")
    else:
        print("ready  (Ollama not available — plain semantic search)")

    print('Type a query, or "quit" to exit.\n')

    while True:
        try:
            raw_query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw_query:
            continue
        if raw_query.lower() in {"quit", "exit", "q"}:
            break

        state = initial_query_state(
            raw_query,
            ollama_available=ollama_available,
            top_k=top_k,
            min_score=min_score,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
        )
        result = graph.invoke(state)
        print(_format_results(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive semantic search REPL for Knowledge Catalyst."
    )
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default="intfloat/e5-large-v2")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum similarity score (0-1) for a result to be shown; 0 disables the filter",
    )
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_repl(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        min_score=args.min_score,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_url,
    )


if __name__ == "__main__":
    main()
