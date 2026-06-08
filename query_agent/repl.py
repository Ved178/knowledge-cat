"""Interactive REPL for the Knowledge Catalyst query layer."""

from __future__ import annotations

import argparse
import sys

from ingestion_agent.constants import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME
from query_agent.agent import build_query_graph, initial_query_state
from query_agent.constants import DEFAULT_LM_STUDIO_BASE_URL, DEFAULT_LM_STUDIO_MODEL, DEFAULT_TOP_K

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
    lm_studio_model: str = DEFAULT_LM_STUDIO_MODEL,
    lm_studio_base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
) -> None:
    print("Loading query layer...", end=" ", flush=True)
    try:
        graph, lm_studio_available, resolved_model = build_query_graph(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            lm_studio_model=lm_studio_model,
            lm_studio_base_url=lm_studio_base_url,
        )
    except RuntimeError as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    lm_studio_model = resolved_model
    if lm_studio_available:
        print(f"ready  (LM Studio: {lm_studio_model})")
    else:
        print("ready  (LM Studio not available — plain semantic search)")

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
            lm_studio_available=lm_studio_available,
            top_k=top_k,
            lm_studio_model=lm_studio_model,
            lm_studio_base_url=lm_studio_base_url,
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
    parser.add_argument("--lm-studio-model", default=DEFAULT_LM_STUDIO_MODEL)
    parser.add_argument("--lm-studio-url", default=DEFAULT_LM_STUDIO_BASE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_repl(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        lm_studio_model=args.lm_studio_model,
        lm_studio_base_url=args.lm_studio_url,
    )


if __name__ == "__main__":
    main()
