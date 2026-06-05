"""Command-line entry point for the Knowledge Catalyst ingestion layer."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from ingestion_agent.agent import build_graph, initial_state
from ingestion_agent.constants import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DEFAULT_CHECKPOINT_DB_PATH,
    MIN_CHUNK_TOKENS,
)


def _progress_line(state: dict[str, Any]) -> str:
    """Format a live progress summary from graph state."""
    queued = len(state.get("file_queue") or [])
    processed = len(state.get("indexed_files") or [])
    skipped = len(state.get("skipped_files") or [])
    errors = len(state.get("error_log") or [])
    status = state.get("status", "")
    return (
        f"queued={queued} processed={processed} skipped={skipped} "
        f"errors={errors} status={status}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingestion."""
    parser = argparse.ArgumentParser(description="Run local Knowledge Catalyst ingestion.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more folders or files to scan recursively.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume using an existing LangGraph checkpoint for the same thread id.",
    )
    parser.add_argument(
        "--thread-id",
        default="knowledge-catalyst-ingestion",
        help="LangGraph checkpoint thread id.",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=DEFAULT_CHECKPOINT_DB_PATH,
        help="SQLite checkpoint database path.",
    )
    parser.add_argument(
        "--chroma-path",
        default="./chroma_db",
        help="Local persistent ChromaDB directory.",
    )
    parser.add_argument(
        "--log-db",
        default="ingestion_log.db",
        help="SQLite ingestion log database path.",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/e5-large-v2",
        help=(
            "Cached Hugging Face model id or local sentence-transformers model path. "
            "The model is loaded offline only."
        ),
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Reprocess matching files even when Chroma already has the same mtime.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_TOKENS,
        help="Approximate chunk size in whitespace tokens.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP_TOKENS,
        help="Approximate overlap in whitespace tokens.",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=MIN_CHUNK_TOKENS,
        help="Drop chunks below this approximate token count unless they are the only chunks.",
    )
    parser.add_argument(
        "--include-reference-chunks",
        action="store_true",
        help="Keep references and bibliography sections instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ingestion graph and print progress until completion."""
    args = parse_args()
    root_paths = [str(Path(path).expanduser().resolve()) for path in args.paths]
    config = {
        "configurable": {"thread_id": args.thread_id},
        "recursion_limit": 100000,
    }

    checkpoint_connection = sqlite3.connect(args.checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_connection)
    try:
        graph = build_graph(
            checkpointer=checkpointer,
            chroma_path=args.chroma_path,
            log_db_path=args.log_db,
            embedding_model=args.embedding_model,
        )
    except RuntimeError as exc:
        checkpoint_connection.close()
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    saved = graph.get_state(config) if args.resume else None
    if args.resume and saved and saved.values:
        run_state = {
            **saved.values,
            "root_paths": root_paths,
            "force_reindex": args.force_reindex,
            "chunk_size_tokens": args.chunk_size,
            "chunk_overlap_tokens": args.chunk_overlap,
            "min_chunk_tokens": args.min_chunk_size,
            "skip_reference_chunks": not args.include_reference_chunks,
        }
    else:
        run_state = initial_state(
            root_paths,
            force_reindex=args.force_reindex,
            chunk_size_tokens=args.chunk_size,
            chunk_overlap_tokens=args.chunk_overlap,
            min_chunk_tokens=args.min_chunk_size,
            skip_reference_chunks=not args.include_reference_chunks,
        )

    last_line = ""
    try:
        for event in graph.stream(run_state, config=config, stream_mode="values"):
            line = _progress_line(event)
            if line != last_line:
                print(line, flush=True)
                last_line = line
    finally:
        checkpoint_connection.close()


if __name__ == "__main__":
    main()
