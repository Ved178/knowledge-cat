"""Generate 2D PCA and t-SNE plots from stored ChromaDB embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

from ingestion_agent.constants import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME
from ingestion_agent.db.chroma_client import get_chroma_collection


def _require_plot_dependencies() -> None:
    """Import plotting dependencies and raise a clear setup error if missing."""
    global PCA, TSNE, np, pd, px

    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "Embedding plotting requires numpy, pandas, plotly, and scikit-learn. "
            "Install them with: pip install -r requirements.txt"
        ) from exc


def _batched_collection_rows(
    *,
    chroma_path: str,
    collection_name: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Read ids, embeddings, documents, and metadata from Chroma in batches."""
    collection = get_chroma_collection(chroma_path, collection_name)
    total = collection.count()
    rows: list[dict[str, Any]] = []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        ids = batch.get("ids") or []
        embeddings = batch.get("embeddings")
        if embeddings is None:
            embeddings = []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []

        for index, embedding in enumerate(embeddings):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document = documents[index] if index < len(documents) and documents[index] else ""
            rows.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "embedding": embedding,
                    "document": document,
                    "source": metadata.get("source", "unknown"),
                    "source_name": Path(metadata.get("source", "unknown")).name,
                    "page": metadata.get("page", ""),
                    "chunk_index": metadata.get("chunk_index", ""),
                    "file_type": metadata.get("file_type", ""),
                }
            )

    return rows


def _sample_rows(rows: list[dict[str, Any]], max_points: int | None) -> list[dict[str, Any]]:
    """Return a deterministic random sample when the collection is larger than requested."""
    if max_points is None or len(rows) <= max_points:
        return rows
    rng = np.random.default_rng(seed=42)
    indices = sorted(rng.choice(len(rows), size=max_points, replace=False).tolist())
    return [rows[index] for index in indices]


def _make_dataframe(rows: list[dict[str, Any]], coordinates: np.ndarray) -> pd.DataFrame:
    """Build the plotting dataframe from Chroma rows and 2D coordinates."""
    records: list[dict[str, Any]] = []
    for row, point in zip(rows, coordinates):
        document = str(row["document"])
        records.append(
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "id": row["id"],
                "source": row["source"],
                "source_name": row["source_name"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "file_type": row["file_type"],
                "preview": document[:500],
            }
        )
    return pd.DataFrame.from_records(records)


def _validate_embeddings(rows: list[dict[str, Any]]) -> np.ndarray:
    """Convert stored embeddings to a 2D numpy matrix and validate minimum size."""
    if len(rows) < 2:
        raise ValueError("At least 2 embedded chunks are required to make a 2D plot.")
    embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Stored embeddings are not a 2D matrix.")
    return embeddings


def _pca_coordinates(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to two dimensions with PCA."""
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def _tsne_coordinates(embeddings: np.ndarray, perplexity: float | None) -> np.ndarray:
    """Project embeddings to two dimensions with t-SNE."""
    sample_count = embeddings.shape[0]
    if sample_count < 3:
        raise ValueError("At least 3 embedded chunks are required for t-SNE.")

    effective_perplexity = perplexity
    if effective_perplexity is None:
        effective_perplexity = min(30.0, max(2.0, float((sample_count - 1) // 3)))
    if effective_perplexity >= sample_count:
        effective_perplexity = max(1.0, float(sample_count - 1))

    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(embeddings)


def _write_plot(
    *,
    dataframe: pd.DataFrame,
    title: str,
    output_path: Path,
    color_by: str,
) -> None:
    """Write an interactive Plotly scatter chart to an HTML file."""
    fig = px.scatter(
        dataframe,
        x="x",
        y="y",
        color=color_by,
        symbol="file_type" if "file_type" in dataframe else None,
        hover_data={
            "source": True,
            "page": True,
            "chunk_index": True,
            "file_type": True,
            "preview": True,
            "x": False,
            "y": False,
        },
        title=title,
        template="plotly_white",
        height=800,
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.78})
    fig.update_layout(
        legend_title_text=color_by,
        margin={"l": 40, "r": 40, "t": 70, "b": 40},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def generate_embedding_plots(
    *,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    output_dir: str = "embedding_plots",
    methods: Iterable[str] = ("pca", "tsne"),
    color_by: str = "source_name",
    max_points: int | None = None,
    batch_size: int = 1000,
    perplexity: float | None = None,
) -> list[Path]:
    """Generate requested PCA and t-SNE embedding plots from the Chroma collection."""
    _require_plot_dependencies()
    rows = _batched_collection_rows(
        chroma_path=chroma_path,
        collection_name=collection_name,
        batch_size=batch_size,
    )
    sampled_rows = _sample_rows(rows, max_points)
    embeddings = _validate_embeddings(sampled_rows)
    output_root = Path(output_dir)
    written: list[Path] = []

    for method in methods:
        normalized_method = method.lower()
        if normalized_method == "pca":
            coordinates = _pca_coordinates(embeddings)
            title = "Knowledge Catalyst Embeddings - PCA"
        elif normalized_method == "tsne":
            coordinates = _tsne_coordinates(embeddings, perplexity)
            title = "Knowledge Catalyst Embeddings - t-SNE"
        else:
            raise ValueError(f"Unsupported plot method: {method}")

        dataframe = _make_dataframe(sampled_rows, coordinates)
        output_path = output_root / f"{normalized_method}_embedding_plot.html"
        _write_plot(
            dataframe=dataframe,
            title=title,
            output_path=output_path,
            color_by=color_by,
        )
        written.append(output_path)

    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for embedding plot generation."""
    parser = argparse.ArgumentParser(description="Plot Chroma embeddings with PCA and t-SNE.")
    parser.add_argument(
        "--chroma-path",
        default=DEFAULT_CHROMA_PATH,
        help="Local persistent ChromaDB directory.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_plots",
        help="Directory where HTML plots will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pca", "tsne"],
        default=["pca", "tsne"],
        help="One or more projection methods to generate.",
    )
    parser.add_argument(
        "--color-by",
        default="source_name",
        choices=["source_name", "source", "page", "chunk_index", "file_type"],
        help="Metadata field used to color points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional deterministic sample size for large collections.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of Chroma rows to read per batch.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="Optional t-SNE perplexity. Must be less than the number of points.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the embedding plotter from the command line."""
    args = parse_args()
    try:
        written = generate_embedding_plots(
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            output_dir=args.output_dir,
            methods=args.methods,
            color_by=args.color_by,
            max_points=args.max_points,
            batch_size=args.batch_size,
            perplexity=args.perplexity,
        )
    except RuntimeError as exc:
        print(f"Plotting failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
