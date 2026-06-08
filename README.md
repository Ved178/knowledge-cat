# Knowledge Catalyst

A local knowledge base pipeline with semantic search. Layer 1 ingests files into a vector store; Layer 2 lets you query them from an interactive REPL powered by LM Studio.

## Architecture

**Layer 1 — Ingestion**

```text
scan_drive → classify_file → extract_text → chunk_text → embed_chunks → store_to_chroma → update_status
```

Crawls supported local files, extracts text, chunks it with the E5 passage prefix, embeds chunks with `intfloat/e5-large-v2`, and stores vectors in a persistent ChromaDB collection named `knowledge_catalyst`.

**Layer 2 — Query**

```text
reformulate_query → retrieve → rank_documents → summarize
```

Interactive REPL that embeds your query, retrieves the top-k most relevant chunks from ChromaDB, ranks results by source file, and generates a 2-3 sentence summary with inline citations via LM Studio. Falls back to plain semantic search when LM Studio is unavailable.

## Quick Setup

Requires Python 3.10+ — download from [python.org/downloads](https://python.org/downloads).

**macOS** — double-click `setup.command` in Finder (right-click → Open the first time to bypass Gatekeeper), or:

```bash
python3 setup.py
```

**Windows** — double-click `setup.bat`, or:

```bat
python setup.py
```

The setup script (~2.5 GB total):
1. Installs Tesseract OCR + Poppler via Homebrew (macOS) / Chocolatey (Windows)
2. Creates a `./env` virtual environment
3. Installs all Python dependencies (CPU-only PyTorch on Windows to avoid CUDA bloat)
4. Downloads `intfloat/e5-large-v2` in safetensors format only (~1.3 GB — skips unused ONNX/OpenVINO variants)

## One-Click Run

After setup, use the launcher scripts to run each part of the pipeline without a terminal.

| macOS | Windows | What it does |
|---|---|---|
| `run_ingest.command` | `run_ingest.bat` | Ingest documents from `./data` into ChromaDB |
| `run_query.command` | `run_query.bat` | Open the interactive search REPL |
| `run_plots.command` | `run_plots.bat` | Generate PCA + t-SNE embedding plots |

**macOS:** double-click any `.command` file (right-click → Open the first time).  
**Windows:** double-click any `.bat` file.

To ingest a different folder, open `run_ingest.command` or `run_ingest.bat` in a text editor and update the `PATHS` variable at the top.

## Supported Files

- Text PDFs: `.pdf`
- Scanned PDFs: `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff`

Unsupported files are skipped and logged to `ingestion_log.db`.

## Ingestion (CLI)

Run from a terminal after activating the venv (`source env/bin/activate` on macOS, `env\Scripts\activate` on Windows):

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2
```

Resume a checkpointed run:

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --resume
```

Force-reindex already-stored files:

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --force-reindex
```

Tune chunking (run with `--force-reindex` when changing these):

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --force-reindex \
  --chunk-size 220 --chunk-overlap 30 --min-chunk-size 40
```

Keep references/bibliography sections (skipped by default):

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --include-reference-chunks
```

Live progress format:

```text
queued=12 processed=3 skipped=0 errors=0 status=Stored 4 chunks for sample.pdf
```

## Query Layer (CLI)

```bash
python query.py --embedding-model models/e5-large-v2
```

The REPL auto-detects LM Studio and selects the best available chat model. With LM Studio running you get query reformulation and a synthesized summary; without it you get plain ranked semantic search.

```text
Loading query layer... ready  (LM Studio: mistralai/devstral-small-2-2512)
Type a query, or "quit" to exit.

> how do simulations detect convergence
  search: convergence detection in simulations monitoring system disturbances power flow

Simulations detect convergence by monitoring system disturbances... [Nguyen_et_al.pdf p.6]

  1. Nguyen_et_al.pdf  (84%)
     p.6 — "…"
  2. Performance_Evaluation.pdf  (83%)
     p.10 — "…"
```

Options:

```bash
python query.py --top-k 10
python query.py --lm-studio-url http://localhost:1234/v1
python query.py --lm-studio-model "mistralai/devstral-small-2-2512"
python query.py --chroma-path ./chroma_db --collection knowledge_catalyst
```

LM Studio notes:
- Embedding models are ignored automatically when selecting a chat model.
- Thinking/reasoning models (Qwen3, DeepSeek-R1, QwQ) are deprioritised in favour of faster chat models; a wall-clock timeout prevents the REPL from hanging if one is selected.

## Embedding Plots (CLI)

```bash
python plot_embeddings.py --chroma-path ./chroma_db
```

Outputs `embedding_plots/pca_embedding_plot.html` and `embedding_plots/tsne_embedding_plot.html`. Each point is one stored chunk; hover shows source file, page, and a text preview.

```bash
python plot_embeddings.py --color-by page
python plot_embeddings.py --methods pca
python plot_embeddings.py --methods tsne --perplexity 10 --tsne-metric cosine
python plot_embeddings.py --max-points 2000
```

## Persistence

| Path | Purpose |
|---|---|
| `./chroma_db` | Persistent vector store |
| `ingestion_checkpoints.sqlite` | LangGraph checkpoints (enables `--resume`) |
| `ingestion_log.db` | Structured log of skipped and failed files |

The pipeline is idempotent. Files already in ChromaDB with the same path and last-modified timestamp are skipped. Modified files are reindexed — old chunks are deleted before upserting new ones.

## Chunking

- Chunk size: `256` approximate whitespace tokens
- Overlap: `32` tokens
- Minimum chunk size: `40` tokens
- Repeated short headers/footers stripped across pages
- Reference/bibliography sections skipped unless `--include-reference-chunks` is passed

Run with `--force-reindex` after changing chunk settings.

## E5 Prefixes

Both prefixes are defined in `ingestion_agent/constants.py`:

- `PASSAGE_PREFIX = "passage: "` — prepended to every stored chunk at ingestion time
- `QUERY_PREFIX = "query: "` — prepended to every user query at search time (`encode_query()` in the query layer)
