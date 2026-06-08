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

## Supported Files

- Text PDFs: `.pdf`
- Scanned PDFs: `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff`

Unsupported files are skipped and logged to `ingestion_log.db`.

## Offline Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install system OCR/PDF tools before running ingestion:

- Tesseract OCR must be installed and available on `PATH`.
- Poppler must be installed and available on `PATH` for `pdf2image`.
- The `intfloat/e5-large-v2` sentence-transformers model must already exist in the local Hugging Face cache. The loader sets offline mode and uses `local_files_only=True`.
- Alternatively, pass `--embedding-model` with the path to a local sentence-transformers copy of the model.

On Windows, install 64-bit builds of Tesseract and Poppler and add their `bin` directories to `PATH`.

## Downloading the Embedding Model

The ingestion pipeline does not download models at runtime. Download `intfloat/e5-large-v2` once while you have network access, then run ingestion offline.

To download the model into the default Hugging Face cache:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"
```

After the model is cached, the default ingestion command can load it offline:

```bash
python ingest.py --paths ./data
```

To keep the model in a project-local folder instead, download it with the Hugging Face CLI:

```bash
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/intfloat-e5-large-v2
```

Then point ingestion at that folder:

```bash
python ingest.py --paths ./data --embedding-model ./models/intfloat-e5-large-v2
```

If the model is not available locally, startup fails with an offline model error. Re-run one of the download commands above on a machine with network access, or copy the downloaded model folder/cache into place.

## Usage

Run a new ingestion job:

```bash
python ingest.py --paths "C:\path\to\folder1" "C:\path\to\shared_drive\projects"
```

Use a local model folder when the Hugging Face cache does not already contain `intfloat/e5-large-v2`:

```bash
python ingest.py --paths ./data --embedding-model "/models/intfloat-e5-large-v2"
```

Resume a checkpointed job:

```bash
python ingest.py --paths "C:\path\to\folder1" "C:\path\to\shared_drive\projects" --resume
```

Optional paths:

```bash
python ingest.py --paths ./data --chroma-path ./chroma_db --log-db ingestion_log.db --checkpoint-db ingestion_checkpoints.sqlite
```

Rebuild existing files with the cleaner chunking defaults:

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --force-reindex
```

Tune chunking if the embedding plots are too crowded:

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --force-reindex --chunk-size 220 --chunk-overlap 30 --min-chunk-size 40
```

By default, ingestion skips references and bibliography sections because they usually add noise to embedding clusters. Keep them when needed:

```bash
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --include-reference-chunks
```

The CLI prints live progress in this format:

```text
queued=12 processed=3 skipped=0 errors=0 status=Stored 4 chunks for sample.pdf
```

## Embedding Plots

Generate interactive PCA and t-SNE charts from the vectors already stored in ChromaDB:

```bash
python plot_embeddings.py --chroma-path ./chroma_db
```

This writes:

- `embedding_plots/pca_embedding_plot.html`
- `embedding_plots/tsne_embedding_plot.html`

Color points by another metadata field:

```bash
python plot_embeddings.py --color-by page
```

Generate only one chart:

```bash
python plot_embeddings.py --methods pca
```

Tune t-SNE when clusters are too compressed or too spread out:

```bash
python plot_embeddings.py --methods tsne --perplexity 10 --tsne-metric cosine
```

For large collections, sample a deterministic subset before plotting:

```bash
python plot_embeddings.py --max-points 2000
```

Each point represents one stored chunk. Hover shows the source file, page, chunk index, file type, and a text preview. PCA is a linear projection, so overlap is expected when the first two principal components explain only a small part of the embedding variance. t-SNE uses cosine distance by default because the E5 embeddings are normalized.

## Query Layer (Layer 2)

Start the interactive REPL after ingestion is complete:

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
python query.py --embedding-model models/e5-large-v2 --top-k 10
python query.py --embedding-model models/e5-large-v2 --lm-studio-url http://localhost:1234/v1
python query.py --embedding-model models/e5-large-v2 --lm-studio-model "mistralai/devstral-small-2-2512"
python query.py --chroma-path ./chroma_db --collection knowledge_catalyst
```

LM Studio notes:
- Embedding models in LM Studio are ignored automatically.
- Thinking/reasoning models (Qwen3, DeepSeek-R1, QwQ) are deprioritized in favour of faster chat models; if one is selected anyway a wall-clock timeout prevents the REPL from hanging.

## Persistence

- Vector store: `./chroma_db`
- LangGraph checkpoints: `ingestion_checkpoints.sqlite`
- Structured ingestion logs: `ingestion_log.db`

The pipeline is idempotent. Files already present in ChromaDB with the same absolute path and last modified timestamp are skipped. Modified files are reindexed with deterministic chunk IDs after old chunks for that source are deleted.

## Chunking

The default chunker is section-aware and uses smaller chunks for cleaner embedding clusters:

- chunk size: `256` approximate whitespace tokens
- overlap: `32` approximate whitespace tokens
- minimum chunk size: `40` approximate whitespace tokens unless a file only has shorter chunks
- repeated short headers and footers are removed across pages
- page number/footer lines are removed
- reference/bibliography sections are skipped unless `--include-reference-chunks` is passed

Run with `--force-reindex` after changing chunk settings so unchanged files are rebuilt in ChromaDB.

## E5 Prefixes

The codebase defines both E5 prefixes in `ingestion_agent/constants.py`:

- `PASSAGE_PREFIX = "passage: "` prepended to every stored chunk at ingestion time
- `QUERY_PREFIX = "query: "` prepended to every user query at search time (used by `encode_query()` in the query layer)
# knowledge-cat
