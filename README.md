# Knowledge Catalyst

A local knowledge base pipeline with semantic search. Ingests files into a ChromaDB vector store and lets you query them via a Streamlit UI or interactive REPL, with LLM-powered summarisation through Ollama.

## Setup (Docker)

The Docker setup bundles the app and an Ollama instance — no local Python or system dependencies required.

**1. Set up your environment file**

```bash
cp .env.example .env
```

Open `.env` and optionally add a Hugging Face token. A token bypasses anonymous rate limits and speeds up the embedding model download (~1.3 GB). Get a free read-only token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Leave `HF_TOKEN=` blank to download anonymously.

**2. Build and start**

```bash
docker compose up --build
```

The first build downloads the embedding model and all dependencies (~3–4 GB total). Subsequent starts are fast.

**3. Open the UI**

Go to **http://localhost:8501**.

**4. Pull a language model into Ollama** (one-time per model, persists across restarts)

```bash
docker compose exec ollama ollama pull llama3.2
```

Any model you pull appears in the UI's model selector immediately. The app falls back to plain semantic search if no model is loaded yet.

### CLI commands inside Docker

```bash
# Ingest documents from ./data
docker compose run --rm app python ingest.py \
  --paths /app/data \
  --embedding-model /app/models/e5-large-v2

# Interactive query REPL
docker compose run --rm -it app python query.py \
  --embedding-model /app/models/e5-large-v2

# Pull a different model
docker compose exec ollama ollama pull mistral
# Then select it in the UI sidebar, or:
docker compose run --rm -it app python query.py \
  --embedding-model /app/models/e5-large-v2 --ollama-model mistral
```

---

## Supported Files

| Format | Extensions | Notes |
|---|---|---|
| Text PDFs | `.pdf` | Native text extraction via pdfplumber |
| Scanned PDFs | `.pdf` | Auto-detected; OCR via Tesseract |
| Images | `.png` `.jpg` `.jpeg` `.tiff` | OCR via Tesseract |
| Word documents | `.docx` `.doc` | `.doc` requires LibreOffice |
| PowerPoint | `.pptx` `.ppt` | Per-slide page markers; `.ppt` requires LibreOffice |
| Excel / CSV | `.xlsx` `.xls` `.csv` | `.xls` requires LibreOffice |

Legacy formats (`.doc`, `.ppt`, `.xls`) need LibreOffice (`soffice`), which is **not** included in the Docker image — they are logged as failed ingestions and skipped.

Unsupported files are skipped and logged to `ingestion_log.db`.

---

## Architecture

**Layer 1 — Ingestion**

```text
scan_drive → classify_file → extract_text → chunk_text → embed_chunks → store_to_chroma → update_status
```

Crawls local files, extracts text, chunks with the E5 passage prefix, embeds with `intfloat/e5-large-v2`, and stores vectors in a ChromaDB collection named `knowledge_catalyst`.

**Layer 2 — Query**

```text
reformulate_query → retrieve → rank_documents → summarize
```

Embeds the query, retrieves top-k chunks, ranks by source file, and generates a 2-3 sentence summary with inline citations via Ollama. Falls back to plain semantic search when Ollama has no model loaded.

---

## Ingestion CLI Reference

All commands run inside the app container via `docker compose run`:

```bash
# Resume a checkpointed run
docker compose run --rm app python ingest.py --paths /app/data \
  --embedding-model /app/models/e5-large-v2 --resume

# Force-reindex already-stored files
docker compose run --rm app python ingest.py --paths /app/data \
  --embedding-model /app/models/e5-large-v2 --force-reindex

# Tune chunking (always pair with --force-reindex)
docker compose run --rm app python ingest.py --paths /app/data \
  --embedding-model /app/models/e5-large-v2 --force-reindex \
  --chunk-size 220 --chunk-overlap 30 --min-chunk-size 40

# Keep references/bibliography sections (skipped by default)
docker compose run --rm app python ingest.py --paths /app/data \
  --embedding-model /app/models/e5-large-v2 --include-reference-chunks
```

Live progress:

```text
queued=12 processed=3 skipped=0 errors=0 status=Stored 4 chunks for sample.pdf
```

## Query CLI Reference

```bash
docker compose run --rm -it app python query.py \
  --embedding-model /app/models/e5-large-v2 --top-k 10
docker compose run --rm -it app python query.py \
  --embedding-model /app/models/e5-large-v2 --ollama-model llama3.2
docker compose run --rm -it app python query.py \
  --embedding-model /app/models/e5-large-v2 \
  --chroma-path ./chroma_db --collection knowledge_catalyst
```

The Ollama endpoint defaults to the bundled `ollama` container (`OLLAMA_BASE_URL=http://ollama:11434/v1`); override with `--ollama-url` only if you run Ollama elsewhere.

Example session:

```text
Loading query layer... ready  (Ollama: llama3.2)
Type a query, or "quit" to exit.

> how do simulations detect convergence
  search: convergence detection simulations monitoring disturbances power flow

Simulations detect convergence by monitoring system disturbances... [Nguyen_et_al.pdf p.6]

  1. Nguyen_et_al.pdf  (84%)
     p.6 — "…"
  2. Performance_Evaluation.pdf  (83%)
     p.10 — "…"
```

## Embedding Plots

Bind-mount an output directory so the plots survive the throwaway container:

```bash
docker compose run --rm -v ./embedding_plots:/app/embedding_plots app \
  python plot_embeddings.py --chroma-path ./chroma_db
```

Outputs `embedding_plots/pca_embedding_plot.html` and `embedding_plots/tsne_embedding_plot.html`. Each point is one stored chunk; hover shows source, page, and a text preview.

Extra options: `--color-by page`, `--methods tsne --perplexity 10 --tsne-metric cosine`, `--max-points 2000`.

---

## Persistence

| Path | Purpose |
|---|---|
| `./chroma_db` | Persistent vector store (`kc_chroma` named volume) |
| `ingestion_checkpoints.sqlite` | LangGraph checkpoints (enables `--resume`) |
| `ingestion_log.db` | Structured log of skipped and failed files |
| `ollama_models` volume | Ollama model weights |

The two SQLite files are written to `/app` inside the container, which is discarded when a `docker compose run` container exits — to `--resume` across runs, bind-mount them (e.g. `-v ./ingestion_checkpoints.sqlite:/app/ingestion_checkpoints.sqlite`).

The pipeline is idempotent — files already in ChromaDB with the same path and last-modified timestamp are skipped. Modified files are reindexed; old chunks are deleted before upserting new ones.

## Chunking

- Size: 256 approximate whitespace tokens
- Overlap: 32 tokens
- Minimum: 40 tokens (smaller chunks dropped unless sole result for a file)
- Repeated short headers/footers stripped across pages
- Reference/bibliography sections skipped unless `--include-reference-chunks` is passed

Always run `--force-reindex` after changing chunk settings.

## E5 Prefixes

Defined in `ingestion_agent/constants.py`:

- `PASSAGE_PREFIX = "passage: "` — prepended to every stored chunk at ingestion time
- `QUERY_PREFIX = "query: "` — prepended to every query at search time
