# Knowledge Catalyst

A local knowledge base pipeline with semantic search. Ingests files into a ChromaDB vector store and lets you query them via a Streamlit UI or interactive REPL, with LLM-powered summarisation through Ollama.

## Docker (recommended)

The Docker setup bundles the app and an Ollama instance — no local Python or system dependencies required.

**First run** (builds the image and downloads the embedding model, ~3–4 GB total):

```bash
docker compose up --build
```

Open **http://localhost:8501** in your browser.

**Pull a model into Ollama** (one-time, persists in a named volume):

```bash
docker compose exec ollama ollama pull llama3.2
```

Any model you pull is available in the UI's model selector immediately. The app falls back to plain semantic search if no model is loaded.

### Ingest and query via CLI

```bash
# Ingest documents from ./data
docker compose run --rm app python ingest.py \
  --paths /app/data \
  --embedding-model /app/models/e5-large-v2

# Interactive query REPL
docker compose run --rm -it app python query.py \
  --embedding-model /app/models/e5-large-v2
```

### Using a different Ollama model

```bash
docker compose exec ollama ollama pull mistral
```

Then select it in the UI sidebar, or pass `--ollama-model mistral` to the query CLI.

---

## Local Setup (alternative)

Requires Python 3.10+.

**macOS** — double-click `setup.command` in Finder (right-click → Open the first time), or:

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
3. Installs all Python dependencies
4. Downloads `intfloat/e5-large-v2` in safetensors format (~1.3 GB)

To also ingest `.doc`, `.ppt`, and `.xls` files, install LibreOffice ([libreoffice.org](https://www.libreoffice.org/)) and ensure `soffice` is on your PATH.

### One-click launchers

| macOS | Windows | What it does |
|---|---|---|
| `run_ingest.command` | `run_ingest.bat` | Ingest documents from `./data` |
| `run_query.command` | `run_query.bat` | Open the interactive search REPL |
| `run_plots.command` | `run_plots.bat` | Generate PCA + t-SNE embedding plots |

### Local CLI

```bash
# Activate venv first
source env/bin/activate          # macOS
env\Scripts\activate             # Windows

python ingest.py --paths ./data --embedding-model models/e5-large-v2
python query.py --embedding-model models/e5-large-v2
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

```bash
# Resume a checkpointed run
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --resume

# Force-reindex already-stored files
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --force-reindex

# Tune chunking (always pair with --force-reindex)
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --force-reindex \
  --chunk-size 220 --chunk-overlap 30 --min-chunk-size 40

# Keep references/bibliography sections (skipped by default)
python ingest.py --paths ./data --embedding-model models/e5-large-v2 --include-reference-chunks
```

Live progress:

```text
queued=12 processed=3 skipped=0 errors=0 status=Stored 4 chunks for sample.pdf
```

## Query CLI Reference

```bash
python query.py --embedding-model models/e5-large-v2 --top-k 10
python query.py --ollama-url http://localhost:11434/v1 --ollama-model llama3.2
python query.py --chroma-path ./chroma_db --collection knowledge_catalyst
```

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

```bash
python plot_embeddings.py --chroma-path ./chroma_db
```

Outputs `embedding_plots/pca_embedding_plot.html` and `embedding_plots/tsne_embedding_plot.html`. Each point is one stored chunk; hover shows source, page, and a text preview.

```bash
python plot_embeddings.py --color-by page
python plot_embeddings.py --methods tsne --perplexity 10 --tsne-metric cosine
python plot_embeddings.py --max-points 2000
```

---

## Persistence

| Path | Purpose |
|---|---|
| `./chroma_db` | Persistent vector store (Docker: `kc_chroma` named volume) |
| `ingestion_checkpoints.sqlite` | LangGraph checkpoints (enables `--resume`) |
| `ingestion_log.db` | Structured log of skipped and failed files |
| `ollama_models` volume | Ollama model weights (Docker only) |

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
