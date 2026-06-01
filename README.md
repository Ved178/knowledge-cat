# Knowledge Catalyst Ingestion Layer

This package implements Layer 1 of the local Knowledge Catalyst pipeline:

```text
scan_drive -> classify_file -> extract_text -> chunk_text -> embed_chunks -> store_to_chroma -> update_status
```

It crawls supported local files, extracts text, chunks it with the E5 passage prefix, embeds chunks with `intfloat/e5-large-v2`, and stores the vectors in a persistent local ChromaDB collection named `knowledge_catalyst`.

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

For large collections, sample a deterministic subset before plotting:

```bash
python plot_embeddings.py --max-points 2000
```

Each point represents one stored chunk. Hover shows the source file, page, chunk index, file type, and a text preview.

## Persistence

- Vector store: `./chroma_db`
- LangGraph checkpoints: `ingestion_checkpoints.sqlite`
- Structured ingestion logs: `ingestion_log.db`

The pipeline is idempotent. Files already present in ChromaDB with the same absolute path and last modified timestamp are skipped. Modified files are reindexed with deterministic chunk IDs after old chunks for that source are deleted.

## E5 Prefixes

The codebase defines both E5 prefixes in `ingestion_agent/constants.py`:

- `PASSAGE_PREFIX = "passage: "` for stored document chunks
- `QUERY_PREFIX = "query: "` for query-time embeddings in downstream layers
# knowledge-cat
