# ── Stage 1: install Python deps + download embedding model ──────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt streamlit

# Optional HF token — pass with: docker compose build --build-arg HF_TOKEN=hf_xxx
# Authenticated downloads bypass rate limits. Leave blank for anonymous download.
ARG HF_TOKEN=""

# Bake the embedding model into the image so HF_HUB_OFFLINE=1 works at runtime.
# Uses snapshot_download (pure HTTP, no git-lfs dependency).
RUN python - <<'EOF'
import os
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN") or None
print(f"Downloading intfloat/e5-large-v2 ({'authenticated' if token else 'anonymous'})...")
snapshot_download("intfloat/e5-large-v2", local_dir="/model", token=token)
EOF

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Tesseract OCR + Poppler for PDF/image extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and CLI entry-points from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Embedding model at the path the app already checks (models/e5-large-v2)
COPY --from=builder /model /app/models/e5-large-v2

WORKDIR /app
COPY . .

# Never attempt to download models at runtime
ENV HF_HUB_OFFLINE=1

# Ollama endpoint — override via OLLAMA_BASE_URL env var or docker-compose
# host.docker.internal reaches the host machine's Ollama on macOS/Windows;
# on Linux the extra_hosts entry in docker-compose.yml provides the same alias.
ENV OLLAMA_BASE_URL=http://ollama:11434/v1

EXPOSE 8501

CMD ["streamlit", "run", "ui/app_ui.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
