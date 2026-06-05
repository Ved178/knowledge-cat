"""Shared constants for the Knowledge Catalyst ingestion pipeline."""

from __future__ import annotations

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "knowledge_catalyst"
DEFAULT_LOG_DB_PATH = "ingestion_log.db"
DEFAULT_CHECKPOINT_DB_PATH = "ingestion_checkpoints.sqlite"

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32
MIN_CHUNK_TOKENS = 40
OCR_CONFIG = "--psm 6"
