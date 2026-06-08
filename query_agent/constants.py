"""Shared constants for the Knowledge Catalyst query layer."""

from __future__ import annotations

DEFAULT_TOP_K = 5
# LM Studio serves an OpenAI-compatible API at port 1234 by default.
DEFAULT_LM_STUDIO_MODEL = "local-model"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

REFORMULATION_PROMPT = """\
You are a search query optimizer for a scientific document knowledge base.

Given the user's raw query, output 2-4 concise, precise search terms or short phrases \
that will retrieve the most relevant document chunks via semantic vector search. \
Output only the expanded query string — no explanation, no bullets, no labels.

Raw query: {query}
Expanded query:"""

SUMMARY_PROMPT = """\
You are a research assistant. Based on the retrieved document excerpts below, \
write a 2-3 sentence answer to the user's question. Cite sources inline as [filename p.N].

Question: {query}

Retrieved excerpts:
{excerpts}

Answer:"""
