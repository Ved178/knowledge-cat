"""File classification node for ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber

from ingestion_agent.constants import (
    IMAGE_EXTENSIONS,
    OFFICE_DOC_EXTENSIONS,
    OFFICE_PPTX_EXTENSIONS,
    OFFICE_TABULAR_EXTENSIONS,
)


def classify_file(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the current file as a text PDF, scanned PDF, image, office type, or unsupported."""
    file_path = state.get("current_file", "")
    suffix = Path(file_path).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        file_type = "image"
    elif suffix == ".pdf":
        file_type = "scanned_pdf"
        try:
            with pdfplumber.open(file_path) as pdf:
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""
                    if len(first_page_text.strip()) >= 50:
                        file_type = "text_pdf"
        except Exception as exc:
            return {
                **state,
                "file_type": "unsupported",
                "skip_reason": f"PDF classification failed: {exc}",
                "status": f"Skipped {file_path}",
            }
    elif suffix in OFFICE_DOC_EXTENSIONS:
        file_type = "office_doc"
    elif suffix in OFFICE_PPTX_EXTENSIONS:
        file_type = "office_pptx"
    elif suffix in OFFICE_TABULAR_EXTENSIONS:
        file_type = "office_tabular"
    else:
        file_type = "unsupported"

    skip_reason = "" if file_type != "unsupported" else f"Unsupported extension: {suffix}"
    return {**state, "file_type": file_type, "skip_reason": skip_reason, "status": file_type}

