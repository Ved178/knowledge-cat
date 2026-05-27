"""Text extraction node for PDFs and images."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from ingestion_agent.constants import OCR_CONFIG


def _markdown_table(rows: list[list[Any]] | None) -> str:
    """Convert a pdfplumber table result to GitHub-flavored Markdown."""
    if not rows:
        return ""
    cleaned = [["" if cell is None else str(cell).strip() for cell in row] for row in rows]
    width = max(len(row) for row in cleaned)
    normalized = [row + [""] * (width - len(row)) for row in cleaned]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    table_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _extract_text_pdf(file_path: str) -> str:
    """Extract text and table content from a text-based PDF."""
    pages: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            page_parts = [f"[PAGE {index}]"]
            text = page.extract_text() or ""
            if text.strip():
                page_parts.append(text.strip())
            for table in page.extract_tables() or []:
                markdown = _markdown_table(table)
                if markdown:
                    page_parts.append(markdown)
            pages.append("\n".join(page_parts))
    return "\n\n".join(pages)


def _extract_scanned_pdf(file_path: str) -> str:
    """OCR every page of a scanned PDF and preserve page markers."""
    pages: list[str] = []
    for index, image in enumerate(convert_from_path(file_path), start=1):
        text = pytesseract.image_to_string(image, config=OCR_CONFIG)
        pages.append(f"[PAGE {index}]\n{text.strip()}")
    return "\n\n".join(pages)


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Convert an image to thresholded grayscale for stronger OCR contrast."""
    grayscale = image.convert("L")
    return grayscale.point(lambda pixel: 255 if pixel > 180 else 0, mode="1")


def _extract_image(file_path: str) -> str:
    """OCR a standalone image file."""
    with Image.open(file_path) as image:
        processed = _preprocess_image(image)
        text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
    return f"[PAGE 1]\n{text.strip()}"


def extract_text(state: dict[str, Any]) -> dict[str, Any]:
    """Extract raw text for the current file, recording failures in state."""
    file_path = state.get("current_file", "")
    file_type = state.get("file_type", "")

    try:
        if file_type == "text_pdf":
            extracted_text = _extract_text_pdf(file_path)
        elif file_type == "scanned_pdf":
            extracted_text = _extract_scanned_pdf(file_path)
        elif file_type == "image":
            extracted_text = _extract_image(file_path)
        else:
            raise ValueError(f"Unsupported file type for extraction: {file_type}")

        if not re.sub(r"\s+", "", extracted_text):
            raise ValueError("No text could be extracted")

        return {
            **state,
            "extracted_text": extracted_text,
            "last_error": "",
            "status": f"Extracted {Path(file_path).name}",
        }
    except Exception as exc:
        return {
            **state,
            "extracted_text": "",
            "last_error": str(exc),
            "status": f"Extraction failed for {file_path}: {exc}",
        }

