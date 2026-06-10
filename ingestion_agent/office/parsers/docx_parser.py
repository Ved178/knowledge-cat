"""DOCX text extraction."""

from __future__ import annotations

from docx import Document


def parse_docx(file_path: str) -> str:
    """Return full text of a .docx file with paragraph breaks preserved."""
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)
