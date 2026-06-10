"""Office document parser factory."""

from __future__ import annotations

import os
import tempfile


def parse_office_file(file_path: str) -> dict:
    """Parse a supported Office document and return ``{"content": str, "metadata": dict}``.

    Legacy formats (.doc/.ppt/.xls) require LibreOffice; a RuntimeError is raised
    if it is absent so the ingestion pipeline can log the failure gracefully.
    """
    ext = os.path.splitext(file_path)[1].lower()
    content = _extract(file_path, ext)
    metadata = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": ext.lstrip("."),
        "file_size": os.path.getsize(file_path),
    }
    return {"content": content, "metadata": metadata}


def _extract(file_path: str, ext: str) -> str:
    if ext == ".docx":
        from ingestion_agent.office.parsers.docx_parser import parse_docx
        return parse_docx(file_path)

    if ext == ".pptx":
        from ingestion_agent.office.parsers.pptx_parser import parse_pptx
        return parse_pptx(file_path)

    if ext == ".xlsx":
        from ingestion_agent.office.parsers.xlsx_parser import parse_xlsx
        return parse_xlsx(file_path)

    if ext == ".csv":
        from ingestion_agent.office.parsers.csv_parser import parse_csv
        return parse_csv(file_path)

    # Legacy formats require LibreOffice conversion
    if ext in (".doc", ".ppt", ".xls"):
        from ingestion_agent.office.converters.converter import convert_to_modern_format
        with tempfile.TemporaryDirectory() as tmp_dir:
            converted = convert_to_modern_format(file_path, output_dir=tmp_dir)
            modern_ext = os.path.splitext(converted)[1].lower()
            return _extract(converted, modern_ext)

    raise ValueError(f"Unsupported office extension: {ext}")
