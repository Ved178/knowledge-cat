"""LibreOffice-based legacy format converter (.doc/.ppt/.xls → modern equivalents)."""

from __future__ import annotations

import os
import shutil
import subprocess


def _get_libreoffice_path() -> str:
    path = shutil.which("soffice")
    if path is None:
        raise RuntimeError(
            "LibreOffice (soffice) not found in PATH. "
            "Install it to enable .doc/.ppt/.xls support."
        )
    return path


_LEGACY_FORMAT_MAP = {
    ".doc": "docx",
    ".xls": "xlsx",
    ".ppt": "pptx",
}


def convert_to_modern_format(file_path: str, output_dir: str | None = None) -> str:
    """Convert a legacy Office file to its modern equivalent via LibreOffice.

    Raises RuntimeError if LibreOffice is not installed.
    """
    soffice = _get_libreoffice_path()
    ext = os.path.splitext(file_path)[1].lower()
    target_format = _LEGACY_FORMAT_MAP.get(ext)
    if target_format is None:
        raise ValueError(f"No conversion mapping for extension: {ext}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(file_path))

    subprocess.run(
        [soffice, "--headless", "--convert-to", target_format, file_path, "--outdir", output_dir],
        check=True,
        capture_output=True,
    )

    base = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(output_dir, f"{base}.{target_format}")
