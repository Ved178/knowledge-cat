"""CSV text extraction."""

from __future__ import annotations

import pandas as pd


def parse_csv(file_path: str) -> str:
    """Return a human-readable string representation of a CSV file."""
    df = pd.read_csv(file_path)
    return df.to_string(index=False)
