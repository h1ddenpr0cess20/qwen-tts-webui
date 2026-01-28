"""File cleanup helpers."""

from pathlib import Path
from typing import Optional


def cleanup_files(*paths: Optional[Path]) -> None:
    """Delete temporary files if they exist.

    Args:
        *paths: File paths to remove.
    """

    for path in paths:
        if not path:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
