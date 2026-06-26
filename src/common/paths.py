from __future__ import annotations

from pathlib import Path


def ensure_parent(path: Path) -> Path:
    """Create a file path's parent directory and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
