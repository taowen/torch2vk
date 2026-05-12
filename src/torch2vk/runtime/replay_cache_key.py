"""Helpers for replay cache keys."""

from __future__ import annotations

import hashlib
from pathlib import Path


def source_tree_digest(source_file: str | Path) -> str:
    root = Path(source_file).parent
    hasher = hashlib.sha256()
    for path in (
        Path(source_file),
        *sorted((root / "dispatch").glob("*.py")),
        *sorted((root / "shaders").glob("*.py")),
        *sorted((root / "tensors").glob("*.py")),
    ):
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]
