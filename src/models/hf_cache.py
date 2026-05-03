"""Hugging Face cache resolution helpers used by model examples."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_cached_model(repo_id: str, model_dir: str | Path | None = None) -> Path:
    """Resolve a local model directory or a repo already present in the HF cache."""
    if model_dir is not None:
        resolved = Path(model_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"Model directory does not exist: {resolved}")
        return resolved
    return Path(snapshot_download(repo_id=repo_id, local_files_only=True)).resolve()


def load_config_json(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json is missing: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {config_path}, got {type(value).__name__}")
    return value


def require_file(model_dir: Path, relative_path: str) -> Path:
    path = model_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required model file is missing: {path}")
    return path
