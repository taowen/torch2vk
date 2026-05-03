"""OmniVoice safetensor model-directory resolution."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_DIR = Path("models/weights/omnivoice-safetensor")


def resolve_omnivoice_model_dir(model_dir: str | Path | None = None) -> Path:
    candidate = Path(
        model_dir
        if model_dir is not None
        else os.environ.get("OMNIVOICE_SAFETENSOR_DIR", str(DEFAULT_MODEL_DIR))
    )
    resolved = candidate.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"missing OmniVoice safetensor model directory: {resolved}")
    config = resolved / "config.json"
    generator_weights = resolved / "model.safetensors"
    tokenizer_weights = resolved / "audio_tokenizer" / "model.safetensors"
    missing = [path for path in (config, generator_weights, tokenizer_weights) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "OmniVoice safetensor model directory is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    return resolved
