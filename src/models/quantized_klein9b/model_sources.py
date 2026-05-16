"""ModelScope download locations for quantized Klein 9B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modelscope import snapshot_download


FLUX_REPO_ID = "black-forest-labs/FLUX.2-klein-9B"
TEXT_ENCODER_REPO_ID = "Qwen/Qwen3-8B"
AE_REPO_ID = "black-forest-labs/FLUX.2-dev"


@dataclass(frozen=True, slots=True)
class Klein9BModelDirs:
    flux: Path
    text_encoder: Path
    ae: Path


def resolve_model_dirs(
    *,
    model_dir: str | Path | None = None,
    text_encoder_dir: str | Path | None = None,
    ae_dir: str | Path | None = None,
) -> Klein9BModelDirs:
    return Klein9BModelDirs(
        flux=_resolve_or_download(
            model_dir,
            model_id=FLUX_REPO_ID,
            allow_patterns=("flux-2-klein-9b.safetensors",),
        ),
        text_encoder=_resolve_or_download(
            text_encoder_dir,
            model_id=TEXT_ENCODER_REPO_ID,
            allow_patterns=("*.json", "*.safetensors", "*.txt"),
        ),
        ae=_resolve_or_download(
            ae_dir,
            model_id=AE_REPO_ID,
            allow_patterns=("ae.safetensors",),
        ),
    )


def _resolve_or_download(
    local_dir: str | Path | None,
    *,
    model_id: str,
    allow_patterns: tuple[str, ...],
) -> Path:
    if local_dir is not None:
        path = Path(local_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {path}")
        return path

    return Path(
        snapshot_download(
            model_id=model_id,
            revision="master",
            allow_patterns=list(allow_patterns),
            max_workers=8,
            progress_callbacks=[],
        )
    ).resolve()
