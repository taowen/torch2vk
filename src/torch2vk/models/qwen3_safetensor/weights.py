"""Qwen3 safetensors weight manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch2vk.logical import LogicalTensor

from .schema import qwen3_weight_tensors
from .spec import Qwen3Spec, load_qwen3_spec


@dataclass(frozen=True, slots=True)
class Qwen3WeightManifest:
    model_dir: Path
    checkpoint_path: Path
    weights: tuple[LogicalTensor, ...]

    def by_name(self) -> dict[str, LogicalTensor]:
        return {weight.name: weight for weight in self.weights}


def qwen3_weight_manifest(
    model_dir: str | Path,
    spec: Qwen3Spec | None = None,
) -> Qwen3WeightManifest:
    resolved = Path(model_dir).expanduser().resolve()
    resolved_spec = load_qwen3_spec(resolved) if spec is None else spec
    return Qwen3WeightManifest(
        model_dir=resolved,
        checkpoint_path=_checkpoint_path(resolved),
        weights=qwen3_weight_tensors(resolved_spec),
    )


def _checkpoint_path(model_dir: Path) -> Path:
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return single_file
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return index
    raise FileNotFoundError(f"Qwen3 checkpoint is missing: {single_file}")
