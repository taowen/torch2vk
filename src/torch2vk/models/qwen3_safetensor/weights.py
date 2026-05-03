"""Qwen3 safetensors weight manifest helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from safetensors import safe_open

from torch2vk.logical import LogicalTensor

from .schema import qwen3_weight_tensors
from .spec import Qwen3Spec, load_qwen3_spec


class _SafetensorSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> Sequence[int]: ...


class _SafetensorHandle(Protocol):
    def keys(self) -> list[str]: ...

    def get_slice(self, name: str) -> _SafetensorSlice: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class Qwen3WeightManifest:
    model_dir: Path
    checkpoint_path: Path
    weights: tuple[LogicalTensor, ...]

    def by_name(self) -> dict[str, LogicalTensor]:
        return {weight.name: weight for weight in self.weights}


@dataclass(frozen=True, slots=True)
class Qwen3CheckpointTensor:
    key: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Qwen3WeightMismatch:
    logical_name: str
    safetensor_key: str
    reason: str


@dataclass(frozen=True, slots=True)
class Qwen3WeightVerification:
    manifest: Qwen3WeightManifest
    checkpoint_tensors: Mapping[str, Qwen3CheckpointTensor]
    mismatches: tuple[Qwen3WeightMismatch, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def raise_for_mismatches(self) -> None:
        if self.ok:
            return
        first = self.mismatches[0]
        raise ValueError(
            "Qwen3 safetensor weight mismatch: "
            f"{first.logical_name} -> {first.safetensor_key}: {first.reason}"
        )


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


def verify_qwen3_safetensor_weights(
    model_dir: str | Path,
    spec: Qwen3Spec | None = None,
) -> Qwen3WeightVerification:
    manifest = qwen3_weight_manifest(model_dir, spec=spec)
    checkpoint_tensors = _checkpoint_tensors(manifest.model_dir, manifest.checkpoint_path)
    mismatches: list[Qwen3WeightMismatch] = []
    for weight in manifest.weights:
        if weight.source is None:
            mismatches.append(
                Qwen3WeightMismatch(
                    logical_name=weight.name,
                    safetensor_key="",
                    reason="missing safetensor source",
                )
            )
            continue
        checkpoint_tensor = checkpoint_tensors.get(weight.source.key)
        if checkpoint_tensor is None:
            mismatches.append(
                Qwen3WeightMismatch(
                    logical_name=weight.name,
                    safetensor_key=weight.source.key,
                    reason="missing checkpoint tensor",
                )
            )
            continue
        if checkpoint_tensor.dtype != weight.dtype:
            mismatches.append(
                Qwen3WeightMismatch(
                    logical_name=weight.name,
                    safetensor_key=weight.source.key,
                    reason=(
                        f"dtype mismatch logical={weight.dtype} "
                        f"checkpoint={checkpoint_tensor.dtype}"
                    ),
                )
            )
        expected_shape = tuple(int(dim) for dim in weight.shape)
        if checkpoint_tensor.shape != expected_shape:
            mismatches.append(
                Qwen3WeightMismatch(
                    logical_name=weight.name,
                    safetensor_key=weight.source.key,
                    reason=(
                        f"shape mismatch logical={expected_shape} "
                        f"checkpoint={checkpoint_tensor.shape}"
                    ),
                )
            )
    return Qwen3WeightVerification(
        manifest=manifest,
        checkpoint_tensors=checkpoint_tensors,
        mismatches=tuple(mismatches),
    )


def qwen3_safetensor_weight_bytes(
    model_dir: str | Path,
    weight: LogicalTensor,
    *,
    spec: Qwen3Spec | None = None,
) -> bytes:
    verification = verify_qwen3_safetensor_weights(model_dir, spec=spec)
    verification.raise_for_mismatches()
    if weight.source is None:
        raise ValueError(f"{weight.name} has no safetensor source")
    checkpoint_tensor = verification.checkpoint_tensors[weight.source.key]
    handle_context = cast(
        "AbstractContextManager[_SafetensorHandle]",
        safe_open(checkpoint_tensor.shard, framework="pt", device="cpu"),
    )
    with handle_context as handle:
        tensor = handle.get_tensor(weight.source.key).contiguous()
    return _raw_tensor_bytes(weight, tensor)


def qwen3_safetensor_weight_payloads(
    model_dir: str | Path,
    *,
    spec: Qwen3Spec | None = None,
) -> dict[str, bytes]:
    manifest = qwen3_weight_manifest(model_dir, spec=spec)
    return {
        weight.name: qwen3_safetensor_weight_bytes(model_dir, weight, spec=spec)
        for weight in manifest.weights
    }


def _checkpoint_path(model_dir: Path) -> Path:
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return single_file
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return index
    raise FileNotFoundError(f"Qwen3 checkpoint is missing: {single_file}")


def _checkpoint_tensors(
    model_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Qwen3CheckpointTensor]:
    if checkpoint_path.name == "model.safetensors.index.json":
        return _indexed_checkpoint_tensors(model_dir, checkpoint_path)
    return _safetensor_file_tensors(checkpoint_path)


def _indexed_checkpoint_tensors(
    model_dir: Path,
    index_path: Path,
) -> dict[str, Qwen3CheckpointTensor]:
    raw_json: object = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw_json, dict):
        raise TypeError(f"{index_path} must contain a JSON object")
    raw = cast("Mapping[str, object]", raw_json)
    weight_map_json = raw.get("weight_map")
    if not isinstance(weight_map_json, dict):
        raise TypeError(f"{index_path}:weight_map must be an object")
    weight_map = cast("Mapping[object, object]", weight_map_json)
    by_shard: dict[Path, list[str]] = {}
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard_name, str):
            raise TypeError(f"{index_path}:weight_map entries must be string -> string")
        by_shard.setdefault(model_dir / shard_name, []).append(key)
    tensors: dict[str, Qwen3CheckpointTensor] = {}
    for shard, keys in by_shard.items():
        tensors.update(_safetensor_file_tensors(shard, keys=tuple(keys)))
    return tensors


def _safetensor_file_tensors(
    path: Path,
    *,
    keys: tuple[str, ...] | None = None,
) -> dict[str, Qwen3CheckpointTensor]:
    tensors: dict[str, Qwen3CheckpointTensor] = {}
    handle_context = cast(
        "AbstractContextManager[_SafetensorHandle]",
        safe_open(path, framework="pt", device="cpu"),
    )
    with handle_context as handle:
        selected_keys = tuple(handle.keys()) if keys is None else keys
        for key in selected_keys:
            tensor_slice = handle.get_slice(key)
            tensors[key] = Qwen3CheckpointTensor(
                key=key,
                shard=path,
                dtype=_normalize_safetensor_dtype(tensor_slice.get_dtype()),
                shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
            )
    return tensors


def _normalize_safetensor_dtype(dtype: str) -> str:
    normalized = dtype.lower()
    aliases = {
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
        "i32": "int32",
        "i64": "int64",
    }
    return aliases.get(normalized, normalized)


def _raw_tensor_bytes(weight: LogicalTensor, tensor: torch.Tensor) -> bytes:
    if weight.dtype == "bfloat16":
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{weight.name} expected torch.bfloat16, got {tensor.dtype}")
        return tensor.view(torch.uint16).numpy().tobytes()
    if weight.dtype == "float16":
        if tensor.dtype != torch.float16:
            raise ValueError(f"{weight.name} expected torch.float16, got {tensor.dtype}")
        return tensor.numpy().tobytes()
    if weight.dtype == "float32":
        if tensor.dtype != torch.float32:
            raise ValueError(f"{weight.name} expected torch.float32, got {tensor.dtype}")
        return tensor.numpy().tobytes()
    raise ValueError(f"{weight.name} has unsupported raw safetensor dtype {weight.dtype!r}")
