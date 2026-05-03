"""OmniVoice safetensors weight ABI verification."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from safetensors import safe_open

from torch2vk.logical import LogicalTensor

from .spec import OmniVoiceSpec, load_omnivoice_spec
from .tensors.weights import omnivoice_weight_tensors, omnivoice_weights


class _SafetensorSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> Sequence[int]: ...


class _SafetensorHandle(Protocol):
    def keys(self) -> list[str]: ...

    def get_slice(self, name: str) -> _SafetensorSlice: ...


@dataclass(frozen=True, slots=True)
class OmniVoiceWeightManifest:
    model_dir: Path
    checkpoint_path: Path
    audio_tokenizer_checkpoint_path: Path
    weights: tuple[LogicalTensor, ...]


@dataclass(frozen=True, slots=True)
class OmniVoiceCheckpointTensor:
    key: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OmniVoiceWeightMismatch:
    logical_name: str
    safetensor_key: str
    reason: str


@dataclass(frozen=True, slots=True)
class OmniVoiceWeightVerification:
    manifest: OmniVoiceWeightManifest
    checkpoint_tensors: dict[str, OmniVoiceCheckpointTensor]
    mismatches: tuple[OmniVoiceWeightMismatch, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def raise_for_mismatches(self) -> None:
        if self.ok:
            return
        first = self.mismatches[0]
        raise ValueError(
            "OmniVoice safetensor weight mismatch: "
            f"{first.logical_name} -> {first.safetensor_key}: {first.reason}"
        )


def omnivoice_weight_manifest(
    model_dir: str | Path,
    spec: OmniVoiceSpec | None = None,
) -> OmniVoiceWeightManifest:
    resolved = Path(model_dir).expanduser().resolve()
    resolved_spec = load_omnivoice_spec(resolved) if spec is None else spec
    return OmniVoiceWeightManifest(
        model_dir=resolved,
        checkpoint_path=resolved / "model.safetensors",
        audio_tokenizer_checkpoint_path=resolved / "audio_tokenizer" / "model.safetensors",
        weights=omnivoice_weight_tensors(omnivoice_weights(resolved_spec)),
    )


def verify_omnivoice_safetensor_weights(
    model_dir: str | Path,
    spec: OmniVoiceSpec | None = None,
) -> OmniVoiceWeightVerification:
    manifest = omnivoice_weight_manifest(model_dir, spec=spec)
    checkpoint_tensors = {
        **_checkpoint_tensors(manifest.checkpoint_path),
        **_checkpoint_tensors(
            manifest.audio_tokenizer_checkpoint_path,
            prefix="audio_tokenizer:",
        ),
    }
    mismatches: list[OmniVoiceWeightMismatch] = []
    for weight in manifest.weights:
        if weight.source is None:
            mismatches.append(
                OmniVoiceWeightMismatch(
                    logical_name=weight.name,
                    safetensor_key="",
                    reason="missing safetensor source",
                )
            )
            continue
        checkpoint_tensor = checkpoint_tensors.get(weight.source.key)
        if checkpoint_tensor is None:
            mismatches.append(
                OmniVoiceWeightMismatch(
                    logical_name=weight.name,
                    safetensor_key=weight.source.key,
                    reason="missing checkpoint tensor",
                )
            )
            continue
        if checkpoint_tensor.dtype != weight.dtype:
            mismatches.append(
                OmniVoiceWeightMismatch(
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
                OmniVoiceWeightMismatch(
                    logical_name=weight.name,
                    safetensor_key=weight.source.key,
                    reason=(
                        f"shape mismatch logical={expected_shape} "
                        f"checkpoint={checkpoint_tensor.shape}"
                    ),
                )
            )
    return OmniVoiceWeightVerification(
        manifest=manifest,
        checkpoint_tensors=checkpoint_tensors,
        mismatches=tuple(mismatches),
    )


def _checkpoint_tensors(
    path: Path,
    *,
    prefix: str = "",
) -> dict[str, OmniVoiceCheckpointTensor]:
    if not path.exists():
        raise FileNotFoundError(f"OmniVoice generator checkpoint is missing: {path}")
    tensors: dict[str, OmniVoiceCheckpointTensor] = {}
    handle_context = cast(
        "AbstractContextManager[_SafetensorHandle]",
        safe_open(path, framework="pt", device="cpu"),
    )
    with handle_context as handle:
        selected_keys = tuple(handle.keys())
        for key in selected_keys:
            tensor_slice = handle.get_slice(key)
            full_key = f"{prefix}{key}"
            tensors[full_key] = OmniVoiceCheckpointTensor(
                key=full_key,
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
