"""Generic Q4_K_M GGUF export with Vulkan offline quantization."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import torch
from safetensors import safe_open

from torch2vk.checkpoints.gguf import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUFTensorType,
    GGUFValueType,
    open_gguf_mmap,
)
from torch2vk.quantize.vulkan import quantize_q4_k_vulkan, quantize_q8_0_vulkan
from torch2vk.runtime.session import RuntimeSession


GGUF_VERSION = 3
GGUF_TYPE_MODEL = "model"
GGUF_FILE_TYPE_MOSTLY_Q4_K_M = 15
GGUF_QUANTIZATION_VERSION = 2


@dataclass(frozen=True, slots=True)
class Q4KMQuantizationConfig:
    model_name: str
    gguf_arch: str
    q8_tensor_names: tuple[str, ...] = ()
    extra_uint32_metadata: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _GGUFTensor:
    name: str
    data: np.ndarray
    ggml_type: GGUFTensorType
    logical_shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass(frozen=True, slots=True)
class _GGUFMetadataValue:
    key: str
    value_type: GGUFValueType
    value: int | str


def export_q4_k_m_gguf(
    *,
    model_dir: str | Path,
    output: str | Path,
    config: Q4KMQuantizationConfig,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not overwrite and _gguf_matches_quantization(
        output_path,
        q8_tensor_names=config.q8_tensor_names,
    ):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    tensors: list[_GGUFTensor] = []
    with RuntimeSession.open(device_index=0) as rt:
        for name, tensor in _iter_safetensor_tensors(Path(model_dir).expanduser().resolve()):
            tensors.append(_tensor_to_gguf_tensor(
                rt,
                name=name,
                tensor=tensor,
                q8_tensor_names=config.q8_tensor_names,
            ))
    _write_gguf(path=output_path, metadata=_metadata(config), tensors=tuple(tensors))
    return output_path


def _gguf_matches_quantization(path: Path, *, q8_tensor_names: tuple[str, ...]) -> bool:
    with open_gguf_mmap(path) as gguf:
        if gguf.metadata.get("general.file_type") != GGUF_FILE_TYPE_MOSTLY_Q4_K_M:
            return False
        for name in q8_tensor_names:
            if gguf.entry(name).ggml_type is not GGUFTensorType.Q8_0:
                return False
    return True


def _iter_safetensor_tensors(model_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    safetensor_paths = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_paths:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    for safetensor_path in safetensor_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                yield cast(str, name), cast(torch.Tensor, checkpoint.get_tensor(name))


def _tensor_to_gguf_tensor(
    rt: RuntimeSession,
    *,
    name: str,
    tensor: torch.Tensor,
    q8_tensor_names: tuple[str, ...],
) -> _GGUFTensor:
    array = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    if array.ndim != 2:
        return _GGUFTensor(
            name=name,
            data=np.asarray(array, dtype=np.float32),
            ggml_type=GGUFTensorType.F32,
            logical_shape=tuple(int(dim) for dim in array.shape),
        )
    f32 = np.ascontiguousarray(array, dtype=np.float32)
    if name in q8_tensor_names or f32.shape[-1] % 256 != 0:
        if f32.shape[-1] % 32 != 0:
            return _GGUFTensor(
                name=name,
                data=f32,
                ggml_type=GGUFTensorType.F32,
                logical_shape=tuple(int(dim) for dim in f32.shape),
            )
        return _GGUFTensor(
            name=name,
            data=quantize_q8_0_vulkan(rt, f32, name=name),
            ggml_type=GGUFTensorType.Q8_0,
            logical_shape=tuple(int(dim) for dim in f32.shape),
        )
    return _GGUFTensor(
        name=name,
        data=quantize_q4_k_vulkan(rt, f32, name=name),
        ggml_type=GGUFTensorType.Q4_K,
        logical_shape=tuple(int(dim) for dim in f32.shape),
    )


def _metadata(config: Q4KMQuantizationConfig) -> tuple[_GGUFMetadataValue, ...]:
    values = [
        _GGUFMetadataValue("general.architecture", GGUFValueType.STRING, config.gguf_arch),
        _GGUFMetadataValue("general.name", GGUFValueType.STRING, config.model_name),
        _GGUFMetadataValue("general.type", GGUFValueType.STRING, GGUF_TYPE_MODEL),
        _GGUFMetadataValue("general.file_type", GGUFValueType.UINT32, GGUF_FILE_TYPE_MOSTLY_Q4_K_M),
        _GGUFMetadataValue("general.quantization_version", GGUFValueType.UINT32, GGUF_QUANTIZATION_VERSION),
    ]
    for key, value in config.extra_uint32_metadata:
        values.append(_GGUFMetadataValue(key, GGUFValueType.UINT32, value))
    return tuple(values)


def _write_gguf(
    *,
    path: Path,
    metadata: tuple[_GGUFMetadataValue, ...],
    tensors: tuple[_GGUFTensor, ...],
) -> None:
    tensor_offsets = _tensor_offsets(tensors)
    with path.open("wb") as handle:
        _write_u32(handle, GGUF_MAGIC)
        _write_u32(handle, GGUF_VERSION)
        _write_u64(handle, len(tensors))
        _write_u64(handle, len(metadata))
        for item in metadata:
            _write_metadata(handle, item)
        for tensor, offset in zip(tensors, tensor_offsets, strict=True):
            _write_tensor_info(handle, tensor, offset)
        _write_alignment_padding(handle)
        for tensor in tensors:
            handle.write(np.ascontiguousarray(tensor.data).tobytes(order="C"))
            _write_alignment_padding(handle)


def _tensor_offsets(tensors: tuple[_GGUFTensor, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    current = 0
    for tensor in tensors:
        offsets.append(current)
        current += _pad_to_alignment(tensor.nbytes)
    return tuple(offsets)


def _write_metadata(handle: BinaryIO, item: _GGUFMetadataValue) -> None:
    _write_string(handle, item.key)
    _write_u32(handle, int(item.value_type))
    if item.value_type is GGUFValueType.STRING:
        if not isinstance(item.value, str):
            raise TypeError(f"Expected string metadata value for {item.key}, got {item.value!r}")
        _write_string(handle, item.value)
    elif item.value_type is GGUFValueType.UINT32:
        if not isinstance(item.value, int):
            raise TypeError(f"Expected uint32 metadata value for {item.key}, got {item.value!r}")
        _write_u32(handle, item.value)
    else:
        raise ValueError(f"Unsupported GGUF metadata type for writer: {item.value_type}")


def _write_tensor_info(handle: BinaryIO, tensor: _GGUFTensor, offset: int) -> None:
    _write_string(handle, tensor.name)
    _write_u32(handle, len(tensor.logical_shape))
    for dim in reversed(tensor.logical_shape):
        _write_u64(handle, dim)
    _write_u32(handle, int(tensor.ggml_type))
    _write_u64(handle, offset)


def _write_alignment_padding(handle: BinaryIO) -> None:
    padding = handle.tell() % GGUF_DEFAULT_ALIGNMENT
    if padding != 0:
        handle.write(bytes(GGUF_DEFAULT_ALIGNMENT - padding))


def _pad_to_alignment(nbytes: int) -> int:
    padding = nbytes % GGUF_DEFAULT_ALIGNMENT
    if padding == 0:
        return nbytes
    return nbytes + GGUF_DEFAULT_ALIGNMENT - padding


def _write_string(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    _write_u64(handle, len(encoded))
    handle.write(encoded)


def _write_u32(handle: BinaryIO, value: int) -> None:
    handle.write(int(value).to_bytes(4, byteorder="little", signed=False))


def _write_u64(handle: BinaryIO, value: int) -> None:
    handle.write(int(value).to_bytes(8, byteorder="little", signed=False))
