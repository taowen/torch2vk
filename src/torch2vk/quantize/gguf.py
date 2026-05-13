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
from torch2vk.quantize.vulkan import quantize_q4_k_vulkan, quantize_q6_k_vulkan, quantize_q8_0_vulkan
from torch2vk.runtime.session import RuntimeSession


GGUF_VERSION = 3
GGUF_TYPE_MODEL = "model"
GGUF_FILE_TYPE_MOSTLY_Q4_K_M = 15
GGUF_QUANTIZATION_VERSION = 2


@dataclass(frozen=True, slots=True)
class Q4KMQuantizationConfig:
    model_name: str
    gguf_arch: str
    q6_tensor_names: tuple[str, ...] = ()
    q6_tensor_prefixes: tuple[str, ...] = ()
    q8_tensor_names: tuple[str, ...] = ()
    q8_tensor_prefixes: tuple[str, ...] = ()
    safetensor_subdirs: tuple[str, ...] = ()
    extra_uint32_metadata: tuple[tuple[str, int], ...] = ()

    @property
    def has_q6(self) -> bool:
        return bool(self.q6_tensor_names or self.q6_tensor_prefixes)

    def gguf_type(
        self,
        *,
        checkpoint_key: str,
        shape: tuple[int, ...],
        dtype: str = "float32",
    ) -> GGUFTensorType:
        dtype = _checkpoint_float_dtype(dtype)
        if dtype != "float32":
            return GGUFTensorType.F32
        force_q6 = checkpoint_key in self.q6_tensor_names or checkpoint_key.startswith(
            self.q6_tensor_prefixes
        )
        force_q8 = checkpoint_key in self.q8_tensor_names or checkpoint_key.startswith(
            self.q8_tensor_prefixes
        )
        if force_q6 and len(shape) >= 2:
            return GGUFTensorType.Q6_K
        if force_q8 and len(shape) >= 2:
            return GGUFTensorType.Q8_0
        if len(shape) != 2:
            return GGUFTensorType.F32
        cols = shape[-1]
        if cols % 256 == 0:
            return GGUFTensorType.Q4_K
        if cols % 32 == 0:
            return GGUFTensorType.Q8_0
        return GGUFTensorType.F32

    def declare_weight(
        self,
        *,
        checkpoint_key: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> _WeightDeclaration:
        dtype = _checkpoint_float_dtype(dtype)
        gguf_type = self.gguf_type(
            checkpoint_key=checkpoint_key,
            dtype=dtype,
            shape=shape,
        )
        if gguf_type is GGUFTensorType.Q6_K:
            n, k = _matrix_shape(shape)
            if k % 256 != 0:
                raise ValueError(
                    f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}"
                )
            return _WeightDeclaration(
                gguf_type=gguf_type,
                dtype="uint16",
                shape=(n, k // 256 * 105),
                layout_source=f"q6_k_halfwords_layout(logical_k={k})",
            )
        if gguf_type is GGUFTensorType.Q8_0:
            n, k = _matrix_shape(shape)
            padded_k = _round_up(k, 32)
            return _WeightDeclaration(
                gguf_type=gguf_type,
                dtype="uint16",
                shape=(n, padded_k // 32 * 17),
                layout_source=f"q8_0_halfwords_layout(logical_k={k})",
            )
        if gguf_type is GGUFTensorType.Q4_K:
            n, k = shape
            return _WeightDeclaration(
                gguf_type=gguf_type,
                dtype="uint32",
                shape=(n, k // 256 * 36),
                layout_source=f"q4_k_words_layout(logical_k={k})",
            )
        return _WeightDeclaration(
            gguf_type=gguf_type,
            dtype=dtype,
            shape=shape,
            layout_source="CONTIGUOUS_LAYOUT",
        )


@dataclass(frozen=True, slots=True)
class _WeightDeclaration:
    gguf_type: GGUFTensorType
    dtype: str
    shape: tuple[int, ...]
    layout_source: str


def q4_k_m_more_bits_layer_indices(num_layers: int) -> tuple[int, ...]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    first_eighth = num_layers // 8
    last_eighth = (7 * num_layers) // 8
    return tuple(
        layer_idx
        for layer_idx in range(num_layers)
        if layer_idx < first_eighth
        or layer_idx >= last_eighth
        or (layer_idx - first_eighth) % 3 == 2
    )


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
        config=config,
        extra_uint32_metadata=config.extra_uint32_metadata,
    ):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    tensors: list[_GGUFTensor] = []
    seen_tensor_names: set[str] = set()
    with RuntimeSession.open(device_index=0) as rt:
        for name, tensor in _iter_safetensor_tensors(
            Path(model_dir).expanduser().resolve(),
            safetensor_subdirs=config.safetensor_subdirs,
        ):
            if name in seen_tensor_names:
                raise ValueError(f"Duplicate safetensor tensor name while writing GGUF: {name}")
            seen_tensor_names.add(name)
            tensors.append(_tensor_to_gguf_tensor(
                rt,
                name=name,
                tensor=tensor,
                config=config,
            ))
    _write_gguf(path=output_path, metadata=_metadata(config), tensors=tuple(tensors))
    return output_path


def _gguf_matches_quantization(
    path: Path,
    *,
    config: Q4KMQuantizationConfig,
    extra_uint32_metadata: tuple[tuple[str, int], ...],
) -> bool:
    with open_gguf_mmap(path) as gguf:
        if gguf.metadata.get("general.file_type") != GGUF_FILE_TYPE_MOSTLY_Q4_K_M:
            return False
        for key, value in extra_uint32_metadata:
            if gguf.metadata.get(key) != value:
                return False
        for name, entry in gguf.tensors.items():
            expected_type = config.gguf_type(checkpoint_key=name, shape=entry.logical_shape)
            if entry.ggml_type is not expected_type:
                return False
    return True


def _iter_safetensor_tensors(
    model_dir: Path,
    *,
    safetensor_subdirs: tuple[str, ...],
) -> Iterator[tuple[str, torch.Tensor]]:
    for source_dir in (model_dir, *(model_dir / subdir for subdir in safetensor_subdirs)):
        safetensor_paths = sorted(source_dir.glob("*.safetensors"))
        if not safetensor_paths:
            raise FileNotFoundError(f"No safetensors files found in {source_dir}")
        for safetensor_path in safetensor_paths:
            with safe_open(safetensor_path, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    yield cast(str, name), cast(torch.Tensor, checkpoint.get_tensor(name))


def _tensor_to_gguf_tensor(
    rt: RuntimeSession,
    *,
    name: str,
    tensor: torch.Tensor,
    config: Q4KMQuantizationConfig,
) -> _GGUFTensor:
    array = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    gguf_type = config.gguf_type(
        checkpoint_key=name,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(dim) for dim in array.shape),
    )
    if gguf_type is GGUFTensorType.Q6_K:
        rows, cols = _matrix_shape(tuple(int(dim) for dim in array.shape))
        if cols % 256 != 0:
            raise ValueError(f"Q6_K tensor {name} requires K to be divisible by 256, got {cols}")
        f32 = np.ascontiguousarray(array.reshape(rows, cols), dtype=np.float32)
        return _GGUFTensor(
            name=name,
            data=quantize_q6_k_vulkan(rt, f32, name=name),
            ggml_type=GGUFTensorType.Q6_K,
            logical_shape=(rows, cols),
        )
    if gguf_type is GGUFTensorType.Q8_0:
        rows, cols = _matrix_shape(tuple(int(dim) for dim in array.shape))
        f32 = np.ascontiguousarray(array.reshape(rows, cols), dtype=np.float32)
        padded_cols = _round_up(cols, 32)
        if padded_cols != cols:
            padded = np.zeros((rows, padded_cols), dtype=np.float32)
            padded[:, :cols] = f32
            f32 = padded
        return _GGUFTensor(
            name=name,
            data=quantize_q8_0_vulkan(rt, f32, name=name),
            ggml_type=GGUFTensorType.Q8_0,
            logical_shape=(rows, padded_cols),
        )

    if gguf_type is GGUFTensorType.F32:
        return _GGUFTensor(
            name=name,
            data=np.asarray(array, dtype=np.float32),
            ggml_type=GGUFTensorType.F32,
            logical_shape=tuple(int(dim) for dim in array.shape),
        )
    if gguf_type is not GGUFTensorType.Q4_K:
        raise ValueError(f"Unsupported GGUF tensor type for {name}: {gguf_type}")
    f32 = np.ascontiguousarray(array, dtype=np.float32)
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


def _matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    rows = shape[0]
    cols = 1
    for dim in shape[1:]:
        cols *= dim
    return rows, cols


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _checkpoint_float_dtype(dtype: str) -> str:
    if dtype in {"float32", "float16", "bfloat16"}:
        return "float32"
    return dtype


def _write_string(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    _write_u64(handle, len(encoded))
    handle.write(encoded)


def _write_u32(handle: BinaryIO, value: int) -> None:
    handle.write(int(value).to_bytes(4, byteorder="little", signed=False))


def _write_u64(handle: BinaryIO, value: int) -> None:
    handle.write(int(value).to_bytes(8, byteorder="little", signed=False))
