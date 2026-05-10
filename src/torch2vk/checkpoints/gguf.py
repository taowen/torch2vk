"""GGUF mmap reader for direct tensor views over llama.cpp-exported checkpoints."""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO

from torch2vk.vulkan.types import Residency, TensorSpec, dtype_nbytes


GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFTensorType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass(frozen=True, slots=True)
class _GGMLTypeInfo:
    block_size: int
    type_size: int
    spec_dtype: str | None


_GGML_TYPE_INFO: dict[GGUFTensorType, _GGMLTypeInfo] = {
    GGUFTensorType.F32: _GGMLTypeInfo(block_size=1, type_size=4, spec_dtype="float32"),
    GGUFTensorType.F16: _GGMLTypeInfo(block_size=1, type_size=2, spec_dtype="float16"),
    GGUFTensorType.I8: _GGMLTypeInfo(block_size=1, type_size=1, spec_dtype="int8"),
    GGUFTensorType.I16: _GGMLTypeInfo(block_size=1, type_size=2, spec_dtype="int16"),
    GGUFTensorType.I32: _GGMLTypeInfo(block_size=1, type_size=4, spec_dtype="int32"),
    GGUFTensorType.I64: _GGMLTypeInfo(block_size=1, type_size=8, spec_dtype="int64"),
    GGUFTensorType.F64: _GGMLTypeInfo(block_size=1, type_size=8, spec_dtype="float64"),
    GGUFTensorType.BF16: _GGMLTypeInfo(block_size=1, type_size=2, spec_dtype="bfloat16"),
    GGUFTensorType.Q8_0: _GGMLTypeInfo(block_size=32, type_size=34, spec_dtype=None),
    GGUFTensorType.Q2_K: _GGMLTypeInfo(block_size=256, type_size=84, spec_dtype=None),
    GGUFTensorType.Q3_K: _GGMLTypeInfo(block_size=256, type_size=110, spec_dtype=None),
    GGUFTensorType.Q4_K: _GGMLTypeInfo(block_size=256, type_size=144, spec_dtype=None),
    GGUFTensorType.Q5_K: _GGMLTypeInfo(block_size=256, type_size=176, spec_dtype=None),
    GGUFTensorType.Q6_K: _GGMLTypeInfo(block_size=256, type_size=210, spec_dtype=None),
    GGUFTensorType.Q8_K: _GGMLTypeInfo(block_size=256, type_size=292, spec_dtype=None),
}


@dataclass(frozen=True, slots=True)
class GGUFTensorEntry:
    name: str
    ggml_type: GGUFTensorType
    ggml_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    physical_dtype: str
    physical_shape: tuple[int, ...]
    data_offset: int
    nbytes: int

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(
            dtype=self.physical_dtype,
            shape=self.physical_shape,
            residency=Residency.HOST,
        )


@dataclass(slots=True)
class _MappedGGUFFile:
    path: Path
    _file: BinaryIO | None
    _mmap: mmap.mmap | None

    @classmethod
    def open(cls, path: Path) -> "_MappedGGUFFile":
        file_handle = path.open("rb")
        mapped = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_COPY)
        return cls(path=path, _file=file_handle, _mmap=mapped)

    def close(self) -> None:
        mapped = self._mmap
        if mapped is not None:
            mapped.close()
            self._mmap = None
        file_handle = self._file
        if file_handle is not None:
            file_handle.close()
            self._file = None

    def slice(self, *, offset: int, size: int) -> memoryview:
        mapped = self._mmap
        if mapped is None:
            raise RuntimeError(f"GGUF file is closed: {self.path}")
        return memoryview(mapped)[offset : offset + size]


class GGUFMmap:
    """Mmap-backed GGUF storage with logical tensor metadata and raw row slicing."""

    def __init__(
        self,
        *,
        path: str | Path,
        tensors: dict[str, GGUFTensorEntry],
        metadata: dict[str, object],
    ) -> None:
        self.path = Path(path)
        self.tensors = dict(tensors)
        self.metadata = dict(metadata)
        self._mapped_file = _MappedGGUFFile.open(self.path)

    def close(self) -> None:
        self._mapped_file.close()

    def __enter__(self) -> "GGUFMmap":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def entry(self, name: str) -> GGUFTensorEntry:
        try:
            return self.tensors[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.tensors))
            raise KeyError(f"Unknown GGUF tensor entry {name!r}. Available: {known}") from exc

    def buffer_slice(self, name: str) -> memoryview:
        entry = self.entry(name)
        return self._mapped_file.slice(offset=entry.data_offset, size=entry.nbytes)

    def buffer_rows(self, name: str, *, row_offset: int, row_count: int) -> memoryview:
        entry = self.entry(name)
        if len(entry.physical_shape) < 1:
            raise ValueError(f"buffer_rows requires rank >= 1 tensor, got {name}")
        total_rows = int(entry.physical_shape[0])
        if row_offset < 0 or row_count <= 0 or row_offset + row_count > total_rows:
            raise ValueError(
                f"Invalid row slice for {name}: row_offset={row_offset}, row_count={row_count}, total_rows={total_rows}"
            )
        row_nbytes = dtype_nbytes(entry.physical_dtype)
        for dim in entry.physical_shape[1:]:
            row_nbytes *= int(dim)
        full_slice = self.buffer_slice(name)
        byte_start = row_offset * row_nbytes
        byte_end = (row_offset + row_count) * row_nbytes
        return full_slice[byte_start:byte_end]


def open_gguf_mmap(path: str | Path) -> GGUFMmap:
    resolved_path = Path(path)
    tensors, metadata = _load_gguf_index(resolved_path)
    return GGUFMmap(path=resolved_path, tensors=tensors, metadata=metadata)


def _load_gguf_index(path: Path) -> tuple[dict[str, GGUFTensorEntry], dict[str, object]]:
    with path.open("rb") as handle:
        magic = _read_u32(handle)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file: expected magic 0x{GGUF_MAGIC:08x}, got 0x{magic:08x}")
        version = _read_u32(handle)
        if version not in {2, 3}:
            raise ValueError(f"Unsupported GGUF version {version} in {path}")
        tensor_count = _read_u64(handle)
        kv_count = _read_u64(handle)

        metadata: dict[str, object] = {"GGUF.version": version}
        for _ in range(kv_count):
            key = _read_string(handle)
            value_type = GGUFValueType(_read_u32(handle))
            metadata[key] = _read_metadata_value(handle, value_type)

        tensor_descriptors: list[tuple[str, tuple[int, ...], GGUFTensorType, int]] = []
        for _ in range(tensor_count):
            name = _read_string(handle)
            n_dims = _read_u32(handle)
            ggml_dims = tuple(_read_u64(handle) for _ in range(n_dims))
            ggml_type = GGUFTensorType(_read_u32(handle))
            tensor_offset = _read_u64(handle)
            tensor_descriptors.append((name, ggml_dims, ggml_type, tensor_offset))

        alignment = _extract_alignment(metadata)
        data_start_offset = handle.tell()
        padding = data_start_offset % alignment
        if padding != 0:
            data_start_offset += alignment - padding

    tensors: dict[str, GGUFTensorEntry] = {}
    for name, ggml_dims, ggml_type, tensor_offset in tensor_descriptors:
        entry = _build_tensor_entry(
            name=name,
            ggml_dims=ggml_dims,
            ggml_type=ggml_type,
            tensor_offset=tensor_offset,
            data_start_offset=data_start_offset,
        )
        tensors[name] = entry
    return tensors, metadata


def _build_tensor_entry(
    *,
    name: str,
    ggml_dims: tuple[int, ...],
    ggml_type: GGUFTensorType,
    tensor_offset: int,
    data_start_offset: int,
) -> GGUFTensorEntry:
    info = _require_ggml_type_info(ggml_type)
    logical_shape = tuple(reversed(ggml_dims))
    n_elements = 1
    for dim in ggml_dims:
        n_elements *= int(dim)
    nbytes = n_elements * info.type_size // info.block_size
    if info.spec_dtype is not None:
        physical_dtype = info.spec_dtype
        physical_shape = logical_shape
    elif ggml_type is GGUFTensorType.Q4_K:
        physical_dtype = "uint32"
        physical_shape = _quant_shape_to_word_shape(logical_shape, ggml_type)
    elif ggml_type is GGUFTensorType.Q8_0:
        physical_dtype = "uint16"
        physical_shape = _quant_shape_to_halfword_shape(logical_shape, ggml_type)
    elif ggml_type is GGUFTensorType.Q6_K:
        physical_dtype = "uint16"
        physical_shape = _quant_shape_to_halfword_shape(logical_shape, ggml_type)
    else:
        physical_dtype = "uint8"
        physical_shape = _quant_shape_to_byte_shape(logical_shape, ggml_type)
    return GGUFTensorEntry(
        name=name,
        ggml_type=ggml_type,
        ggml_shape=ggml_dims,
        logical_shape=logical_shape,
        physical_dtype=physical_dtype,
        physical_shape=physical_shape,
        data_offset=data_start_offset + tensor_offset,
        nbytes=nbytes,
    )


def _quant_shape_to_byte_shape(shape: tuple[int, ...], ggml_type: GGUFTensorType) -> tuple[int, ...]:
    info = _require_ggml_type_info(ggml_type)
    if shape[-1] % info.block_size != 0:
        raise ValueError(
            f"Quantized tensor row size ({shape[-1]}) is not a multiple of {ggml_type.name} block size ({info.block_size})"
        )
    return (*shape[:-1], shape[-1] // info.block_size * info.type_size)


def _quant_shape_to_word_shape(shape: tuple[int, ...], ggml_type: GGUFTensorType) -> tuple[int, ...]:
    byte_shape = _quant_shape_to_byte_shape(shape, ggml_type)
    if byte_shape[-1] % 4 != 0:
        raise ValueError(f"{ggml_type.name} row byte width must be divisible by 4, got {byte_shape[-1]}")
    return (*byte_shape[:-1], byte_shape[-1] // 4)


def _quant_shape_to_halfword_shape(shape: tuple[int, ...], ggml_type: GGUFTensorType) -> tuple[int, ...]:
    byte_shape = _quant_shape_to_byte_shape(shape, ggml_type)
    if byte_shape[-1] % 2 != 0:
        raise ValueError(f"{ggml_type.name} row byte width must be divisible by 2, got {byte_shape[-1]}")
    return (*byte_shape[:-1], byte_shape[-1] // 2)


def _extract_alignment(metadata: dict[str, object]) -> int:
    raw_alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if not isinstance(raw_alignment, int):
        raise ValueError(f"GGUF general.alignment must be int, got {raw_alignment!r}")
    if raw_alignment == 0 or (raw_alignment & (raw_alignment - 1)) != 0:
        raise ValueError(f"Invalid GGUF alignment {raw_alignment}: must be a non-zero power of two")
    return raw_alignment


def _read_metadata_value(handle: BinaryIO, value_type: GGUFValueType) -> object:
    if value_type is GGUFValueType.UINT8:
        return _read_u8(handle)
    if value_type is GGUFValueType.INT8:
        return _read_i8(handle)
    if value_type is GGUFValueType.UINT16:
        return _read_u16(handle)
    if value_type is GGUFValueType.INT16:
        return _read_i16(handle)
    if value_type is GGUFValueType.UINT32:
        return _read_u32(handle)
    if value_type is GGUFValueType.INT32:
        return _read_i32(handle)
    if value_type is GGUFValueType.FLOAT32:
        return _read_f32(handle)
    if value_type is GGUFValueType.BOOL:
        return bool(_read_u8(handle))
    if value_type is GGUFValueType.STRING:
        return _read_string(handle)
    if value_type is GGUFValueType.ARRAY:
        element_type = GGUFValueType(_read_u32(handle))
        element_count = _read_u64(handle)
        return [_read_metadata_value(handle, element_type) for _ in range(element_count)]
    if value_type is GGUFValueType.UINT64:
        return _read_u64(handle)
    if value_type is GGUFValueType.INT64:
        return _read_i64(handle)
    if value_type is GGUFValueType.FLOAT64:
        return _read_f64(handle)
    raise ValueError(f"Unsupported GGUF value type: {value_type}")


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("Truncated GGUF string value")
    return raw.decode("utf-8")


def _read_u8(handle: BinaryIO) -> int:
    return _unpack_int("<B", handle)


def _read_i8(handle: BinaryIO) -> int:
    return _unpack_int("<b", handle)


def _read_u16(handle: BinaryIO) -> int:
    return _unpack_int("<H", handle)


def _read_i16(handle: BinaryIO) -> int:
    return _unpack_int("<h", handle)


def _read_u32(handle: BinaryIO) -> int:
    return _unpack_int("<I", handle)


def _read_i32(handle: BinaryIO) -> int:
    return _unpack_int("<i", handle)


def _read_u64(handle: BinaryIO) -> int:
    return _unpack_int("<Q", handle)


def _read_i64(handle: BinaryIO) -> int:
    return _unpack_int("<q", handle)


def _read_f32(handle: BinaryIO) -> float:
    return _unpack_float("<f", handle)


def _read_f64(handle: BinaryIO) -> float:
    return _unpack_float("<d", handle)


def _unpack_int(fmt: str, handle: BinaryIO) -> int:
    value = _unpack(fmt, handle)
    if not isinstance(value, int):
        raise ValueError(f"GGUF format {fmt} decoded to non-int value {value!r}")
    return value


def _unpack_float(fmt: str, handle: BinaryIO) -> float:
    value = _unpack(fmt, handle)
    if not isinstance(value, float):
        raise ValueError(f"GGUF format {fmt} decoded to non-float value {value!r}")
    return value


def _unpack(fmt: str, handle: BinaryIO) -> int | float:
    size = struct.calcsize(fmt)
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError(f"Truncated GGUF field for format {fmt}")
    value = struct.unpack(fmt, raw)[0]
    if not isinstance(value, int | float):
        raise ValueError(f"GGUF format {fmt} decoded to unsupported value {value!r}")
    return value


def _require_ggml_type_info(ggml_type: GGUFTensorType) -> _GGMLTypeInfo:
    try:
        return _GGML_TYPE_INFO[ggml_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported GGUF tensor type {ggml_type.name}") from exc
