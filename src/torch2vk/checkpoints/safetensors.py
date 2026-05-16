"""Safetensors checkpoint reader shared across model families."""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from torch2vk.vulkan.types import Residency, TensorSpec, dtype_nbytes

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _DTypeInfo:
    spec_dtype: str
    item_size: int


_DTYPE_INFO: dict[str, _DTypeInfo] = {
    "BF16": _DTypeInfo(spec_dtype="bfloat16", item_size=2),
    "F16": _DTypeInfo(spec_dtype="float16", item_size=2),
    "F32": _DTypeInfo(spec_dtype="float32", item_size=4),
    "I8": _DTypeInfo(spec_dtype="int8", item_size=1),
    "U8": _DTypeInfo(spec_dtype="uint8", item_size=1),
    "I32": _DTypeInfo(spec_dtype="int32", item_size=4),
    "I64": _DTypeInfo(spec_dtype="int64", item_size=8),
}


@dataclass(frozen=True, slots=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    @property
    def item_size(self) -> int:
        return _dtype_info(self.dtype).item_size

    @property
    def nbytes(self) -> int:
        start, end = self.data_offsets
        return int(end) - int(start)

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(
            dtype=_dtype_info(self.dtype).spec_dtype,
            shape=self.shape,
            residency=Residency.HOST,
        )


@dataclass(frozen=True, slots=True)
class _TensorSource:
    shard_path: Path
    data_start_offset: int


@dataclass(slots=True)
class _MappedSafetensorsShard:
    path: Path
    _file: BinaryIO | None
    _mmap: mmap.mmap | None

    @classmethod
    def open(cls, path: Path) -> "_MappedSafetensorsShard":
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
            raise RuntimeError(f"safetensors shard is closed: {self.path}")
        return memoryview(mapped)[offset : offset + size]


class SafetensorsMmap:
    """Mmap-backed safetensors checkpoint storage over one or more shards."""

    def __init__(
        self,
        *,
        path: str | Path,
        tensors: dict[str, TensorEntry],
        metadata: JsonObject,
        tensor_sources: dict[str, _TensorSource],
    ) -> None:
        self.path = Path(path)
        self.tensors = dict(tensors)
        self.metadata = dict(metadata)
        self._tensor_sources = dict(tensor_sources)
        self._shards: dict[Path, _MappedSafetensorsShard] = {}

    def close(self) -> None:
        self.release_mapping()

    def release_mapping(self) -> None:
        shard_errors: list[RuntimeError] = []
        for shard in self._shards.values():
            try:
                shard.close()
            except BufferError:
                shard_errors.append(
                    RuntimeError(
                        f"Could not close safetensors shard for {shard.path}: "
                        "exported checkpoint buffer views still exist"
                    )
                )
        self._shards.clear()
        if shard_errors:
            raise shard_errors[0]

    def __enter__(self) -> "SafetensorsMmap":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def entry(self, name: str) -> TensorEntry:
        try:
            return self.tensors[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.tensors))
            raise KeyError(f"Unknown safetensors entry {name!r}. Available: {known}") from exc

    def buffer_slice(self, name: str) -> memoryview:
        entry = self.entry(name)
        source = self._tensor_source(name)
        shard = self._require_shard(source.shard_path)
        start, _ = entry.data_offsets
        return shard.slice(offset=source.data_start_offset + int(start), size=entry.nbytes)

    def buffer_rows(self, name: str, *, row_offset: int, row_count: int) -> memoryview:
        entry = self.entry(name)
        if len(entry.shape) < 1:
            raise ValueError(f"buffer_rows requires a tensor with at least one dimension, got {name}")
        total_rows = int(entry.shape[0])
        if row_offset < 0 or row_count <= 0 or row_offset + row_count > total_rows:
            raise ValueError(
                f"Invalid row slice for {name}: row_offset={row_offset}, row_count={row_count}, total_rows={total_rows}"
            )
        row_nbytes = dtype_nbytes(entry.spec.dtype)
        for dim in entry.shape[1:]:
            row_nbytes *= int(dim)
        full_slice = self.buffer_slice(name)
        byte_start = row_offset * row_nbytes
        byte_end = (row_offset + row_count) * row_nbytes
        return full_slice[byte_start:byte_end]

    def _tensor_source(self, name: str) -> _TensorSource:
        try:
            return self._tensor_sources[name]
        except KeyError as exc:
            raise KeyError(f"Missing safetensors shard mapping for entry {name!r}") from exc

    def _require_shard(self, shard_path: Path) -> _MappedSafetensorsShard:
        shard = self._shards.get(shard_path)
        if shard is None:
            shard = _MappedSafetensorsShard.open(shard_path)
            self._shards[shard_path] = shard
        return shard


def open_safetensors_mmap(path: str | Path) -> SafetensorsMmap:
    resolved_path = Path(path)
    if resolved_path.suffix == ".json":
        tensors, metadata, tensor_sources = load_safetensors_weight_map(resolved_path)
    else:
        tensors, metadata, data_start_offset = load_safetensors_index(resolved_path)
        tensor_sources = {
            name: _TensorSource(shard_path=resolved_path, data_start_offset=data_start_offset)
            for name in tensors
        }
    return SafetensorsMmap(path=resolved_path, tensors=tensors, metadata=metadata, tensor_sources=tensor_sources)


def load_safetensors_index(path: str | Path) -> tuple[dict[str, TensorEntry], JsonObject, int]:
    resolved_path = Path(path)
    with resolved_path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"{resolved_path} is too small to be a safetensors file")
        (header_size,) = struct.unpack("<Q", header_size_bytes)
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{resolved_path} has a truncated safetensors header")

    raw_header = _require_json_object(json.loads(header_bytes.decode("utf-8")), context=str(resolved_path))

    raw_metadata = raw_header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{resolved_path} must contain object field '__metadata__', got {raw_metadata!r}")
    metadata: JsonObject = {}
    for key, value in raw_metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"{resolved_path} metadata must contain only string keys, got {key!r}")
        metadata[key] = _require_json_value(value, context=f"{resolved_path}.__metadata__.{key}")
    data_start_offset = 8 + int(header_size)

    tensors: dict[str, TensorEntry] = {}
    for name, entry in raw_header.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{resolved_path}:{name} must decode to an object entry, got {entry!r}")
        dtype_value = entry.get("dtype")
        shape_value = entry.get("shape")
        offsets_value = entry.get("data_offsets")
        if not isinstance(dtype_value, str):
            raise ValueError(f"{resolved_path}:{name} must contain string field 'dtype', got {dtype_value!r}")
        if not isinstance(shape_value, list):
            raise ValueError(f"{resolved_path}:{name} must contain list[int] field 'shape', got {shape_value!r}")
        if not isinstance(offsets_value, list) or len(offsets_value) != 2:
            raise ValueError(
                f"{resolved_path}:{name} must contain two-element list[int] field 'data_offsets', got {offsets_value!r}"
            )
        tensor_entry = TensorEntry(
            name=name,
            dtype=dtype_value,
            shape=tuple(_required_int_value(dim, context=f"{resolved_path}:{name}.shape") for dim in shape_value),
            data_offsets=(
                _required_int_value(offsets_value[0], context=f"{resolved_path}:{name}.data_offsets[0]"),
                _required_int_value(offsets_value[1], context=f"{resolved_path}:{name}.data_offsets[1]"),
            ),
        )
        if tensor_entry.nbytes != tensor_entry.numel * tensor_entry.item_size:
            raise ValueError(
                f"{resolved_path}:{name} has inconsistent data_offsets for dtype={tensor_entry.dtype} "
                f"shape={tensor_entry.shape}"
            )
        tensors[name] = tensor_entry

    return tensors, metadata, data_start_offset


def _dtype_info(dtype: str) -> _DTypeInfo:
    try:
        return _DTYPE_INFO[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}") from exc


def load_safetensors_weight_map(
    path: str | Path,
) -> tuple[dict[str, TensorEntry], JsonObject, dict[str, _TensorSource]]:
    resolved_path = Path(path)
    decoded = _require_json_object(json.loads(resolved_path.read_text(encoding="utf-8")), context=str(resolved_path))

    metadata_value = decoded.get("metadata", {})
    if not isinstance(metadata_value, dict):
        raise ValueError(f"{resolved_path} must contain object field 'metadata'")
    metadata: JsonObject = {}
    for key, value in metadata_value.items():
        if not isinstance(key, str):
            raise ValueError(f"{resolved_path} metadata must contain only string keys, got {key!r}")
        metadata[key] = _require_json_value(value, context=f"{resolved_path}.metadata.{key}")

    weight_map = decoded.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{resolved_path} must contain object field 'weight_map'")

    shard_headers: dict[Path, tuple[dict[str, TensorEntry], JsonObject, int]] = {}
    tensors: dict[str, TensorEntry] = {}
    tensor_sources: dict[str, _TensorSource] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str):
            raise ValueError(f"{resolved_path} has invalid weight_map entry: {name!r} -> {shard_name!r}")
        shard_path = (resolved_path.parent / shard_name).resolve()
        shard_tensors, _, data_start_offset = shard_headers.setdefault(
            shard_path,
            load_safetensors_index(shard_path),
        )
        try:
            entry = shard_tensors[name]
        except KeyError as exc:
            raise KeyError(
                f"{resolved_path} maps tensor {name!r} to shard {shard_path}, but the shard does not contain it"
            ) from exc
        tensors[name] = entry
        tensor_sources[name] = _TensorSource(shard_path=shard_path, data_start_offset=data_start_offset)

    return tensors, metadata, tensor_sources


def _required_int_value(value: JsonValue, *, context: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{context} must be an integer, got {value!r}")
    return int(value)


def _require_json_object(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must decode to an object, got {value!r}")
    parsed: JsonObject = {}
    for key, item_value in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} must contain only string object keys, got {key!r}")
        parsed[key] = _require_json_value(item_value, context=f"{context}.{key}")
    return parsed


def _require_json_value(value: object, *, context: str) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [_require_json_value(item, context=f"{context}[]") for item in value]
    if isinstance(value, dict):
        return _require_json_object(value, context=context)
    raise ValueError(f"{context} contains unsupported JSON value {value!r}")
