"""Minimal typed parser for RGP captures used by the local SQTT profiler."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

SQTT_FILE_MAGIC_NUMBER = 0x50303042
ASIC_INFO_CHUNK_TYPE = 0
SQTT_DATA_CHUNK_TYPE = 2
DERIVED_SPM_DB_CHUNK_TYPE = 128

ASIC_INFO_CHUNK_SIZE = 768
DERIVED_SPM_HEADER_SIZE = 44
DERIVED_SPM_GROUP_INFO_SIZE = 20
DERIVED_SPM_COUNTER_INFO_SIZE = 24
DERIVED_SPM_COMPONENT_INFO_SIZE = 20


@dataclass(frozen=True, slots=True)
class _RgpChunkHeader:
    offset: int
    type_id: int
    size_in_bytes: int


@dataclass(frozen=True, slots=True)
class RgpSqttDataChunk:
    payload_offset: int
    payload_end: int


@dataclass(frozen=True, slots=True)
class RgpAsicInfo:
    gpu_name: str
    gpu_type: int
    gfxip_level: int
    shader_engines: int
    compute_unit_per_shader_engine: int
    simd_per_compute_unit: int
    vram_size_bytes: int
    vram_bus_width_bits: int
    l2_cache_size_bytes: int
    l1_cache_size_bytes: int
    gl1_cache_size_bytes: int
    instruction_cache_size_bytes: int
    scalar_cache_size_bytes: int
    mall_cache_size_bytes: int
    trace_shader_core_clock_hz: int
    trace_memory_clock_hz: int
    max_shader_core_clock_hz: int
    max_memory_clock_hz: int
    gpu_timestamp_frequency_hz: int
    memory_ops_per_clock: int
    memory_chip_type: int


@dataclass(frozen=True, slots=True)
class RgpDerivedSpmGroup:
    name: str
    counter_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RgpDerivedSpmComponent:
    name: str
    usage_type: int


@dataclass(frozen=True, slots=True)
class RgpDerivedSpmCounter:
    name: str
    description: str
    group_name: str
    usage_type: int
    component_ids: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RgpDerivedSpmChunk:
    timestamps: tuple[int, ...]
    sample_interval: int
    groups: tuple[RgpDerivedSpmGroup, ...]
    counters: tuple[RgpDerivedSpmCounter, ...]
    components: tuple[RgpDerivedSpmComponent, ...]

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True, slots=True)
class RgpCapture:
    blob: bytes
    asic_info: RgpAsicInfo | None
    sqtt_data_chunks: tuple[RgpSqttDataChunk, ...]
    derived_spm_chunks: tuple[RgpDerivedSpmChunk, ...]


def _parse_chunk_offset(blob: bytes) -> int:
    if len(blob) < 56:
        raise ValueError("file too small for sqtt file header")
    fields = struct.unpack_from("<IIIIiiiiiiiiii", blob, 0)
    magic, _version_major, _version_minor, _flags, chunk_offset, *_timestamp = fields
    if magic != SQTT_FILE_MAGIC_NUMBER:
        raise ValueError(f"invalid RGP magic 0x{magic:08x}")
    return int(chunk_offset)


def _parse_chunk_header(blob: bytes, offset: int) -> _RgpChunkHeader:
    chunk_id_raw, _minor_version, _major_version, size_in_bytes, _ = struct.unpack_from("<IHHii", blob, offset)
    type_id = chunk_id_raw & 0xFF
    return _RgpChunkHeader(
        offset=offset,
        type_id=type_id,
        size_in_bytes=size_in_bytes,
    )


def _parse_sqtt_data(blob: bytes, chunk: _RgpChunkHeader) -> RgpSqttDataChunk:
    body_offset = chunk.offset + 16
    offset, size = struct.unpack_from("<ii", blob, body_offset)
    return RgpSqttDataChunk(payload_offset=offset, payload_end=offset + size)


def _parse_asic_info(blob: bytes, chunk: _RgpChunkHeader) -> RgpAsicInfo:
    if chunk.size_in_bytes < ASIC_INFO_CHUNK_SIZE:
        raise ValueError(f"ASIC info chunk too small: {chunk.size_in_bytes}")
    base = chunk.offset
    gpu_name = _read_ascii(blob, base + 152, 256)
    return RgpAsicInfo(
        gpu_name=gpu_name,
        gpu_type=_unpack_int(blob, base + 92),
        gfxip_level=_unpack_int(blob, base + 96),
        shader_engines=_unpack_int(blob, base + 56),
        compute_unit_per_shader_engine=_unpack_int(blob, base + 60),
        simd_per_compute_unit=_unpack_int(blob, base + 64),
        vram_size_bytes=_unpack_int64(blob, base + 128),
        vram_bus_width_bits=_unpack_int(blob, base + 136),
        l2_cache_size_bytes=_unpack_int(blob, base + 140),
        l1_cache_size_bytes=_unpack_int(blob, base + 144),
        gl1_cache_size_bytes=_unpack_uint(blob, base + 588),
        instruction_cache_size_bytes=_unpack_uint(blob, base + 592),
        scalar_cache_size_bytes=_unpack_uint(blob, base + 596),
        mall_cache_size_bytes=_unpack_uint(blob, base + 600),
        trace_shader_core_clock_hz=_unpack_uint64(blob, base + 24),
        trace_memory_clock_hz=_unpack_uint64(blob, base + 32),
        max_shader_core_clock_hz=_unpack_uint64(blob, base + 432),
        max_memory_clock_hz=_unpack_uint64(blob, base + 440),
        gpu_timestamp_frequency_hz=_unpack_uint64(blob, base + 424),
        memory_ops_per_clock=_unpack_uint(blob, base + 448),
        memory_chip_type=_unpack_uint(blob, base + 452),
    )


def _parse_derived_spm(blob: bytes, chunk: _RgpChunkHeader) -> RgpDerivedSpmChunk:
    if chunk.size_in_bytes < DERIVED_SPM_HEADER_SIZE:
        raise ValueError(f"Derived SPM chunk too small: {chunk.size_in_bytes}")
    chunk_end = chunk.offset + chunk.size_in_bytes
    (
        _chunk_id,
        _minor_version,
        _major_version,
        _size_in_bytes,
        _reserved,
        data_offset,
        _flags,
        num_timestamps,
        num_groups,
        num_counters,
        num_components,
        sample_interval,
    ) = struct.unpack_from("<IHHiiIIIIIII", blob, chunk.offset)
    pos = chunk.offset + data_offset
    _require_range(pos, num_timestamps * 8, chunk_end, "Derived SPM timestamps")
    timestamps = struct.unpack_from(f"<{num_timestamps}Q", blob, pos)
    pos += num_timestamps * 8

    groups: list[RgpDerivedSpmGroup] = []
    counter_to_group: dict[int, str] = {}
    for _ in range(num_groups):
        _require_range(pos, DERIVED_SPM_GROUP_INFO_SIZE, chunk_end, "Derived SPM group")
        size_in_bytes, offset, name_length, description_length, num_group_counters = (
            struct.unpack_from("<IIIII", blob, pos)
        )
        item_end = pos + size_in_bytes
        _require_range(pos, size_in_bytes, chunk_end, "Derived SPM group payload")
        name_start = pos + offset
        name = _read_ascii(blob, name_start, name_length)
        counter_ids_start = name_start + name_length + description_length
        _require_range(counter_ids_start, num_group_counters * 4, item_end, "Derived SPM group counters")
        counter_ids = struct.unpack_from(f"<{num_group_counters}I", blob, counter_ids_start)
        for counter_id in counter_ids:
            counter_to_group[counter_id] = name
        groups.append(RgpDerivedSpmGroup(name=name, counter_ids=tuple(counter_ids)))
        pos = item_end

    counter_headers: list[tuple[str, str, int, tuple[int, ...]]] = []
    for _ in range(num_counters):
        _require_range(pos, DERIVED_SPM_COUNTER_INFO_SIZE, chunk_end, "Derived SPM counter")
        size_in_bytes, offset, name_length, description_length, num_counter_components, usage_type = (
            struct.unpack_from("<IIIIIB", blob, pos)
        )
        item_end = pos + size_in_bytes
        _require_range(pos, size_in_bytes, chunk_end, "Derived SPM counter payload")
        name_start = pos + offset
        name = _read_ascii(blob, name_start, name_length)
        description_start = name_start + name_length
        description = _read_ascii(blob, description_start, description_length)
        component_ids_start = description_start + description_length
        _require_range(
            component_ids_start,
            num_counter_components * 4,
            item_end,
            "Derived SPM counter components",
        )
        component_ids = struct.unpack_from(
            f"<{num_counter_components}I",
            blob,
            component_ids_start,
        )
        counter_headers.append((name, description, usage_type, tuple(component_ids)))
        pos = item_end

    components: list[RgpDerivedSpmComponent] = []
    for _ in range(num_components):
        _require_range(pos, DERIVED_SPM_COMPONENT_INFO_SIZE, chunk_end, "Derived SPM component")
        size_in_bytes, offset, name_length, _description_length, usage_type = (
            struct.unpack_from("<IIIII", blob, pos)
        )
        item_end = pos + size_in_bytes
        _require_range(pos, size_in_bytes, chunk_end, "Derived SPM component payload")
        name = _read_ascii(blob, pos + offset, name_length)
        components.append(RgpDerivedSpmComponent(name=name, usage_type=usage_type))
        pos = item_end

    counters: list[RgpDerivedSpmCounter] = []
    values_nbytes = num_timestamps * 8
    for counter_id, (name, description, usage_type, component_ids) in enumerate(counter_headers):
        _require_range(pos, values_nbytes, chunk_end, "Derived SPM counter values")
        values = struct.unpack_from(f"<{num_timestamps}d", blob, pos)
        counters.append(
            RgpDerivedSpmCounter(
                name=name,
                description=description,
                group_name=counter_to_group.get(counter_id, ""),
                usage_type=usage_type,
                component_ids=component_ids,
                values=tuple(values),
            )
        )
        pos += values_nbytes

    return RgpDerivedSpmChunk(
        timestamps=tuple(timestamps),
        sample_interval=sample_interval,
        groups=tuple(groups),
        counters=tuple(counters),
        components=tuple(components),
    )


def parse_rgp_capture(path: Path) -> RgpCapture:
    blob = path.read_bytes()
    asic_info: RgpAsicInfo | None = None
    sqtt_data_chunks: list[RgpSqttDataChunk] = []
    derived_spm_chunks: list[RgpDerivedSpmChunk] = []
    offset = _parse_chunk_offset(blob)
    while offset + 16 <= len(blob):
        chunk = _parse_chunk_header(blob, offset)
        if chunk.size_in_bytes <= 0:
            break
        if chunk.type_id == ASIC_INFO_CHUNK_TYPE:
            asic_info = _parse_asic_info(blob, chunk)
        elif chunk.type_id == SQTT_DATA_CHUNK_TYPE:
            sqtt_data_chunks.append(_parse_sqtt_data(blob, chunk))
        elif chunk.type_id == DERIVED_SPM_DB_CHUNK_TYPE:
            derived_spm_chunks.append(_parse_derived_spm(blob, chunk))
        offset += chunk.size_in_bytes
    return RgpCapture(
        blob=blob,
        asic_info=asic_info,
        sqtt_data_chunks=tuple(sqtt_data_chunks),
        derived_spm_chunks=tuple(derived_spm_chunks),
    )


def _unpack_uint(blob: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", blob, offset)[0])


def _unpack_int(blob: bytes, offset: int) -> int:
    return int(struct.unpack_from("<i", blob, offset)[0])


def _unpack_uint64(blob: bytes, offset: int) -> int:
    return int(struct.unpack_from("<Q", blob, offset)[0])


def _unpack_int64(blob: bytes, offset: int) -> int:
    return int(struct.unpack_from("<q", blob, offset)[0])


def _read_ascii(blob: bytes, offset: int, length: int) -> str:
    raw = blob[offset:offset + length]
    raw = raw.split(b"\0", 1)[0]
    return raw.decode("utf-8", errors="replace")


def _require_range(offset: int, size: int, end: int, name: str) -> None:
    if size < 0 or offset < 0 or offset + size > end:
        raise ValueError(f"{name} outside chunk bounds")
