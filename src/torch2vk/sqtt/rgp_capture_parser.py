"""Minimal typed parser for RGP captures used by the local SQTT profiler."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .amdgpu_metadata import CodeObjectMetadata, parse_code_object_elf_layout


SQTT_FILE_MAGIC_NUMBER = 0x50303042

CHUNK_TYPE_NAMES = {
    0: "ASIC_INFO",
    1: "SQTT_DESC",
    2: "SQTT_DATA",
    3: "API_INFO",
    4: "RESERVED",
    5: "QUEUE_EVENT_TIMINGS",
    6: "CLOCK_CALIBRATION",
    7: "CPU_INFO",
    8: "SPM_DB",
    9: "CODE_OBJECT_DATABASE",
    10: "CODE_OBJECT_LOADER_EVENTS",
    11: "PSO_CORRELATION",
    12: "INSTRUMENTATION_TABLE",
    128: "DERIVED_SPM_DB",
}

QUEUE_EVENT_TYPE_NAMES = {
    0: "CMDBUF_SUBMIT",
    1: "SIGNAL_SEMAPHORE",
    2: "WAIT_SEMAPHORE",
    3: "PRESENT",
}

QUEUE_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "UNIVERSAL",
    2: "COMPUTE",
    3: "DMA",
}

ENGINE_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "UNIVERSAL",
    2: "COMPUTE",
    3: "EXCLUSIVE_COMPUTE",
    4: "DMA",
    7: "HIGH_PRIORITY_UNIVERSAL",
    8: "HIGH_PRIORITY_GRAPHICS",
}


@dataclass(frozen=True, slots=True)
class RgpTimestamp:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


@dataclass(frozen=True, slots=True)
class RgpFileHeader:
    magic: int
    version_major: int
    version_minor: int
    flags: int
    chunk_offset: int
    timestamp: RgpTimestamp

    @property
    def valid_magic(self) -> bool:
        return self.magic == SQTT_FILE_MAGIC_NUMBER


@dataclass(frozen=True, slots=True)
class RgpChunkHeader:
    offset: int
    type_id: int
    type_name: str
    index: int
    major_version: int
    minor_version: int
    size_in_bytes: int


@dataclass(frozen=True, slots=True)
class RgpQueueInfo:
    queue_id: int
    queue_context: int
    queue_type_name: str
    engine_type_name: str


@dataclass(frozen=True, slots=True)
class RgpQueueEvent:
    event_name: str
    frame_index: int
    queue_info_index: int
    submit_sub_index: int
    gpu_timestamp_start: int
    gpu_timestamp_end: int

    @property
    def gpu_duration(self) -> int:
        return self.gpu_timestamp_end - self.gpu_timestamp_start


@dataclass(frozen=True, slots=True)
class RgpQueueEventChunk:
    queue_infos: tuple[RgpQueueInfo, ...]
    queue_events: tuple[RgpQueueEvent, ...]


@dataclass(frozen=True, slots=True)
class ParsedElfEnvelope:
    valid_elf: bool
    required_size: int


@dataclass(frozen=True, slots=True)
class RgpCodeObjectRecord:
    index: int
    payload_offset: int
    payload_size: int
    embedded_strings: tuple[str, ...]
    elf: ParsedElfEnvelope
    metadata: CodeObjectMetadata
    executable_virtual_address_end: int


@dataclass(frozen=True, slots=True)
class RgpLoaderEvent:
    index: int
    loader_event_type: int
    base_address: int
    code_object_hash: tuple[int, int]
    timestamp: int


@dataclass(frozen=True, slots=True)
class RgpPsoCorrelation:
    index: int
    api_pso_hash: int
    pipeline_hash: tuple[int, int]
    api_level_object_name: str


@dataclass(frozen=True, slots=True)
class RgpSqttDesc:
    shader_engine_index: int
    compute_unit_index: int
    sqtt_version: int
    instrumentation_spec_version: int
    instrumentation_api_version: int


@dataclass(frozen=True, slots=True)
class RgpSqttDataChunk:
    offset: int
    size: int
    payload_offset: int
    payload_end: int


@dataclass(frozen=True, slots=True)
class RgpCapture:
    path: Path
    blob: bytes
    header: RgpFileHeader
    chunks: tuple[RgpChunkHeader, ...]
    queue_event_chunks: tuple[RgpQueueEventChunk, ...]
    code_object_records: tuple[RgpCodeObjectRecord, ...]
    loader_events: tuple[RgpLoaderEvent, ...]
    pso_correlations: tuple[RgpPsoCorrelation, ...]
    sqtt_descs: tuple[RgpSqttDesc, ...]
    sqtt_data_chunks: tuple[RgpSqttDataChunk, ...]


def _read_c_string(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _extract_strings(blob: bytes, min_length: int = 4, limit: int = 20) -> tuple[str, ...]:
    strings: list[str] = []
    current = bytearray()
    for byte in blob:
        if 32 <= byte <= 126:
            current.append(byte)
            continue
        if len(current) >= min_length:
            strings.append(current.decode("ascii", errors="replace"))
            if len(strings) >= limit:
                return tuple(strings)
        current.clear()
    if len(current) >= min_length and len(strings) < limit:
        strings.append(current.decode("ascii", errors="replace"))
    return tuple(strings)


def _probe_elf_required_size(payload: bytes) -> ParsedElfEnvelope:
    if len(payload) < 64 or payload[:4] != b"\x7fELF":
        return ParsedElfEnvelope(valid_elf=False, required_size=len(payload))
    if payload[4] != 2 or payload[5] != 1:
        return ParsedElfEnvelope(valid_elf=False, required_size=len(payload))
    _, _, _, _, _, e_shoff, _, _, _, _, e_shentsize, e_shnum, _ = struct.unpack_from(
        "<HHIQQQIHHHHHH",
        payload,
        16,
    )
    required_size = int(e_shoff + e_shentsize * e_shnum)
    return ParsedElfEnvelope(valid_elf=True, required_size=max(len(payload), required_size))


def materialize_code_object_payload(capture: RgpCapture, record: RgpCodeObjectRecord) -> bytes:
    payload = bytearray(capture.blob[record.payload_offset : record.payload_offset + record.payload_size])
    if record.elf.valid_elf and len(payload) < record.elf.required_size:
        payload.extend(b"\x00" * (record.elf.required_size - len(payload)))
    return bytes(payload)


def _parse_file_header(blob: bytes) -> RgpFileHeader:
    if len(blob) < 56:
        raise ValueError("file too small for sqtt file header")
    fields = struct.unpack_from("<IIIIiiiiiiiiii", blob, 0)
    magic, version_major, version_minor, flags, chunk_offset, second, minute, hour, day, month, year, _, _, _ = fields
    return RgpFileHeader(
        magic=magic,
        version_major=version_major,
        version_minor=version_minor,
        flags=flags,
        chunk_offset=chunk_offset,
        timestamp=RgpTimestamp(
            year=year + 1900,
            month=month + 1,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        ),
    )


def _parse_chunk_header(blob: bytes, offset: int) -> RgpChunkHeader:
    chunk_id_raw, minor_version, major_version, size_in_bytes, _ = struct.unpack_from("<IHHii", blob, offset)
    type_id = chunk_id_raw & 0xFF
    index = (chunk_id_raw >> 8) & 0xFF
    return RgpChunkHeader(
        offset=offset,
        type_id=type_id,
        type_name=CHUNK_TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}"),
        index=index,
        major_version=major_version,
        minor_version=minor_version,
        size_in_bytes=size_in_bytes,
    )


def _parse_queue_event_chunk(blob: bytes, chunk: RgpChunkHeader) -> RgpQueueEventChunk:
    body_offset = chunk.offset + 16
    queue_info_record_count, _, queue_event_record_count, _ = struct.unpack_from("<IIII", blob, body_offset)
    cursor = body_offset + 16
    queue_infos: list[RgpQueueInfo] = []
    for _ in range(queue_info_record_count):
        queue_id, queue_context, hardware_info, _reserved = struct.unpack_from("<QQII", blob, cursor)
        queue_type = hardware_info & 0xFF
        engine_type = (hardware_info >> 8) & 0xFF
        queue_infos.append(
            RgpQueueInfo(
                queue_id=queue_id,
                queue_context=queue_context,
                queue_type_name=QUEUE_TYPE_NAMES.get(queue_type, f"UNKNOWN_{queue_type}"),
                engine_type_name=ENGINE_TYPE_NAMES.get(engine_type, f"UNKNOWN_{engine_type}"),
            )
        )
        cursor += 24
    queue_events: list[RgpQueueEvent] = []
    for _ in range(queue_event_record_count):
        event_type, _sqtt_cb_id, frame_index, queue_info_index, submit_sub_index, _api_id, _cpu_timestamp, gpu0, gpu1 = struct.unpack_from(
            "<IIQIIQQQQ",
            blob,
            cursor,
        )
        queue_events.append(
            RgpQueueEvent(
                event_name=QUEUE_EVENT_TYPE_NAMES.get(event_type, f"UNKNOWN_{event_type}"),
                frame_index=frame_index,
                queue_info_index=queue_info_index,
                submit_sub_index=submit_sub_index,
                gpu_timestamp_start=gpu0,
                gpu_timestamp_end=gpu1,
            )
        )
        cursor += 56
    return RgpQueueEventChunk(queue_infos=tuple(queue_infos), queue_events=tuple(queue_events))


def _parse_code_object_records(blob: bytes, chunk: RgpChunkHeader) -> tuple[RgpCodeObjectRecord, ...]:
    body_offset = chunk.offset + 16
    _offset, _flags, size, record_count = struct.unpack_from("<IIII", blob, body_offset)
    payload_start = body_offset + 16
    payload_end = min(chunk.offset + chunk.size_in_bytes, payload_start + size)
    cursor = payload_start
    records: list[RgpCodeObjectRecord] = []
    for index in range(record_count):
        if cursor + 4 > payload_end:
            break
        (payload_size,) = struct.unpack_from("<I", blob, cursor)
        if payload_size == 0:
            break
        record_end = cursor + 4 + payload_size
        if record_end > payload_end:
            break
        payload = blob[cursor + 4 : record_end]
        elf = _probe_elf_required_size(payload)
        materialized_payload = payload
        if elf.valid_elf and len(materialized_payload) < elf.required_size:
            materialized_payload = materialized_payload + (b"\x00" * (elf.required_size - len(materialized_payload)))
        elf_layout = parse_code_object_elf_layout(materialized_payload)
        records.append(
            RgpCodeObjectRecord(
                index=index,
                payload_offset=cursor + 4,
                payload_size=payload_size,
                embedded_strings=_extract_strings(payload),
                elf=elf,
                metadata=elf_layout.metadata,
                executable_virtual_address_end=elf_layout.executable_virtual_address_end,
            )
        )
        cursor = record_end
    return tuple(records)


def _parse_loader_events(blob: bytes, chunk: RgpChunkHeader) -> tuple[RgpLoaderEvent, ...]:
    body_offset = chunk.offset + 16
    _offset, _flags, record_size, record_count = struct.unpack_from("<IIII", blob, body_offset)
    cursor = body_offset + 16
    records: list[RgpLoaderEvent] = []
    for index in range(record_count):
        if record_size < 40 or cursor + record_size > chunk.offset + chunk.size_in_bytes:
            break
        loader_event_type, _reserved, base_address, hash_lo, hash_hi, timestamp = struct.unpack_from(
            "<IIQQQQ",
            blob,
            cursor,
        )
        records.append(
            RgpLoaderEvent(
                index=index,
                loader_event_type=loader_event_type,
                base_address=base_address,
                code_object_hash=(hash_lo, hash_hi),
                timestamp=timestamp,
            )
        )
        cursor += record_size
    return tuple(records)


def _parse_pso_correlations(blob: bytes, chunk: RgpChunkHeader) -> tuple[RgpPsoCorrelation, ...]:
    body_offset = chunk.offset + 16
    _offset, _flags, record_size, record_count = struct.unpack_from("<IIII", blob, body_offset)
    cursor = body_offset + 16
    records: list[RgpPsoCorrelation] = []
    for index in range(record_count):
        if record_size < 88 or cursor + record_size > chunk.offset + chunk.size_in_bytes:
            break
        api_pso_hash, hash_lo, hash_hi = struct.unpack_from("<QQQ", blob, cursor)
        name = _read_c_string(blob[cursor + 24 : cursor + 88])
        records.append(
            RgpPsoCorrelation(
                index=index,
                api_pso_hash=api_pso_hash,
                pipeline_hash=(hash_lo, hash_hi),
                api_level_object_name=name,
            )
        )
        cursor += record_size
    return tuple(records)


def _parse_sqtt_desc(blob: bytes, chunk: RgpChunkHeader) -> RgpSqttDesc:
    body_offset = chunk.offset + 16
    shader_engine_index, sqtt_version, spec_version, api_version, compute_unit_index = struct.unpack_from(
        "<iihhi",
        blob,
        body_offset,
    )
    return RgpSqttDesc(
        shader_engine_index=shader_engine_index,
        compute_unit_index=compute_unit_index,
        sqtt_version=sqtt_version,
        instrumentation_spec_version=spec_version,
        instrumentation_api_version=api_version,
    )


def _parse_sqtt_data(blob: bytes, chunk: RgpChunkHeader) -> RgpSqttDataChunk:
    body_offset = chunk.offset + 16
    offset, size = struct.unpack_from("<ii", blob, body_offset)
    return RgpSqttDataChunk(offset=offset, size=size, payload_offset=offset, payload_end=offset + size)


def parse_rgp_capture(path: Path) -> RgpCapture:
    blob = path.read_bytes()
    header = _parse_file_header(blob)
    chunks: list[RgpChunkHeader] = []
    queue_event_chunks: list[RgpQueueEventChunk] = []
    code_object_records: list[RgpCodeObjectRecord] = []
    loader_events: list[RgpLoaderEvent] = []
    pso_correlations: list[RgpPsoCorrelation] = []
    sqtt_descs: list[RgpSqttDesc] = []
    sqtt_data_chunks: list[RgpSqttDataChunk] = []
    offset = int(header.chunk_offset)
    while offset + 16 <= len(blob):
        chunk = _parse_chunk_header(blob, offset)
        if chunk.size_in_bytes <= 0:
            break
        chunks.append(chunk)
        if chunk.type_name == "QUEUE_EVENT_TIMINGS":
            queue_event_chunks.append(_parse_queue_event_chunk(blob, chunk))
        elif chunk.type_name == "CODE_OBJECT_DATABASE":
            code_object_records.extend(_parse_code_object_records(blob, chunk))
        elif chunk.type_name == "CODE_OBJECT_LOADER_EVENTS":
            loader_events.extend(_parse_loader_events(blob, chunk))
        elif chunk.type_name == "PSO_CORRELATION":
            pso_correlations.extend(_parse_pso_correlations(blob, chunk))
        elif chunk.type_name == "SQTT_DESC":
            sqtt_descs.append(_parse_sqtt_desc(blob, chunk))
        elif chunk.type_name == "SQTT_DATA":
            sqtt_data_chunks.append(_parse_sqtt_data(blob, chunk))
        offset += chunk.size_in_bytes
    return RgpCapture(
        path=path,
        blob=blob,
        header=header,
        chunks=tuple(chunks),
        queue_event_chunks=tuple(queue_event_chunks),
        code_object_records=tuple(code_object_records),
        loader_events=tuple(loader_events),
        pso_correlations=tuple(pso_correlations),
        sqtt_descs=tuple(sqtt_descs),
        sqtt_data_chunks=tuple(sqtt_data_chunks),
    )
