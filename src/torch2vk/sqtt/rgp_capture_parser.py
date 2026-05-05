"""Minimal typed parser for RGP captures used by the local SQTT profiler."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

SQTT_FILE_MAGIC_NUMBER = 0x50303042
SQTT_DATA_CHUNK_TYPE = 2


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
class RgpCapture:
    blob: bytes
    sqtt_data_chunks: tuple[RgpSqttDataChunk, ...]


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


def parse_rgp_capture(path: Path) -> RgpCapture:
    blob = path.read_bytes()
    sqtt_data_chunks: list[RgpSqttDataChunk] = []
    offset = _parse_chunk_offset(blob)
    while offset + 16 <= len(blob):
        chunk = _parse_chunk_header(blob, offset)
        if chunk.size_in_bytes <= 0:
            break
        if chunk.type_id == SQTT_DATA_CHUNK_TYPE:
            sqtt_data_chunks.append(_parse_sqtt_data(blob, chunk))
        offset += chunk.size_in_bytes
    return RgpCapture(
        blob=blob,
        sqtt_data_chunks=tuple(sqtt_data_chunks),
    )
