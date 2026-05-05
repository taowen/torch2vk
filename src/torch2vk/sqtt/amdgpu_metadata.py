"""Structured AMDGPU metadata parsing from RGP code-object ELF notes."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TypeGuard

ElfSectionHeader = tuple[int, int, int, int, int, int, int, int, int, int]
ElfHeader = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class CodeObjectMetadata:
    entry_point: str | None
    api: str | None
    internal_pipeline_hash: tuple[int, int] | None
    api_shader_hash: tuple[int, int] | None
    lds_size: int | None
    scratch_memory_size: int | None
    sgpr_count: int | None
    vgpr_count: int | None
    wavefront_size: int | None
    spill_threshold: int | None
    user_data_limit: int | None


@dataclass(frozen=True, slots=True)
class CodeObjectElfLayout:
    metadata: CodeObjectMetadata
    executable_virtual_address_end: int


def parse_code_object_metadata(payload: bytes) -> CodeObjectMetadata:
    return parse_code_object_elf_layout(payload).metadata


def parse_code_object_elf_layout(payload: bytes) -> CodeObjectElfLayout:
    note_payload = _find_amdgpu_metadata_note(payload)
    executable_virtual_address_end = _find_executable_virtual_address_end(payload)
    if note_payload is None:
        return CodeObjectElfLayout(
            metadata=CodeObjectMetadata(
                entry_point=None,
                api=None,
                internal_pipeline_hash=None,
                api_shader_hash=None,
                lds_size=None,
                scratch_memory_size=None,
                sgpr_count=None,
                vgpr_count=None,
                wavefront_size=None,
                spill_threshold=None,
                user_data_limit=None,
            ),
            executable_virtual_address_end=executable_virtual_address_end,
        )
    decoded = decode_first_msgpack_object(note_payload)
    if not _is_object_map(decoded):
        raise RuntimeError("AMDGPU metadata note must decode to a string-keyed mapping")
    pipelines = decoded.get("amdpal.pipelines")
    pipeline: dict[str, object] = {}
    if _is_object_list(pipelines):
        first_pipeline: object = pipelines[0] if pipelines else {}
        if _is_object_map(first_pipeline):
            pipeline = first_pipeline
    hardware_stages = pipeline.get(".hardware_stages")
    if not _is_object_map(hardware_stages):
        hardware_stages = {}
    compute_stage = hardware_stages.get(".cs")
    if not _is_object_map(compute_stage):
        compute_stage = {}
    shaders = pipeline.get(".shaders")
    if not _is_object_map(shaders):
        shaders = {}
    compute_shader = shaders.get(".compute")
    if not _is_object_map(compute_shader):
        compute_shader = {}
    return CodeObjectElfLayout(
        metadata=CodeObjectMetadata(
            entry_point=_optional_str(compute_stage.get(".entry_point")),
            api=_optional_str(pipeline.get(".api")),
            internal_pipeline_hash=_optional_hash_pair(pipeline.get(".internal_pipeline_hash")),
            api_shader_hash=_optional_hash_pair(compute_shader.get(".api_shader_hash")),
            lds_size=_optional_int(compute_stage.get(".lds_size")),
            scratch_memory_size=_optional_int(compute_stage.get(".scratch_memory_size")),
            sgpr_count=_optional_int(compute_stage.get(".sgpr_count")),
            vgpr_count=_optional_int(compute_stage.get(".vgpr_count")),
            wavefront_size=_optional_int(compute_stage.get(".wavefront_size")),
            spill_threshold=_optional_int(pipeline.get(".spill_threshold")),
            user_data_limit=_optional_int(pipeline.get(".user_data_limit")),
        ),
        executable_virtual_address_end=executable_virtual_address_end,
    )


def decode_first_msgpack_object(data: bytes) -> object:
    if not data:
        raise RuntimeError("AMDGPU metadata note is empty")
    value, _next_offset = _decode_msgpack(data, 0)
    return value


def decode_msgpack_stream(data: bytes) -> tuple[object, ...]:
    values: list[object] = []
    offset = 0
    while offset < len(data):
        if data[offset:] and data[offset:].strip(b"\x00") == b"":
            break
        value, next_offset = _decode_msgpack(data, offset)
        if next_offset <= offset:
            raise RuntimeError("MessagePack stream decoder did not make progress")
        values.append(value)
        offset = next_offset
    return tuple(values)


def _find_amdgpu_metadata_note(payload: bytes) -> bytes | None:
    if len(payload) < 64 or payload[:4] != b"\x7fELF":
        raise RuntimeError("Code object payload is not a 64-bit ELF")
    (
        _e_type,
        _e_machine,
        _e_version,
        _e_entry,
        _e_phoff,
        e_shoff,
        _e_flags,
        _e_ehsize,
        _e_phentsize,
        _e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = _unpack_elf_header(payload)
    sections: list[ElfSectionHeader] = []
    for index in range(e_shnum):
        section_offset = e_shoff + index * e_shentsize
        if section_offset + 64 > len(payload):
            raise RuntimeError(f"ELF section header {index} exceeds payload size")
        sections.append(_unpack_elf_section_header(payload, section_offset))
    if not (0 <= e_shstrndx < len(sections)):
        raise RuntimeError("ELF string table section index is invalid")
    section_name_table: ElfSectionHeader = sections[e_shstrndx]
    string_blob = payload[section_name_table[4] : section_name_table[4] + section_name_table[5]]
    note_blob = None
    for raw in sections:
        name_offset, section_type, _flags, _addr, offset, size, _link, _info, _align, _entsize = raw
        if section_type != 7:
            continue
        section_name = _read_c_string(string_blob, name_offset)
        if section_name == ".note":
            note_blob = payload[offset : offset + size]
            break
    if note_blob is None:
        return None
    cursor = 0
    while cursor + 12 <= len(note_blob):
        namesz, descsz, note_type = struct.unpack_from("<III", note_blob, cursor)
        cursor += 12
        name_blob = note_blob[cursor : cursor + namesz]
        cursor += (namesz + 3) & ~3
        desc_blob = note_blob[cursor : cursor + descsz]
        cursor += (descsz + 3) & ~3
        owner = name_blob.rstrip(b"\x00").decode("utf-8", errors="replace")
        if owner == "AMDGPU" and note_type == 32:
            return desc_blob
    return None


def _find_executable_virtual_address_end(payload: bytes) -> int:
    if len(payload) < 64 or payload[:4] != b"\x7fELF":
        raise RuntimeError("Code object payload is not a 64-bit ELF")
    (
        _e_type,
        _e_machine,
        _e_version,
        _e_entry,
        _e_phoff,
        e_shoff,
        _e_flags,
        _e_ehsize,
        _e_phentsize,
        _e_phnum,
        e_shentsize,
        e_shnum,
        _e_shstrndx,
    ) = _unpack_elf_header(payload)
    executable_virtual_address_end = 0
    for index in range(e_shnum):
        section_offset = e_shoff + index * e_shentsize
        if section_offset + 64 > len(payload):
            raise RuntimeError(f"ELF section header {index} exceeds payload size")
        (
            _name,
            section_type,
            section_flags,
            section_address,
            _offset,
            section_size,
            _link,
            _info,
            _align,
            _entsize,
        ) = _unpack_elf_section_header(payload, section_offset)
        if section_type == 8:
            continue
        if (section_flags & 0x4) == 0:
            continue
        executable_virtual_address_end = max(executable_virtual_address_end, int(section_address + section_size))
    return executable_virtual_address_end


def _read_c_string(blob: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(blob):
        return ""
    return blob[offset:].split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _decode_msgpack(data: bytes, offset: int) -> tuple[object, int]:
    if offset >= len(data):
        raise RuntimeError("MessagePack decode overran payload")
    marker = data[offset]
    offset += 1
    if marker <= 0x7F:
        return marker, offset
    if marker >= 0xE0:
        return marker - 0x100, offset
    if 0xA0 <= marker <= 0xBF:
        size = marker & 0x1F
        return _read_text(data, offset, size)
    if 0x90 <= marker <= 0x9F:
        size = marker & 0x0F
        return _read_array(data, offset, size)
    if 0x80 <= marker <= 0x8F:
        size = marker & 0x0F
        return _read_map(data, offset, size)
    if marker == 0xC0:
        return None, offset
    if marker == 0xC2:
        return False, offset
    if marker == 0xC3:
        return True, offset
    if marker == 0xCC:
        return data[offset], offset + 1
    if marker == 0xCD:
        return struct.unpack_from(">H", data, offset)[0], offset + 2
    if marker == 0xCE:
        return struct.unpack_from(">I", data, offset)[0], offset + 4
    if marker == 0xCF:
        return struct.unpack_from(">Q", data, offset)[0], offset + 8
    if marker == 0xD0:
        return struct.unpack_from(">b", data, offset)[0], offset + 1
    if marker == 0xD1:
        return struct.unpack_from(">h", data, offset)[0], offset + 2
    if marker == 0xD2:
        return struct.unpack_from(">i", data, offset)[0], offset + 4
    if marker == 0xD3:
        return struct.unpack_from(">q", data, offset)[0], offset + 8
    if marker == 0xD9:
        size = data[offset]
        return _read_text(data, offset + 1, size)
    if marker == 0xDA:
        size = struct.unpack_from(">H", data, offset)[0]
        return _read_text(data, offset + 2, size)
    if marker == 0xDB:
        size = struct.unpack_from(">I", data, offset)[0]
        return _read_text(data, offset + 4, size)
    if marker == 0xDC:
        size = struct.unpack_from(">H", data, offset)[0]
        return _read_array(data, offset + 2, size)
    if marker == 0xDD:
        size = struct.unpack_from(">I", data, offset)[0]
        return _read_array(data, offset + 4, size)
    if marker == 0xDE:
        size = struct.unpack_from(">H", data, offset)[0]
        return _read_map(data, offset + 2, size)
    if marker == 0xDF:
        size = struct.unpack_from(">I", data, offset)[0]
        return _read_map(data, offset + 4, size)
    raise RuntimeError(f"Unsupported MessagePack marker 0x{marker:02x}")


def _read_text(data: bytes, offset: int, size: int) -> tuple[str, int]:
    end = offset + size
    if end > len(data):
        raise RuntimeError("MessagePack string exceeds payload size")
    return data[offset:end].decode("utf-8", errors="replace"), end


def _read_array(data: bytes, offset: int, size: int) -> tuple[list[object], int]:
    items: list[object] = []
    for _ in range(size):
        item, offset = _decode_msgpack(data, offset)
        items.append(item)
    return items, offset


def _read_map(data: bytes, offset: int, size: int) -> tuple[dict[str, object], int]:
    mapping: dict[str, object] = {}
    for _ in range(size):
        key, offset = _decode_msgpack(data, offset)
        value, offset = _decode_msgpack(data, offset)
        if not isinstance(key, str):
            raise RuntimeError(f"MessagePack map key must be string, got {type(key).__name__}")
        mapping[key] = value
    return mapping, offset


def _unpack_elf_section_header(payload: bytes, offset: int) -> ElfSectionHeader:
    raw = struct.unpack_from("<IIQQQQIIQQ", payload, offset)
    return (
        int(raw[0]),
        int(raw[1]),
        int(raw[2]),
        int(raw[3]),
        int(raw[4]),
        int(raw[5]),
        int(raw[6]),
        int(raw[7]),
        int(raw[8]),
        int(raw[9]),
    )


def _unpack_elf_header(payload: bytes) -> ElfHeader:
    raw = struct.unpack_from("<HHIQQQIHHHHHH", payload, 16)
    return (
        int(raw[0]),
        int(raw[1]),
        int(raw[2]),
        int(raw[3]),
        int(raw[4]),
        int(raw[5]),
        int(raw[6]),
        int(raw[7]),
        int(raw[8]),
        int(raw[9]),
        int(raw[10]),
        int(raw[11]),
        int(raw[12]),
    )


def _is_object_map(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _optional_hash_pair(value: object) -> tuple[int, int] | None:
    if not _is_object_list(value) or len(value) != 2:
        return None
    lhs = value[0]
    rhs = value[1]
    if not isinstance(lhs, int) or not isinstance(rhs, int):
        return None
    return lhs, rhs
