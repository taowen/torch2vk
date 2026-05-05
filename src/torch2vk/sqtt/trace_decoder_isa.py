"""Direct rocprof-trace-decoder FFI with driver disassembly as the ISA service."""

from __future__ import annotations

import ctypes
import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .disasm_index import CompilerNativeDisasmIndex, DisasmInstruction, parse_compiler_native_disasm
from .pipeline_attribution import AttributedCodeObject, PipelineAttribution
from .rgp_capture_parser import RgpCapture, RgpSqttDataChunk
from .rocm_libraries import find_trace_decoder_library
from .rocprofiler_sdk_thread_trace import INSTRUCTION_CATEGORY_NAMES, RECORD_TYPE_NAMES, ThreadTraceInstructionEvent


TRACE_DECODER_STATUS_SUCCESS = 0
TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES = 2


class _Pc(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("code_object_id", ctypes.c_uint64),
    ]


class _Instruction(ctypes.Structure):
    _fields_ = [
        ("category", ctypes.c_uint32, 8),
        ("stall", ctypes.c_uint32, 24),
        ("duration", ctypes.c_int32),
        ("time", ctypes.c_int64),
        ("pc", _Pc),
    ]


class _Wave(ctypes.Structure):
    _fields_ = [
        ("cu", ctypes.c_uint8),
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("contexts", ctypes.c_uint8),
        ("_reserved1", ctypes.c_uint32),
        ("_reserved2", ctypes.c_uint32),
        ("_reserved3", ctypes.c_uint32),
        ("begin_time", ctypes.c_int64),
        ("end_time", ctypes.c_int64),
        ("timeline_size", ctypes.c_uint64),
        ("instructions_size", ctypes.c_uint64),
        ("timeline_array", ctypes.c_void_p),
        ("instructions_array", ctypes.POINTER(_Instruction)),
    ]


_TraceCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
_IsaCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    _Pc,
    ctypes.c_void_p,
)
_SeDataCallback = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_void_p,
)


@dataclass(frozen=True, slots=True)
class _CodeObjectLookup:
    load_id: int
    load_address: int
    item: AttributedCodeObject
    disasm_index: CompilerNativeDisasmIndex
    offsets: tuple[int, ...]


class TraceDecoderIsaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DecodedInstructionStreams:
    streams: tuple[tuple[ThreadTraceInstructionEvent, ...], ...]

    def stream_events(self, stream_index: int) -> tuple[ThreadTraceInstructionEvent, ...]:
        if stream_index < 0 or stream_index >= len(self.streams):
            return ()
        return self.streams[stream_index]


class _TraceDecoderLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = library_path
        self._library = ctypes.CDLL(str(library_path))
        self._library.rocprof_trace_decoder_parse_data.argtypes = [
            _SeDataCallback,
            _TraceCallback,
            _IsaCallback,
            ctypes.c_void_p,
        ]
        self._library.rocprof_trace_decoder_parse_data.restype = ctypes.c_int
        self._library.rocprof_trace_decoder_get_status_string.argtypes = [ctypes.c_int]
        self._library.rocprof_trace_decoder_get_status_string.restype = ctypes.c_char_p

    def status_string(self, status: int) -> str:
        raw = self._library.rocprof_trace_decoder_get_status_string(status)
        if raw is None:
            return f"TRACE_DECODER_STATUS_{status}"
        return raw.decode("utf-8", errors="replace")

    def parse(
        self,
        *,
        data_callback: object,
        trace_callback: object,
        isa_callback: object,
    ) -> None:
        status = self._library.rocprof_trace_decoder_parse_data(
            data_callback,
            trace_callback,
            isa_callback,
            None,
        )
        if status != TRACE_DECODER_STATUS_SUCCESS:
            raise TraceDecoderIsaError(
                f"rocprof_trace_decoder_parse_data failed with {self.status_string(status)}"
            )


@dataclass(slots=True)
class _EventAccumulator:
    instruction_events: list[ThreadTraceInstructionEvent]

    def __init__(self) -> None:
        self.instruction_events = []

    def trace_callback(self, record_type_id: int, trace_events: ctypes.c_void_p, trace_size: int) -> int:
        if RECORD_TYPE_NAMES.get(record_type_id) != "WAVE":
            return TRACE_DECODER_STATUS_SUCCESS
        waves = ctypes.cast(trace_events, ctypes.POINTER(_Wave))
        for wave_index in range(trace_size):
            wave = waves[wave_index]
            for instruction_index in range(int(wave.instructions_size)):
                instruction = wave.instructions_array[instruction_index]
                category_name = INSTRUCTION_CATEGORY_NAMES.get(int(instruction.category), f"CATEGORY_{instruction.category}")
                code_object_id = int(instruction.pc.code_object_id)
                pc_address = int(instruction.pc.address)
                absolute_pc_address = pc_address if code_object_id == 0 else 0
                self.instruction_events.append(
                    ThreadTraceInstructionEvent(
                        stream_index=0,
                        category=category_name,
                        duration_cycles=max(0, int(instruction.duration)),
                        stall_cycles=max(0, int(instruction.stall)),
                        time=int(instruction.time),
                        pc_address=pc_address,
                        absolute_pc_address=absolute_pc_address,
                        code_object_id=code_object_id,
                    )
                )
        return TRACE_DECODER_STATUS_SUCCESS


class IsaLookup:
    def __init__(self, pipeline_attribution: PipelineAttribution) -> None:
        self._by_code_object_id: dict[int, _CodeObjectLookup] = {}
        self._absolute_ranges: list[tuple[int, int, _CodeObjectLookup]] = []
        self._instruction_cache: dict[tuple[int, int], tuple[DisasmInstruction, int]] = {}
        self._rendered_instruction_cache: dict[tuple[int, int], tuple[bytes, int]] = {}
        for item in pipeline_attribution.attributed_code_objects:
            pipeline_hash = item.internal_pipeline_hash
            if pipeline_hash[0] != pipeline_hash[1]:
                raise TraceDecoderIsaError(f"Unsupported asymmetric internal pipeline hash {pipeline_hash}")
            disasm_index = parse_compiler_native_disasm(Path(item.driver_artifact.compiler_native_disasm_path))
            lookup = _CodeObjectLookup(
                load_id=pipeline_hash[0],
                load_address=item.driver_shader.shader_va,
                item=item,
                disasm_index=disasm_index,
                offsets=tuple(instruction.offset for instruction in disasm_index.instructions),
            )
            self._by_code_object_id[lookup.load_id] = lookup
            self._absolute_ranges.append(
                (
                    item.driver_shader.shader_va,
                    item.driver_shader.shader_va + item.driver_shader.exec_size,
                    lookup,
                )
            )
        self._absolute_ranges.sort(key=lambda item: item[0])

    def resolve(self, *, code_object_id: int, address: int) -> tuple[DisasmInstruction, int]:
        cache_key = (code_object_id, address)
        cached = self._instruction_cache.get(cache_key)
        if cached is not None:
            return cached
        if code_object_id != 0:
            lookup = self._by_code_object_id.get(code_object_id)
            if lookup is None:
                raise TraceDecoderIsaError(f"Unknown code object id {code_object_id}")
            instruction = _find_instruction(lookup.disasm_index, lookup.offsets, address)
            resolved = (instruction, lookup.load_address + address)
            self._instruction_cache[cache_key] = resolved
            return resolved
        for start, end, lookup in self._absolute_ranges:
            if start <= address < end:
                instruction = _find_instruction(lookup.disasm_index, lookup.offsets, address - start)
                resolved = (instruction, address)
                self._instruction_cache[cache_key] = resolved
                return resolved
        raise TraceDecoderIsaError(f"Could not resolve raw PC 0x{address:x}")

    def render_instruction_text(self, *, instruction: DisasmInstruction, code_object_id: int) -> str:
        if code_object_id == 0:
            return instruction.text
        lookup = self._by_code_object_id.get(code_object_id)
        if lookup is None:
            return instruction.text
        parts = instruction.text.rsplit(maxsplit=1)
        if len(parts) != 2:
            return instruction.text
        target_label = parts[1]
        target_offset = lookup.disasm_index.block_offsets.get(target_label)
        if target_offset is None:
            return instruction.text
        delta_bytes = target_offset - (instruction.offset + 4)
        if delta_bytes % 4 != 0:
            raise TraceDecoderIsaError(
                f"Branch target {target_label!r} does not resolve to an instruction-aligned delta for {instruction.text!r}"
            )
        delta = delta_bytes // 4
        return f"{parts[0]} {delta}"

    def render_instruction_bytes(self, *, code_object_id: int, address: int) -> tuple[bytes, int]:
        cache_key = (code_object_id, address)
        cached = self._rendered_instruction_cache.get(cache_key)
        if cached is not None:
            return cached
        instruction, _ = self.resolve(code_object_id=code_object_id, address=address)
        encoded = self.render_instruction_text(
            instruction=instruction,
            code_object_id=code_object_id,
        ).encode("utf-8")
        rendered = (encoded, instruction.size_bytes)
        self._rendered_instruction_cache[cache_key] = rendered
        return rendered


def decode_instruction_events_by_stream(
    *,
    capture: RgpCapture,
    pipeline_attribution: PipelineAttribution,
    stream_limit: int | None = None,
) -> DecodedInstructionStreams:
    chunks = capture.sqtt_data_chunks if stream_limit is None else capture.sqtt_data_chunks[:stream_limit]
    if len(chunks) <= 1:
        library = _TraceDecoderLibrary(find_trace_decoder_library())
        isa_lookup = IsaLookup(pipeline_attribution)
        return DecodedInstructionStreams(
            streams=tuple(
                _decode_single_stream_events(
                    library=library,
                    isa_lookup=isa_lookup,
                    capture=capture,
                    chunk=chunk,
                    stream_index=stream_index,
                )
                for stream_index, chunk in enumerate(chunks)
            )
        )
    max_workers = min(len(chunks), os.cpu_count() or 1)
    streams_by_index: list[tuple[ThreadTraceInstructionEvent, ...] | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _decode_single_stream_events_isolated,
                capture,
                pipeline_attribution,
                chunk,
                stream_index,
            )
            for stream_index, chunk in enumerate(chunks)
        ]
        for stream_index, future in enumerate(futures):
            streams_by_index[stream_index] = future.result()
    return DecodedInstructionStreams(
        streams=tuple(() if events is None else events for events in streams_by_index)
    )


def _decode_single_stream_events_isolated(
    capture: RgpCapture,
    pipeline_attribution: PipelineAttribution,
    chunk: RgpSqttDataChunk,
    stream_index: int,
) -> tuple[ThreadTraceInstructionEvent, ...]:
    library = _TraceDecoderLibrary(find_trace_decoder_library())
    isa_lookup = IsaLookup(pipeline_attribution)
    return _decode_single_stream_events(
        library=library,
        isa_lookup=isa_lookup,
        capture=capture,
        chunk=chunk,
        stream_index=stream_index,
    )


def _decode_single_stream_events(
    *,
    library: _TraceDecoderLibrary,
    isa_lookup: IsaLookup,
    capture: RgpCapture,
    chunk: RgpSqttDataChunk,
    stream_index: int,
) -> tuple[ThreadTraceInstructionEvent, ...]:
    accumulator = _EventAccumulator()
    blob = capture.blob[chunk.payload_offset : chunk.payload_end]
    buffer = ctypes.create_string_buffer(blob)
    delivered = False

    def _data_callback(
        buffer_ptr: ctypes.c_void_p,
        buffer_size: ctypes.c_void_p,
        _userdata: ctypes.c_void_p,
    ) -> int:
        nonlocal delivered
        if delivered:
            _set_void_ptr(buffer_ptr, ctypes.c_void_p())
            _set_uint64_ptr(buffer_size, 0)
            return 0
        delivered = True
        _set_void_ptr(buffer_ptr, ctypes.cast(buffer, ctypes.c_void_p))
        _set_uint64_ptr(buffer_size, len(blob))
        return len(blob)

    def _trace_callback(record_type_id: int, trace_events: ctypes.c_void_p, trace_size: int, _userdata: ctypes.c_void_p) -> int:
        return accumulator.trace_callback(record_type_id, trace_events, trace_size)

    def _isa_callback(
        instruction_buffer: ctypes.c_void_p,
        memory_size: ctypes.c_void_p,
        instruction_size: ctypes.c_void_p,
        address: _Pc,
        _userdata: ctypes.c_void_p,
    ) -> int:
        try:
            encoded, size_bytes = isa_lookup.render_instruction_bytes(
                code_object_id=int(address.code_object_id),
                address=int(address.address),
            )
        except TraceDecoderIsaError:
            return 3
        required = len(encoded)
        available = _get_uint64_ptr(instruction_size)
        _set_uint64_ptr(instruction_size, required)
        _set_uint64_ptr(memory_size, size_bytes)
        if required > available:
            return TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
        ctypes.memmove(instruction_buffer, encoded, required)
        return TRACE_DECODER_STATUS_SUCCESS

    library.parse(
        data_callback=_SeDataCallback(_data_callback),
        trace_callback=_TraceCallback(_trace_callback),
        isa_callback=_IsaCallback(_isa_callback),
    )
    events: list[ThreadTraceInstructionEvent] = []
    for event in accumulator.instruction_events:
        if event.code_object_id == 0:
            absolute_pc = event.absolute_pc_address
        else:
            _, absolute_pc = isa_lookup.resolve(code_object_id=event.code_object_id, address=event.pc_address)
        events.append(
            ThreadTraceInstructionEvent(
                stream_index=stream_index,
                category=event.category,
                duration_cycles=event.duration_cycles,
                stall_cycles=event.stall_cycles,
                time=event.time,
                pc_address=event.pc_address,
                absolute_pc_address=absolute_pc,
                code_object_id=event.code_object_id,
            )
        )
    return tuple(events)


def _set_void_ptr(pointer: ctypes.c_void_p, value: ctypes.c_void_p) -> None:
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = value


def _set_uint64_ptr(pointer: ctypes.c_void_p, value: int) -> None:
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64))[0] = value


def _get_uint64_ptr(pointer: ctypes.c_void_p) -> int:
    return int(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64))[0])


def _find_instruction(
    index: CompilerNativeDisasmIndex,
    offsets: tuple[int, ...],
    offset: int,
) -> DisasmInstruction:
    position = bisect_right(offsets, offset) - 1
    if position < 0:
        raise TraceDecoderIsaError(f"Could not resolve instruction at offset {offset}")
    instruction = index.instructions[position]
    if not (instruction.offset <= offset < instruction.end_offset):
        raise TraceDecoderIsaError(f"Could not resolve instruction at offset {offset}")
    return instruction
