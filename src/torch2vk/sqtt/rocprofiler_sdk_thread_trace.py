"""Decode SQTT streams through the official rocprofiler-sdk thread-trace API."""

from __future__ import annotations

import ctypes
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .rgp_capture_parser import (
    RgpCapture,
    RgpSqttDataChunk,
    RgpSqttDesc,
)
from .rocm_libraries import (
    find_rocprofiler_sdk_library,
    find_trace_decoder_library_dir,
    preload_rocprofiler_sdk_dependencies,
)


ROCPROFILER_STATUS_SUCCESS = 0

RECORD_TYPE_NAMES = {
    0: "GFXIP",
    1: "OCCUPANCY",
    2: "PERFEVENT",
    3: "WAVE",
    4: "INFO",
    5: "DEBUG",
    6: "SHADERDATA",
    7: "REALTIME",
    8: "RT_FREQUENCY",
    9: "INST_OTHER_SIMD",
}

INFO_TYPE_NAMES = {
    0: "NONE",
    1: "DATA_LOST",
    2: "STITCH_INCOMPLETE",
    3: "WAVE_INCOMPLETE",
}

INSTRUCTION_CATEGORY_NAMES = {
    0: "NONE",
    1: "SMEM",
    2: "SALU",
    3: "VMEM",
    4: "FLAT",
    5: "LDS",
    6: "VALU",
    7: "JUMP",
    8: "NEXT",
    9: "IMMED",
    10: "CONTEXT",
    11: "MESSAGE",
    12: "BVH",
}


class _DecoderId(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Pc(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("code_object_id", ctypes.c_uint64),
    ]


class _Occupancy(ctypes.Structure):
    _fields_ = [
        ("pc", _Pc),
        ("time", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8),
        ("cu", ctypes.c_uint8),
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("start", ctypes.c_uint32, 1),
        ("_reserved_bits", ctypes.c_uint32, 31),
    ]


class _WaveState(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("duration", ctypes.c_int32),
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
        ("timeline_array", ctypes.POINTER(_WaveState)),
        ("instructions_array", ctypes.POINTER(_Instruction)),
    ]


_DecoderCallback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)


@dataclass(frozen=True, slots=True)
class ThreadTraceNamedCount:
    name: str
    count: int


@dataclass(frozen=True, slots=True)
class ThreadTraceInstructionCategorySummary:
    name: str
    count: int
    total_cycles: int


@dataclass(frozen=True, slots=True)
class ThreadTraceInfoSummary:
    name: str
    count: int
    description: str


@dataclass(frozen=True, slots=True)
class ThreadTraceStreamSummary:
    stream_index: int
    shader_engine_index: int | None
    compute_unit_index: int | None
    size_bytes: int
    gfxip_major: int | None
    record_type_counts: tuple[ThreadTraceNamedCount, ...]
    instruction_category_counts: tuple[ThreadTraceInstructionCategorySummary, ...]
    info_records: tuple[ThreadTraceInfoSummary, ...]
    occupancy_event_count: int
    occupancy_wave_start_count: int
    occupancy_wave_end_count: int
    wave_count: int
    wave_timeline_event_count: int
    instruction_count: int
    instruction_issue_cycles: int
    stitched_instruction_count: int
    raw_pc_instruction_count: int
    perf_event_count: int
    shaderdata_count: int
    realtime_count: int
    other_simd_instruction_count: int


@dataclass(frozen=True, slots=True)
class ThreadTraceInstructionEvent:
    stream_index: int
    category: str
    duration_cycles: int
    stall_cycles: int
    time: int
    pc_address: int
    absolute_pc_address: int
    code_object_id: int


@dataclass(frozen=True, slots=True)
class ThreadTraceDecodedStream:
    summary: ThreadTraceStreamSummary
    instruction_events: tuple[ThreadTraceInstructionEvent, ...]


class RocprofilerSdkError(RuntimeError):
    pass


class _DecoderLibrary:
    def __init__(self, *, sdk_library_path: Path, decoder_library_dir: Path) -> None:
        self.sdk_library_path = sdk_library_path
        self.decoder_library_dir = decoder_library_dir
        preload_rocprofiler_sdk_dependencies(sdk_library_path)
        self._library = ctypes.CDLL(str(sdk_library_path), mode=ctypes.RTLD_GLOBAL)
        self._library.rocprofiler_thread_trace_decoder_create.argtypes = [
            ctypes.POINTER(_DecoderId),
            ctypes.c_char_p,
        ]
        self._library.rocprofiler_thread_trace_decoder_create.restype = ctypes.c_int
        self._library.rocprofiler_thread_trace_decoder_destroy.argtypes = [_DecoderId]
        self._library.rocprofiler_thread_trace_decoder_destroy.restype = None
        self._library.rocprofiler_trace_decode.argtypes = [
            _DecoderId,
            _DecoderCallback,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._library.rocprofiler_trace_decode.restype = ctypes.c_int
        self._library.rocprofiler_thread_trace_decoder_info_string.argtypes = [_DecoderId, ctypes.c_int]
        self._library.rocprofiler_thread_trace_decoder_info_string.restype = ctypes.c_char_p
        self._library.rocprofiler_get_status_string.argtypes = [ctypes.c_int]
        self._library.rocprofiler_get_status_string.restype = ctypes.c_char_p

    def status_string(self, status: int) -> str:
        raw = self._library.rocprofiler_get_status_string(status)
        if raw is None:
            return f"UNKNOWN_STATUS_{status}"
        return raw.decode("utf-8", errors="replace")

    def create_decoder(self) -> _DecoderId:
        handle = _DecoderId()
        status = self._library.rocprofiler_thread_trace_decoder_create(
            ctypes.byref(handle),
            os.fsencode(self.decoder_library_dir),
        )
        if status != ROCPROFILER_STATUS_SUCCESS:
            raise RocprofilerSdkError(
                f"rocprofiler_thread_trace_decoder_create failed with {self.status_string(status)}"
            )
        return handle

    def destroy_decoder(self, handle: _DecoderId) -> None:
        self._library.rocprofiler_thread_trace_decoder_destroy(handle)

    def decode_stream(self, handle: _DecoderId, blob: bytes, callback: object) -> None:
        buffer = ctypes.create_string_buffer(blob)
        status = self._library.rocprofiler_trace_decode(
            handle,
            callback,
            ctypes.cast(buffer, ctypes.c_void_p),
            len(blob),
            None,
        )
        if status != ROCPROFILER_STATUS_SUCCESS:
            raise RocprofilerSdkError(f"rocprofiler_trace_decode failed with {self.status_string(status)}")

    def info_string(self, handle: _DecoderId, info_type: int) -> str:
        raw = self._library.rocprofiler_thread_trace_decoder_info_string(handle, info_type)
        if raw is None:
            return f"UNKNOWN_INFO_{info_type}"
        return raw.decode("utf-8", errors="replace")


def _top_named_counts(counter: Counter[str], *, limit: int) -> tuple[ThreadTraceNamedCount, ...]:
    return tuple(ThreadTraceNamedCount(name=name, count=count) for name, count in counter.most_common(limit))


def _top_instruction_categories(
    count_by_name: Counter[str],
    cycles_by_name: Counter[str],
    *,
    limit: int,
) -> tuple[ThreadTraceInstructionCategorySummary, ...]:
    names = sorted(count_by_name, key=lambda item: (cycles_by_name[item], count_by_name[item], item), reverse=True)
    return tuple(
        ThreadTraceInstructionCategorySummary(
            name=name,
            count=count_by_name[name],
            total_cycles=cycles_by_name[name],
        )
        for name in names[:limit]
    )


def _find_sdk_library_path(explicit_path: Path | None) -> Path:
    return find_rocprofiler_sdk_library(explicit_path)


def _find_decoder_library_dir(explicit_dir: Path | None) -> Path:
    return find_trace_decoder_library_dir(explicit_dir)


def _record_event_count(record_type_id: int, trace_size: int) -> int:
    if trace_size != 0:
        return trace_size
    if record_type_id in {0, 8}:
        return 1
    return 0


class _StreamAccumulator:
    def __init__(
        self,
        *,
        library: _DecoderLibrary,
        handle: _DecoderId,
        stream_index: int,
    ) -> None:
        self._library = library
        self._handle = handle
        self._stream_index = stream_index
        self.gfxip_major: int | None = None
        self.record_type_counts: Counter[str] = Counter()
        self.instruction_category_counts: Counter[str] = Counter()
        self.instruction_category_cycles: Counter[str] = Counter()
        self.info_counts: Counter[str] = Counter()
        self.info_descriptions: dict[str, str] = {}
        self.instruction_events: list[ThreadTraceInstructionEvent] = []
        self.occupancy_event_count = 0
        self.occupancy_wave_start_count = 0
        self.occupancy_wave_end_count = 0
        self.wave_count = 0
        self.wave_timeline_event_count = 0
        self.instruction_count = 0
        self.instruction_issue_cycles = 0
        self.stitched_instruction_count = 0
        self.raw_pc_instruction_count = 0
        self.perf_event_count = 0
        self.shaderdata_count = 0
        self.realtime_count = 0
        self.other_simd_instruction_count = 0

    def callback(self, record_type_id: int, trace_events: ctypes.c_void_p, trace_size: int) -> None:
        record_name = RECORD_TYPE_NAMES.get(record_type_id, f"RECORD_{record_type_id}")
        self.record_type_counts[record_name] += _record_event_count(record_type_id, trace_size)
        if record_type_id == 0 and trace_size > 0:
            values = ctypes.cast(trace_events, ctypes.POINTER(ctypes.c_uint64))
            self.gfxip_major = int(values[0])
            return
        if record_type_id == 1:
            self._consume_occupancy(trace_events, trace_size)
            return
        if record_type_id == 2:
            self.perf_event_count += trace_size
            return
        if record_type_id == 3:
            self._consume_wave(trace_events, trace_size)
            return
        if record_type_id == 4:
            self._consume_info(trace_events, trace_size)
            return
        if record_type_id == 6:
            self.shaderdata_count += trace_size
            return
        if record_type_id == 7:
            self.realtime_count += trace_size
            return
        if record_type_id == 9:
            self.other_simd_instruction_count += trace_size

    def _consume_occupancy(self, trace_events: ctypes.c_void_p, trace_size: int) -> None:
        records = ctypes.cast(trace_events, ctypes.POINTER(_Occupancy))
        for index in range(trace_size):
            record = records[index]
            self.occupancy_event_count += 1
            if record.start:
                self.occupancy_wave_start_count += 1
            else:
                self.occupancy_wave_end_count += 1

    def _consume_wave(self, trace_events: ctypes.c_void_p, trace_size: int) -> None:
        records = ctypes.cast(trace_events, ctypes.POINTER(_Wave))
        for wave_index in range(trace_size):
            wave = records[wave_index]
            self.wave_count += 1
            self.wave_timeline_event_count += int(wave.timeline_size)
            self.instruction_count += int(wave.instructions_size)
            for instruction_index in range(int(wave.instructions_size)):
                instruction = wave.instructions_array[instruction_index]
                category_name = INSTRUCTION_CATEGORY_NAMES.get(int(instruction.category), f"CATEGORY_{instruction.category}")
                self.instruction_category_counts[category_name] += 1
                self.instruction_category_cycles[category_name] += max(0, int(instruction.duration))
                self.instruction_issue_cycles += max(0, int(instruction.duration))
                self.instruction_events.append(
                    ThreadTraceInstructionEvent(
                        stream_index=self._stream_index,
                        category=category_name,
                        duration_cycles=max(0, int(instruction.duration)),
                        stall_cycles=max(0, int(instruction.stall)),
                        time=int(instruction.time),
                        pc_address=int(instruction.pc.address),
                        absolute_pc_address=int(instruction.pc.address),
                        code_object_id=int(instruction.pc.code_object_id),
                    )
                )
                if int(instruction.pc.code_object_id) != 0:
                    self.stitched_instruction_count += 1
                elif int(instruction.pc.address) != 0:
                    self.raw_pc_instruction_count += 1

    def _consume_info(self, trace_events: ctypes.c_void_p, trace_size: int) -> None:
        records = ctypes.cast(trace_events, ctypes.POINTER(ctypes.c_int))
        for index in range(trace_size):
            info_type = int(records[index])
            info_name = INFO_TYPE_NAMES.get(info_type, f"INFO_{info_type}")
            self.info_counts[info_name] += 1
            self.info_descriptions[info_name] = self._library.info_string(self._handle, info_type)

    def build_summary(
        self,
        *,
        stream_index: int,
        desc: RgpSqttDesc | None,
        chunk: RgpSqttDataChunk,
        top_limit: int,
    ) -> ThreadTraceStreamSummary:
        info_records = tuple(
            ThreadTraceInfoSummary(
                name=name,
                count=count,
                description=self.info_descriptions.get(name, name),
            )
            for name, count in self.info_counts.most_common(top_limit)
        )
        return ThreadTraceStreamSummary(
            stream_index=stream_index,
            shader_engine_index=None if desc is None else desc.shader_engine_index,
            compute_unit_index=None if desc is None else desc.compute_unit_index,
            size_bytes=chunk.size,
            gfxip_major=self.gfxip_major,
            record_type_counts=_top_named_counts(self.record_type_counts, limit=top_limit),
            instruction_category_counts=_top_instruction_categories(
                self.instruction_category_counts,
                self.instruction_category_cycles,
                limit=top_limit,
            ),
            info_records=info_records,
            occupancy_event_count=self.occupancy_event_count,
            occupancy_wave_start_count=self.occupancy_wave_start_count,
            occupancy_wave_end_count=self.occupancy_wave_end_count,
            wave_count=self.wave_count,
            wave_timeline_event_count=self.wave_timeline_event_count,
            instruction_count=self.instruction_count,
            instruction_issue_cycles=self.instruction_issue_cycles,
            stitched_instruction_count=self.stitched_instruction_count,
            raw_pc_instruction_count=self.raw_pc_instruction_count,
            perf_event_count=self.perf_event_count,
            shaderdata_count=self.shaderdata_count,
            realtime_count=self.realtime_count,
            other_simd_instruction_count=self.other_simd_instruction_count,
        )

    def build_decoded_stream(
        self,
        *,
        stream_index: int,
        desc: RgpSqttDesc | None,
        chunk: RgpSqttDataChunk,
        top_limit: int,
    ) -> ThreadTraceDecodedStream:
        return ThreadTraceDecodedStream(
            summary=self.build_summary(stream_index=stream_index, desc=desc, chunk=chunk, top_limit=top_limit),
            instruction_events=tuple(self.instruction_events),
        )


class OfficialThreadTraceDecoder:
    def __init__(
        self,
        *,
        sdk_library_path: Path | None = None,
        decoder_library_dir: Path | None = None,
    ) -> None:
        self._library = _DecoderLibrary(
            sdk_library_path=_find_sdk_library_path(sdk_library_path),
            decoder_library_dir=_find_decoder_library_dir(decoder_library_dir),
        )
        self._handle = self._library.create_decoder()

    def close(self) -> None:
        if self._handle.handle == 0:
            return
        self._library.destroy_decoder(self._handle)
        self._handle = _DecoderId()

    def decode_chunk(
        self,
        *,
        stream_index: int,
        desc: RgpSqttDesc | None,
        chunk: RgpSqttDataChunk,
        capture: RgpCapture,
        top_limit: int,
    ) -> ThreadTraceStreamSummary:
        return self.decode_chunk_detailed(
            stream_index=stream_index,
            desc=desc,
            chunk=chunk,
            capture=capture,
            top_limit=top_limit,
        ).summary

    def decode_chunk_detailed(
        self,
        *,
        stream_index: int,
        desc: RgpSqttDesc | None,
        chunk: RgpSqttDataChunk,
        capture: RgpCapture,
        top_limit: int,
    ) -> ThreadTraceDecodedStream:
        blob = capture.blob[chunk.payload_offset : chunk.payload_end]
        accumulator = _StreamAccumulator(
            library=self._library,
            handle=self._handle,
            stream_index=stream_index,
        )

        def _callback(record_type_id: int, trace_events: ctypes.c_void_p, trace_size: int, _userdata: ctypes.c_void_p) -> None:
            accumulator.callback(record_type_id, trace_events, trace_size)

        callback = _DecoderCallback(_callback)
        self._library.decode_stream(self._handle, blob, callback)
        return accumulator.build_decoded_stream(stream_index=stream_index, desc=desc, chunk=chunk, top_limit=top_limit)

    def __enter__(self) -> OfficialThreadTraceDecoder:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()
