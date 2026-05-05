"""Shared thread-trace event names and decoded instruction event model."""

from __future__ import annotations

from dataclasses import dataclass


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
