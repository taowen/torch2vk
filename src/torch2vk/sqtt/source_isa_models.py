"""Data models for source/ISA SQTT hotspot reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from .driver_artifacts import DriverSourceSpan
from .pipeline_attribution import AttributedCodeObject


def _new_counter() -> Counter[str]:
    return Counter()


def _new_isa_span_list() -> list[tuple[int, int]]:
    return []


@dataclass(frozen=True, slots=True)
class SourceIsaHotRange:
    pipeline_debug_name: str
    shader_variant_name: str
    shader_family: str
    dispatch_labels: tuple[str, ...]
    glsl_path: str
    source_kind: str
    source_label: str
    line: int | None
    column: int | None
    spirv_offset: int | None
    source_text: str
    pc_start: int
    pc_end: int
    isa_offset: int
    isa_end_offset: int
    hit_count: int
    total_cycles: int
    categories: tuple[tuple[str, int], ...]
    disasm: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class SourceHotLine:
    pipeline_debug_name: str
    shader_variant_name: str
    shader_family: str
    dispatch_labels: tuple[str, ...]
    glsl_path: str
    source_kind: str
    source_label: str
    line: int | None
    spirv_offset: int | None
    source_text: str
    hit_count: int
    total_cycles: int
    source_span_count: int
    isa_spans: tuple[tuple[int, int], ...]
    hottest_categories: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class SourceIsaSqttHotspotReport:
    capture_path: str
    decoded_stream_count: int
    total_instruction_event_count: int
    matched_instruction_event_count: int
    unmatched_instruction_event_count: int
    zero_pc_instruction_event_count: int
    availability: str
    unavailable_reason: str | None
    hot_ranges: tuple[SourceIsaHotRange, ...]
    source_hot_lines: tuple[SourceHotLine, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class HotRangeAggregate:
    item: AttributedCodeObject
    source_span: DriverSourceSpan
    hit_count: int = 0
    total_cycles: int = 0
    categories: Counter[str] = field(default_factory=_new_counter)


@dataclass(slots=True)
class _SourceHotLineBucket:
    record: SourceIsaHotRange
    hit_count: int = 0
    total_cycles: int = 0
    source_span_count: int = 0
    isa_spans: list[tuple[int, int]] = field(default_factory=_new_isa_span_list)
    categories: Counter[str] = field(default_factory=_new_counter)
