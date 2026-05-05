"""Join official SQTT instruction events with driver ISA/source mappings."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .disasm_index import CompilerNativeDisasmIndex, DisasmInstruction, parse_compiler_native_disasm
from .driver_artifacts import (
    DriverPipelineArtifact,
    DriverSourceSpan,
    build_driver_instruction_debug_index,
    find_driver_instruction_debug_record_in_index,
)
from .pipeline_attribution import AttributedCodeObject, PipelineAttribution
from .rgp_capture_parser import RgpCapture, parse_rgp_capture
from .rocprofiler_sdk_thread_trace import ThreadTraceInstructionEvent
from .source_actionability import classify_hot_line_actionability
from .trace_decoder_isa import DecodedInstructionStreams, decode_instruction_events_by_stream


def _new_counter() -> Counter[str]:
    return Counter()


def _new_int_zero_pc_stream_aggregate_map() -> dict[int, "_ZeroPcStreamAggregate"]:
    return {}


def _new_str_instruction_coverage_counts_map() -> dict[str, "_InstructionCoverageCounts"]:
    return {}


def _new_int_instruction_coverage_stream_aggregate_map() -> dict[int, "_InstructionCoverageStreamAggregate"]:
    return {}


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
    actionability_score: int
    actionability_label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceIsaRegionLine:
    source_kind: str
    source_label: str
    line: int | None
    column: int | None
    spirv_offset: int | None
    source_text: str
    hit_count: int
    total_cycles: int
    source_span_count: int
    pc_start: int
    pc_end: int
    isa_spans: tuple[tuple[int, int], ...]
    categories: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceIsaRegionInstruction:
    offset: int
    end_offset: int
    text: str
    opcode: str
    block_label: str | None
    source_kind: str
    source_label: str
    line: int | None
    column: int | None
    spirv_offset: int | None
    source_text: str
    hit_count: int
    total_cycles: int
    categories: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceIsaHotRegion:
    pipeline_debug_name: str
    shader_variant_name: str
    shader_family: str
    dispatch_labels: tuple[str, ...]
    glsl_path: str
    source_start_line: int | None
    source_end_line: int | None
    hit_count: int
    total_cycles: int
    pc_start: int
    pc_end: int
    isa_offset: int
    isa_end_offset: int
    hottest_categories: tuple[tuple[str, int], ...]
    lines: tuple[SourceIsaRegionLine, ...]
    isa: tuple[SourceIsaRegionInstruction, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ZeroPcNamedCount:
    name: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InstructionCoverageCategorySummary:
    name: str
    total_count: int
    matched_count: int
    unmatched_count: int
    zero_pc_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InstructionCoverageStreamSummary:
    stream_index: int
    total_count: int
    matched_count: int
    unmatched_count: int
    zero_pc_count: int
    total_category_counts: tuple[ZeroPcNamedCount, ...]
    zero_pc_category_counts: tuple[ZeroPcNamedCount, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InstructionCoverageReport:
    categories: tuple[InstructionCoverageCategorySummary, ...]
    streams: tuple[InstructionCoverageStreamSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ZeroPcStreamSummary:
    stream_index: int
    total_count: int
    expected_count: int
    unexpected_count: int
    reason_counts: tuple[ZeroPcNamedCount, ...]
    category_counts: tuple[ZeroPcNamedCount, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ZeroPcBreakdown:
    expected_zero_pc_instruction_event_count: int
    unexpected_zero_pc_instruction_event_count: int
    reason_counts: tuple[ZeroPcNamedCount, ...]
    category_counts: tuple[ZeroPcNamedCount, ...]
    streams: tuple[ZeroPcStreamSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceIsaSqttHotspotReport:
    capture_path: str
    decoded_stream_count: int
    total_instruction_event_count: int
    matched_instruction_event_count: int
    unmatched_instruction_event_count: int
    zero_pc_instruction_event_count: int
    zero_pc_breakdown: ZeroPcBreakdown
    instruction_coverage: InstructionCoverageReport
    availability: str
    unavailable_reason: str | None
    hot_ranges: tuple[SourceIsaHotRange, ...]
    hot_regions: tuple[SourceIsaHotRegion, ...]
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
class _ZeroPcStreamAggregate:
    expected_count: int = 0
    unexpected_count: int = 0
    reason_counts: Counter[str] = field(default_factory=_new_counter)
    category_counts: Counter[str] = field(default_factory=_new_counter)


@dataclass(slots=True)
class ZeroPcAggregate:
    expected_count: int = 0
    unexpected_count: int = 0
    reason_counts: Counter[str] = field(default_factory=_new_counter)
    category_counts: Counter[str] = field(default_factory=_new_counter)
    streams: dict[int, _ZeroPcStreamAggregate] = field(default_factory=_new_int_zero_pc_stream_aggregate_map)


@dataclass(slots=True)
class _InstructionCoverageCounts:
    total_count: int = 0
    matched_count: int = 0
    unmatched_count: int = 0
    zero_pc_count: int = 0


@dataclass(slots=True)
class _InstructionCoverageStreamAggregate:
    total_count: int = 0
    matched_count: int = 0
    unmatched_count: int = 0
    zero_pc_count: int = 0
    total_category_counts: Counter[str] = field(default_factory=_new_counter)
    zero_pc_category_counts: Counter[str] = field(default_factory=_new_counter)


@dataclass(slots=True)
class _InstructionCoverageAggregate:
    categories: dict[str, _InstructionCoverageCounts] = field(default_factory=_new_str_instruction_coverage_counts_map)
    streams: dict[int, _InstructionCoverageStreamAggregate] = field(
        default_factory=_new_int_instruction_coverage_stream_aggregate_map
    )


@dataclass(slots=True)
class _SourceHotLineBucket:
    record: SourceIsaHotRange
    hit_count: int = 0
    total_cycles: int = 0
    source_span_count: int = 0
    isa_spans: list[tuple[int, int]] = field(default_factory=_new_isa_span_list)
    categories: Counter[str] = field(default_factory=_new_counter)


@dataclass(slots=True)
class _SourceIsaRegionLineBucket:
    record: SourceIsaHotRange
    hit_count: int = 0
    total_cycles: int = 0
    source_span_count: int = 0
    pc_start: int = 0
    pc_end: int = 0
    isa_spans: list[tuple[int, int]] = field(default_factory=_new_isa_span_list)
    categories: Counter[str] = field(default_factory=_new_counter)


def write_source_isa_hotspot_report(
    *,
    output_path: Path,
    capture_path: Path,
    pipeline_attribution: PipelineAttribution,
    top_limit: int = 16,
) -> Path:
    report = build_source_isa_hotspot_report(
        capture_path=capture_path,
        pipeline_attribution=pipeline_attribution,
        top_limit=top_limit,
    )
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_source_isa_hotspot_report(
    *,
    capture_path: Path,
    pipeline_attribution: PipelineAttribution,
    top_limit: int = 16,
    capture: RgpCapture | None = None,
    decoded_instruction_streams: DecodedInstructionStreams | None = None,
) -> SourceIsaSqttHotspotReport:
    resolved_capture = parse_rgp_capture(capture_path) if capture is None else capture
    resolved_decoded_instruction_streams = decoded_instruction_streams
    if resolved_decoded_instruction_streams is None:
        resolved_decoded_instruction_streams = decode_instruction_events_by_stream(
            capture=resolved_capture,
            pipeline_attribution=pipeline_attribution,
        )
    artifacts = {
        item.runtime_pipeline.pipeline_debug_name: item.driver_artifact for item in pipeline_attribution.attributed_code_objects
    }
    instruction_debug_index = build_driver_instruction_debug_index(artifacts)
    attributed_by_pipeline = {
        item.runtime_pipeline.pipeline_debug_name: item for item in pipeline_attribution.attributed_code_objects
    }
    disasm_cache: dict[str, CompilerNativeDisasmIndex] = {}
    source_lines_cache: dict[str, list[str]] = {}
    aggregates: dict[tuple[str, int, int, str, int | None, int | None, int | None], HotRangeAggregate] = {}
    total_instruction_event_count = 0
    matched_instruction_event_count = 0
    unmatched_instruction_event_count = 0
    zero_pc_instruction_event_count = 0
    zero_pc_aggregate = ZeroPcAggregate()
    instruction_coverage_aggregate = _InstructionCoverageAggregate()
    decoded_stream_count = len(resolved_capture.sqtt_data_chunks)
    for stream_index, _chunk in enumerate(resolved_capture.sqtt_data_chunks):
        stream_events = resolved_decoded_instruction_streams.stream_events(stream_index)
        for event in stream_events:
            total_instruction_event_count += 1
            _record_instruction_coverage_event(
                aggregate=instruction_coverage_aggregate,
                event=event,
                outcome="total",
            )
            if event.absolute_pc_address == 0:
                zero_pc_instruction_event_count += 1
                accumulate_zero_pc_event(zero_pc_aggregate, event)
                _record_instruction_coverage_event(
                    aggregate=instruction_coverage_aggregate,
                    event=event,
                    outcome="zero_pc",
                )
                continue
            instruction_match = find_driver_instruction_debug_record_in_index(
                instruction_debug_index,
                pc=event.absolute_pc_address,
            )
            if instruction_match is None:
                unmatched_instruction_event_count += 1
                _record_instruction_coverage_event(
                    aggregate=instruction_coverage_aggregate,
                    event=event,
                    outcome="unmatched",
                )
                continue
            pipeline_name = instruction_match.pipeline_name
            source_span = instruction_match.instruction_debug_record
            item = attributed_by_pipeline.get(pipeline_name)
            if item is None:
                unmatched_instruction_event_count += 1
                _record_instruction_coverage_event(
                    aggregate=instruction_coverage_aggregate,
                    event=event,
                    outcome="unmatched",
                )
                continue
            matched_instruction_event_count += 1
            _record_instruction_coverage_event(
                aggregate=instruction_coverage_aggregate,
                event=event,
                outcome="matched",
            )
            key = (
                pipeline_name,
                source_span.pc_start,
                source_span.pc_end,
                source_span.source_kind,
                source_span.line,
                source_span.column,
                source_span.spirv_offset,
            )
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregate = HotRangeAggregate(item=item, source_span=source_span)
                aggregates[key] = aggregate
            aggregate.hit_count += 1
            aggregate.total_cycles += event.duration_cycles
            aggregate.categories[event.category] += 1
    all_hot_ranges = build_hot_ranges(
        aggregates=aggregates,
        disasm_cache=disasm_cache,
        source_lines_cache=source_lines_cache,
    )
    hot_regions = build_hot_regions(
        hot_ranges=all_hot_ranges,
        artifacts=artifacts,
        disasm_cache=disasm_cache,
        source_lines_cache=source_lines_cache,
        top_limit=top_limit,
    )
    source_hot_lines = build_source_hot_lines(hot_ranges=all_hot_ranges, top_limit=top_limit)
    availability, unavailable_reason = _availability(
        total_instruction_event_count=total_instruction_event_count,
        matched_instruction_event_count=matched_instruction_event_count,
    )
    return SourceIsaSqttHotspotReport(
        capture_path=str(capture_path),
        decoded_stream_count=decoded_stream_count,
        total_instruction_event_count=total_instruction_event_count,
        matched_instruction_event_count=matched_instruction_event_count,
        unmatched_instruction_event_count=unmatched_instruction_event_count,
        zero_pc_instruction_event_count=zero_pc_instruction_event_count,
        zero_pc_breakdown=materialize_zero_pc_breakdown(zero_pc_aggregate),
        instruction_coverage=_materialize_instruction_coverage(instruction_coverage_aggregate),
        availability=availability,
        unavailable_reason=unavailable_reason,
        hot_ranges=all_hot_ranges[:top_limit],
        hot_regions=hot_regions,
        source_hot_lines=source_hot_lines,
    )


def build_hot_ranges(
    *,
    aggregates: dict[tuple[str, int, int, str, int | None, int | None, int | None], HotRangeAggregate],
    disasm_cache: dict[str, CompilerNativeDisasmIndex],
    source_lines_cache: dict[str, list[str]],
) -> tuple[SourceIsaHotRange, ...]:
    ordered = sorted(aggregates.values(), key=lambda item: (item.total_cycles, item.hit_count), reverse=True)
    results: list[SourceIsaHotRange] = []
    for aggregate in ordered:
        item = aggregate.item
        source_text = _source_text(
            glsl_path=item.runtime_pipeline.glsl_path,
            line=aggregate.source_span.line,
            source_lines_cache=source_lines_cache,
        )
        results.append(
            SourceIsaHotRange(
                pipeline_debug_name=item.runtime_pipeline.pipeline_debug_name,
                shader_variant_name=item.runtime_pipeline.shader_variant_name,
                shader_family=item.runtime_pipeline.shader_family,
                dispatch_labels=item.runtime_pipeline.dispatch_labels,
                glsl_path=item.runtime_pipeline.glsl_path,
                source_kind=aggregate.source_span.source_kind,
                source_label=_source_label(aggregate.source_span),
                line=aggregate.source_span.line,
                column=aggregate.source_span.column,
                spirv_offset=aggregate.source_span.spirv_offset,
                source_text=source_text,
                pc_start=aggregate.source_span.pc_start,
                pc_end=aggregate.source_span.pc_end,
                isa_offset=aggregate.source_span.isa_offset,
                isa_end_offset=aggregate.source_span.isa_end_offset,
                hit_count=aggregate.hit_count,
                total_cycles=aggregate.total_cycles,
                categories=tuple(aggregate.categories.most_common()),
                disasm=tuple(
                    asdict(instruction)
                    for instruction in _disasm_span(
                        item.driver_artifact,
                        aggregate.source_span,
                        disasm_cache=disasm_cache,
                    )
                ),
            )
        )
    return tuple(results)


def accumulate_zero_pc_event(aggregate: ZeroPcAggregate, event: ThreadTraceInstructionEvent) -> None:
    expectation, reason = classify_zero_pc_event(event)
    if expectation == "expected":
        aggregate.expected_count += 1
    else:
        aggregate.unexpected_count += 1
    aggregate.reason_counts[reason] += 1
    aggregate.category_counts[event.category] += 1

    stream = aggregate.streams.get(event.stream_index)
    if stream is None:
        stream = _ZeroPcStreamAggregate()
        aggregate.streams[event.stream_index] = stream
    if expectation == "expected":
        stream.expected_count += 1
    else:
        stream.unexpected_count += 1
    stream.reason_counts[reason] += 1
    stream.category_counts[event.category] += 1


def _record_instruction_coverage_event(
    *,
    aggregate: _InstructionCoverageAggregate,
    event: ThreadTraceInstructionEvent,
    outcome: str,
) -> None:
    category = aggregate.categories.get(event.category)
    if category is None:
        category = _InstructionCoverageCounts()
        aggregate.categories[event.category] = category
    stream = aggregate.streams.get(event.stream_index)
    if stream is None:
        stream = _InstructionCoverageStreamAggregate()
        aggregate.streams[event.stream_index] = stream

    if outcome == "total":
        category.total_count += 1
        stream.total_count += 1
        stream.total_category_counts[event.category] += 1
        return
    if outcome == "matched":
        category.matched_count += 1
        stream.matched_count += 1
        return
    if outcome == "unmatched":
        category.unmatched_count += 1
        stream.unmatched_count += 1
        return
    if outcome == "zero_pc":
        category.zero_pc_count += 1
        stream.zero_pc_count += 1
        stream.zero_pc_category_counts[event.category] += 1
        return
    raise ValueError(f"Unsupported instruction coverage outcome: {outcome}")


def classify_zero_pc_event(event: ThreadTraceInstructionEvent) -> tuple[str, str]:
    if event.absolute_pc_address != 0:
        raise ValueError("zero-pc classification only applies to events with absolute_pc_address == 0")
    if event.code_object_id == 0 and event.pc_address == 0:
        return ("expected", "missing_raw_pc")
    if event.code_object_id == 0 and event.pc_address != 0:
        return ("unexpected", "raw_pc_not_promoted")
    if event.code_object_id != 0 and event.pc_address == 0:
        return ("unexpected", "missing_relative_pc_with_code_object")
    return ("unexpected", "unresolved_code_object_pc")


def _materialize_instruction_coverage(aggregate: _InstructionCoverageAggregate) -> InstructionCoverageReport:
    categories = tuple(
        InstructionCoverageCategorySummary(
            name=name,
            total_count=counts.total_count,
            matched_count=counts.matched_count,
            unmatched_count=counts.unmatched_count,
            zero_pc_count=counts.zero_pc_count,
        )
        for name, counts in sorted(
            aggregate.categories.items(),
            key=lambda item: (
                item[1].zero_pc_count,
                item[1].unmatched_count,
                item[1].matched_count,
                item[1].total_count,
                item[0],
            ),
            reverse=True,
        )
    )
    streams = tuple(
        InstructionCoverageStreamSummary(
            stream_index=stream_index,
            total_count=stream.total_count,
            matched_count=stream.matched_count,
            unmatched_count=stream.unmatched_count,
            zero_pc_count=stream.zero_pc_count,
            total_category_counts=tuple(
                ZeroPcNamedCount(name=name, count=count)
                for name, count in stream.total_category_counts.most_common()
            ),
            zero_pc_category_counts=tuple(
                ZeroPcNamedCount(name=name, count=count)
                for name, count in stream.zero_pc_category_counts.most_common()
            ),
        )
        for stream_index, stream in sorted(
            aggregate.streams.items(),
            key=lambda item: (item[1].zero_pc_count, item[1].total_count, item[0]),
            reverse=True,
        )
    )
    return InstructionCoverageReport(categories=categories, streams=streams)


def materialize_zero_pc_breakdown(aggregate: ZeroPcAggregate) -> ZeroPcBreakdown:
    streams = tuple(
        ZeroPcStreamSummary(
            stream_index=stream_index,
            total_count=stream.expected_count + stream.unexpected_count,
            expected_count=stream.expected_count,
            unexpected_count=stream.unexpected_count,
            reason_counts=tuple(
                ZeroPcNamedCount(name=name, count=count)
                for name, count in stream.reason_counts.most_common()
            ),
            category_counts=tuple(
                ZeroPcNamedCount(name=name, count=count)
                for name, count in stream.category_counts.most_common()
            ),
        )
        for stream_index, stream in sorted(aggregate.streams.items())
    )
    return ZeroPcBreakdown(
        expected_zero_pc_instruction_event_count=aggregate.expected_count,
        unexpected_zero_pc_instruction_event_count=aggregate.unexpected_count,
        reason_counts=tuple(
            ZeroPcNamedCount(name=name, count=count)
            for name, count in aggregate.reason_counts.most_common()
        ),
        category_counts=tuple(
            ZeroPcNamedCount(name=name, count=count)
            for name, count in aggregate.category_counts.most_common()
        ),
        streams=streams,
    )


def build_source_hot_lines(
    *,
    hot_ranges: tuple[SourceIsaHotRange, ...],
    top_limit: int,
) -> tuple[SourceHotLine, ...]:
    grouped: dict[tuple[str, str, str, int | None, int | None], _SourceHotLineBucket] = {}
    for record in hot_ranges:
        key = (record.pipeline_debug_name, record.glsl_path, record.source_kind, record.line, record.spirv_offset)
        bucket = grouped.get(key)
        if bucket is None:
            bucket = _SourceHotLineBucket(record=record)
            grouped[key] = bucket
        bucket.hit_count += record.hit_count
        bucket.total_cycles += record.total_cycles
        bucket.source_span_count += 1
        bucket.isa_spans.append((record.isa_offset, record.isa_end_offset))
        for category_name, category_count in record.categories:
            bucket.categories[category_name] += category_count
    lines: list[SourceHotLine] = []
    for bucket in grouped.values():
        record = bucket.record
        line = SourceHotLine(
            pipeline_debug_name=record.pipeline_debug_name,
            shader_variant_name=record.shader_variant_name,
            shader_family=record.shader_family,
            dispatch_labels=record.dispatch_labels,
            glsl_path=record.glsl_path,
            source_kind=record.source_kind,
            source_label=record.source_label,
            line=record.line,
            spirv_offset=record.spirv_offset,
            source_text=record.source_text,
            hit_count=bucket.hit_count,
            total_cycles=bucket.total_cycles,
            source_span_count=bucket.source_span_count,
            isa_spans=_merge_spans(tuple(bucket.isa_spans)),
            hottest_categories=tuple(bucket.categories.most_common()),
            actionability_score=0,
            actionability_label="",
        )
        actionability = classify_hot_line_actionability(source_kind=line.source_kind, source_text=line.source_text)
        lines.append(
            SourceHotLine(
                pipeline_debug_name=line.pipeline_debug_name,
                shader_variant_name=line.shader_variant_name,
                shader_family=line.shader_family,
                dispatch_labels=line.dispatch_labels,
                glsl_path=line.glsl_path,
                source_kind=line.source_kind,
                source_label=line.source_label,
                line=line.line,
                spirv_offset=line.spirv_offset,
                source_text=line.source_text,
                hit_count=line.hit_count,
                total_cycles=line.total_cycles,
                source_span_count=line.source_span_count,
                isa_spans=line.isa_spans,
                hottest_categories=line.hottest_categories,
                actionability_score=actionability.score,
                actionability_label=actionability.label,
            )
        )
    lines.sort(
        key=lambda item: (
            item.total_cycles,
            item.hit_count,
            item.line if item.line is not None else -1,
            item.spirv_offset if item.spirv_offset is not None else -1,
        ),
        reverse=True,
    )
    return tuple(lines[:top_limit])


def build_hot_regions(
    *,
    hot_ranges: tuple[SourceIsaHotRange, ...],
    artifacts: dict[str, DriverPipelineArtifact],
    disasm_cache: dict[str, CompilerNativeDisasmIndex],
    source_lines_cache: dict[str, list[str]],
    top_limit: int,
) -> tuple[SourceIsaHotRegion, ...]:
    by_pipeline: dict[str, list[SourceIsaHotRange]] = {}
    for record in hot_ranges:
        by_pipeline.setdefault(record.pipeline_debug_name, []).append(record)
    regions: list[SourceIsaHotRegion] = []
    for pipeline_name, records in by_pipeline.items():
        pipeline_records = sorted(
            records,
            key=lambda item: (
                item.isa_offset,
                item.isa_end_offset,
                -1 if item.line is None else item.line,
                -1 if item.column is None else item.column,
                -1 if item.spirv_offset is None else item.spirv_offset,
            ),
        )
        current_region: list[SourceIsaHotRange] = []
        current_end = -1
        for record in pipeline_records:
            if current_region and record.isa_offset > current_end:
                regions.append(
                    _materialize_hot_region(
                        records=tuple(current_region),
                        artifact=artifacts[pipeline_name],
                        disasm_cache=disasm_cache,
                        source_lines_cache=source_lines_cache,
                    )
                )
                current_region = []
                current_end = -1
            current_region.append(record)
            current_end = max(current_end, record.isa_end_offset)
        if current_region:
            regions.append(
                _materialize_hot_region(
                    records=tuple(current_region),
                    artifact=artifacts[pipeline_name],
                    disasm_cache=disasm_cache,
                    source_lines_cache=source_lines_cache,
                )
            )
    regions.sort(key=lambda item: (item.total_cycles, item.hit_count), reverse=True)
    return tuple(regions[:top_limit])


def _materialize_hot_region(
    *,
    records: tuple[SourceIsaHotRange, ...],
    artifact: DriverPipelineArtifact,
    disasm_cache: dict[str, CompilerNativeDisasmIndex],
    source_lines_cache: dict[str, list[str]],
) -> SourceIsaHotRegion:
    first = records[0]
    categories = Counter[str]()
    for record in records:
        for category_name, category_count in record.categories:
            categories[category_name] += category_count
    ordered_records: tuple[SourceIsaHotRange, ...] = tuple(sorted(
        records,
        key=lambda item: (
            item.isa_offset,
            -1 if item.line is None else item.line,
            -1 if item.column is None else item.column,
            -1 if item.spirv_offset is None else item.spirv_offset,
        ),
    ))
    region_lines = _build_region_lines(records=ordered_records)
    source_line_numbers = tuple(record.line for record in ordered_records if record.line is not None and record.line > 0)
    source_start_line = min(source_line_numbers) if source_line_numbers else None
    source_end_line = max(source_line_numbers) if source_line_numbers else None
    isa_offset = min(record.isa_offset for record in ordered_records)
    isa_end_offset = max(record.isa_end_offset for record in ordered_records)
    disasm_instructions = _region_disasm(
        artifact,
        isa_offset=isa_offset,
        isa_end_offset=isa_end_offset,
        disasm_cache=disasm_cache,
    )
    annotated_isa = tuple(
        _annotate_region_instruction(
            instruction=instruction,
            records=ordered_records,
            source_lines_cache=source_lines_cache,
            glsl_path=first.glsl_path,
        )
        for instruction in disasm_instructions
    )
    return SourceIsaHotRegion(
        pipeline_debug_name=first.pipeline_debug_name,
        shader_variant_name=first.shader_variant_name,
        shader_family=first.shader_family,
        dispatch_labels=first.dispatch_labels,
        glsl_path=first.glsl_path,
        source_start_line=source_start_line,
        source_end_line=source_end_line,
        hit_count=sum(record.hit_count for record in ordered_records),
        total_cycles=sum(record.total_cycles for record in ordered_records),
        pc_start=min(record.pc_start for record in ordered_records),
        pc_end=max(record.pc_end for record in ordered_records),
        isa_offset=isa_offset,
        isa_end_offset=isa_end_offset,
        hottest_categories=tuple(categories.most_common()),
        lines=region_lines,
        isa=annotated_isa,
    )


def _region_disasm(
    artifact: DriverPipelineArtifact,
    *,
    isa_offset: int,
    isa_end_offset: int,
    disasm_cache: dict[str, CompilerNativeDisasmIndex],
) -> tuple[DisasmInstruction, ...]:
    disasm_index = disasm_cache.get(artifact.pipeline_name)
    if disasm_index is None:
        disasm_index = parse_compiler_native_disasm(Path(artifact.compiler_native_disasm_path))
        disasm_cache[artifact.pipeline_name] = disasm_index
    return tuple(
        instruction
        for instruction in disasm_index.instructions
        if instruction.offset < isa_end_offset and instruction.end_offset > isa_offset
    )


def _annotate_region_instruction(
    *,
    instruction: DisasmInstruction,
    records: tuple[SourceIsaHotRange, ...],
    source_lines_cache: dict[str, list[str]],
    glsl_path: str,
) -> SourceIsaRegionInstruction:
    match = match_instruction_to_hot_range(instruction=instruction, records=records)
    if match is None:
        source_kind = "compiler_generated"
        source_label = "compiler-generated"
        line = None
        column = None
        spirv_offset = None
        source_text = ""
        hit_count = 0
        total_cycles = 0
        categories: tuple[tuple[str, int], ...] = ()
    else:
        source_kind = match.source_kind
        source_label = match.source_label
        line = match.line
        column = match.column
        spirv_offset = match.spirv_offset
        source_text = match.source_text or _source_text(
            glsl_path=glsl_path,
            line=match.line,
            source_lines_cache=source_lines_cache,
        )
        hit_count = match.hit_count
        total_cycles = match.total_cycles
        categories = match.categories
    return SourceIsaRegionInstruction(
        offset=instruction.offset,
        end_offset=instruction.end_offset,
        text=instruction.text,
        opcode=instruction.opcode,
        block_label=instruction.block_label,
        source_kind=source_kind,
        source_label=source_label,
        line=line,
        column=column,
        spirv_offset=spirv_offset,
        source_text=source_text,
        hit_count=hit_count,
        total_cycles=total_cycles,
        categories=categories,
    )


def match_instruction_to_hot_range(
    *,
    instruction: DisasmInstruction,
    records: tuple[SourceIsaHotRange, ...],
) -> SourceIsaHotRange | None:
    best_overlap = -1
    best_cycles = -1
    best_record: SourceIsaHotRange | None = None
    for record in records:
        overlap_start = max(instruction.offset, record.isa_offset)
        overlap_end = min(instruction.end_offset, record.isa_end_offset)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue
        if overlap > best_overlap or (overlap == best_overlap and record.total_cycles > best_cycles):
            best_overlap = overlap
            best_cycles = record.total_cycles
            best_record = record
    return best_record


def _build_region_lines(*, records: tuple[SourceIsaHotRange, ...]) -> tuple[SourceIsaRegionLine, ...]:
    grouped: dict[tuple[str, int | None, int | None, int | None, str], _SourceIsaRegionLineBucket] = {}
    for record in records:
        key = (record.source_kind, record.line, record.column, record.spirv_offset, record.source_text)
        bucket = grouped.get(key)
        if bucket is None:
            bucket = _SourceIsaRegionLineBucket(record=record, pc_start=record.pc_start, pc_end=record.pc_end)
            grouped[key] = bucket
        bucket.hit_count += record.hit_count
        bucket.total_cycles += record.total_cycles
        bucket.source_span_count += 1
        bucket.pc_start = min(bucket.pc_start, record.pc_start)
        bucket.pc_end = max(bucket.pc_end, record.pc_end)
        bucket.isa_spans.append((record.isa_offset, record.isa_end_offset))
        for category_name, category_count in record.categories:
            bucket.categories[category_name] += category_count
    lines: list[SourceIsaRegionLine] = []
    for bucket in sorted(
        grouped.values(),
        key=lambda item: (
            min(span[0] for span in item.isa_spans),
            -1 if item.record.line is None else item.record.line,
            -1 if item.record.column is None else item.record.column,
            -1 if item.record.spirv_offset is None else item.record.spirv_offset,
        ),
    ):
        record = bucket.record
        lines.append(
            SourceIsaRegionLine(
                source_kind=record.source_kind,
                source_label=record.source_label,
                line=record.line,
                column=record.column,
                spirv_offset=record.spirv_offset,
                source_text=record.source_text,
                hit_count=bucket.hit_count,
                total_cycles=bucket.total_cycles,
                source_span_count=bucket.source_span_count,
                pc_start=bucket.pc_start,
                pc_end=bucket.pc_end,
                isa_spans=_merge_spans(tuple(bucket.isa_spans)),
                categories=tuple(bucket.categories.most_common()),
            )
        )
    return tuple(lines)


def _disasm_span(
    artifact: DriverPipelineArtifact,
    pc_range: DriverSourceSpan,
    *,
    disasm_cache: dict[str, CompilerNativeDisasmIndex],
) -> tuple[DisasmInstruction, ...]:
    disasm_index = disasm_cache.get(artifact.pipeline_name)
    if disasm_index is None:
        disasm_index = parse_compiler_native_disasm(Path(artifact.compiler_native_disasm_path))
        disasm_cache[artifact.pipeline_name] = disasm_index
    return tuple(
        instruction
        for instruction in disasm_index.instructions
        if instruction.offset < pc_range.isa_end_offset and instruction.end_offset > pc_range.isa_offset
    )


def _source_text(*, glsl_path: str, line: int | None, source_lines_cache: dict[str, list[str]]) -> str:
    if line is None:
        return ""
    lines = source_lines_cache.get(glsl_path)
    if lines is None:
        lines = Path(glsl_path).read_text(encoding="utf-8").splitlines()
        source_lines_cache[glsl_path] = lines
    if 1 <= line <= len(lines):
        return lines[line - 1].rstrip()
    return ""


def _source_label(pc_range: DriverSourceSpan) -> str:
    if pc_range.source_kind == "compiler_generated":
        return "compiler-generated"
    if pc_range.line is None:
        return "glsl"
    if pc_range.spirv_offset is not None:
        return f"line {pc_range.line} @ spirv {pc_range.spirv_offset}"
    return f"line {pc_range.line}"


def _availability(*, total_instruction_event_count: int, matched_instruction_event_count: int) -> tuple[str, str | None]:
    if total_instruction_event_count == 0:
        return (
            "no_instruction_events",
            "Official SQTT decoder returned no WAVE instruction events for this capture. "
            "There is no trustworthy source->ISA->SQTT hotspot signal to report yet.",
        )
    if matched_instruction_event_count == 0:
        return (
            "no_attributed_events",
            "SQTT instruction events exist, but none matched the driver-exported instruction_debug_map. "
            "The source/ISA/hotspot loop is still incomplete.",
        )
    return ("available", None)


def _merge_spans(spans: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not spans:
        return ()
    ordered = sorted(spans)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1]:
            previous[1] = max(previous[1], end)
            continue
        merged.append([start, end])
    return tuple((start, end) for start, end in merged)
