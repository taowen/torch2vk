"""Join official SQTT instruction events with driver ISA/source mappings."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .disasm_index import CompilerNativeDisasmIndex, DisasmInstruction, parse_compiler_native_disasm
from .driver_artifacts import (
    DriverPipelineArtifact,
    DriverSourceSpan,
    build_driver_instruction_debug_index,
    find_driver_instruction_debug_record_in_index,
)
from .pipeline_attribution import PipelineAttribution
from .rgp_capture_parser import RgpCapture, parse_rgp_capture
from .source_isa_models import (
    HotRangeAggregate,
    SourceHotLine,
    SourceIsaHotRange,
    SourceIsaSqttHotspotReport,
    _SourceHotLineBucket,
)
from .trace_decoder_isa import DecodedInstructionStreams, decode_instruction_events_by_stream


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
    decoded_stream_count = len(resolved_capture.sqtt_data_chunks)
    for stream_index, _chunk in enumerate(resolved_capture.sqtt_data_chunks):
        stream_events = resolved_decoded_instruction_streams.stream_events(stream_index)
        for event in stream_events:
            total_instruction_event_count += 1
            if event.absolute_pc_address == 0:
                zero_pc_instruction_event_count += 1
                continue
            instruction_match = find_driver_instruction_debug_record_in_index(
                instruction_debug_index,
                pc=event.absolute_pc_address,
            )
            if instruction_match is None:
                unmatched_instruction_event_count += 1
                continue
            pipeline_name = instruction_match.pipeline_name
            source_span = instruction_match.instruction_debug_record
            item = attributed_by_pipeline.get(pipeline_name)
            if item is None:
                unmatched_instruction_event_count += 1
                continue
            matched_instruction_event_count += 1
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
        availability=availability,
        unavailable_reason=unavailable_reason,
        hot_ranges=all_hot_ranges[:top_limit],
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
        lines.append(
            SourceHotLine(
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
        try:
            lines = Path(glsl_path).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
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
