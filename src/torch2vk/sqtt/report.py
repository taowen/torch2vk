"""Compact SQTT run reports.

The report layer is intentionally small: it keeps facts that are useful at first
read and leaves raw captures, full driver maps, and decoder internals as evidence
or optional debug artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REPORT_JSON = "report.json"
REPORT_MARKDOWN = "report.md"
UNAVAILABLE_DATA = (
    "cache_hit_miss_counters",
    "memory_transaction_counters",
    "wait_dependency_chain",
    "occupancy_from_sgpr_vgpr_lds",
)


def compact_hotspot_focus(
    *,
    attribution_row: dict[str, Any],
    hotspot_payload: dict[str, Any],
    root: Path,
    hotspot_path: str | None = None,
) -> dict[str, Any]:
    source_hot_lines = _expect_list(hotspot_payload.get("source_hot_lines"), "source_hot_lines")
    hot_ranges = _expect_list(hotspot_payload.get("hot_ranges"), "hot_ranges")
    reported_source_cycles = _sum_int_field(source_hot_lines, "total_cycles")
    reported_range_cycles = _sum_int_field(hot_ranges, "total_cycles")
    total_events = _optional_int(hotspot_payload.get("total_instruction_event_count"))
    matched_events = _optional_int(hotspot_payload.get("matched_instruction_event_count"))
    zero_pc_events = _optional_int(hotspot_payload.get("zero_pc_instruction_event_count"))
    result: dict[str, Any] = {
        "submit_ordinal": attribution_row.get("submit_ordinal"),
        "dispatch_index": attribution_row.get("dispatch_index"),
        "frame": attribution_row.get("frame"),
        "shader": attribution_row.get("shader"),
        "output_op": attribution_row.get("output_op"),
        "pipeline_debug_name": attribution_row.get("pipeline_debug_name"),
        "dispatch_size": attribution_row.get("dispatch_size"),
        "symbols": attribution_row.get("symbols"),
        "reads": attribution_row.get("reads"),
        "writes": attribution_row.get("writes"),
        "rgp_path": attribution_row.get("rgp_path"),
        "coverage": {
            "availability": hotspot_payload.get("availability"),
            "matched_instruction_event_count": matched_events,
            "matched_instruction_event_ratio": _ratio(matched_events, total_events),
            "total_instruction_event_count": total_events,
            "zero_pc_instruction_event_count": zero_pc_events,
            "zero_pc_instruction_event_ratio": _ratio(zero_pc_events, total_events),
        },
        "resource_usage": _resource_usage(attribution_row),
        "top_source_lines": [
            _compact_source_line(line, reported_source_cycles)
            for line in source_hot_lines[:5]
            if isinstance(line, dict)
        ],
        "top_isa_ranges": [
            _compact_hot_range(item, reported_range_cycles)
            for item in hot_ranges[:5]
            if isinstance(item, dict)
        ],
    }
    if hotspot_path is not None:
        result["debug_hotspot_path"] = hotspot_path
    glsl_path = _first_str((line.get("glsl_path") for line in source_hot_lines if isinstance(line, dict)))
    if glsl_path:
        result["glsl_path"] = _relative_or_str(Path(glsl_path), root)
    return result


def build_report_payload(
    *,
    root: Path,
    attribution_rows: list[dict[str, Any]],
    capture_rows: list[dict[str, Any]],
    source_isa_reports: list[dict[str, Any]],
    focus: list[dict[str, Any]],
    debug_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "captured_submits": len(capture_rows),
        "captured_dispatches": len(attribution_rows),
        "dispatches": [
            _compact_attribution_row(row)
            for row in attribution_rows
        ],
        "focus": focus,
        "limits": list(UNAVAILABLE_DATA),
        "source_isa_reports": source_isa_reports,
        "artifact_inventory": _artifact_inventory(root=root, attribution_rows=attribution_rows),
        "debug_artifacts": {} if debug_artifacts is None else debug_artifacts,
    }


def write_reports(*, root: Path, report: dict[str, Any]) -> dict[str, str]:
    json_path = root / REPORT_JSON
    markdown_path = root / REPORT_MARKDOWN
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "report_json_path": REPORT_JSON,
        "report_path": REPORT_MARKDOWN,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SQTT Report",
        "",
        f"- Captured submits: {report.get('captured_submits', 0)}",
        f"- Captured dispatches: {report.get('captured_dispatches', 0)}",
    ]
    limits = report.get("limits")
    if isinstance(limits, list) and limits:
        lines.append(f"- Not reported: {_escape_markdown_cell(', '.join(str(item) for item in limits))}")

    _append_source_isa_status(lines, report.get("source_isa_reports"))
    focus_items = report.get("focus")
    if not isinstance(focus_items, list) or not focus_items:
        lines.extend(["", "No source/ISA focus data was available."])
    else:
        for item in focus_items:
            if not isinstance(item, dict):
                continue
            lines.extend([
                "",
                "## Dispatch",
                "",
                f"- Dispatch: {item.get('dispatch_index')} ({item.get('frame')})",
                f"- Shader: `{item.get('shader')}`",
                f"- Output: `{item.get('output_op')}`",
                f"- RGP: `{item.get('rgp_path')}`",
            ])
            debug_hotspot_path = item.get("debug_hotspot_path")
            if isinstance(debug_hotspot_path, str):
                lines.append(f"- Debug hotspot JSON: `{debug_hotspot_path}`")
            coverage = item.get("coverage")
            if isinstance(coverage, dict):
                lines.append(
                    "- Coverage: "
                    f"availability={coverage.get('availability')}, "
                    f"matched={_format_ratio(_optional_float(coverage.get('matched_instruction_event_ratio')))}, "
                    f"zero_pc={_format_ratio(_optional_float(coverage.get('zero_pc_instruction_event_ratio')))}"
                )
            resources = item.get("resource_usage")
            if isinstance(resources, dict):
                lines.append(
                    "- Resources: "
                    f"SGPR={resources.get('sgpr_count')}, "
                    f"VGPR={resources.get('vgpr_count')}, "
                    f"LDS={resources.get('lds_size')}, "
                    f"scratch={resources.get('scratch_memory_size')}, "
                    f"wave={resources.get('wave_size')}"
                )
            lines.extend(["", "### Top Source Lines", ""])
            lines.append("| Rank | Line | Share | Cycles | Source |")
            lines.append("| ---: | ---: | ---: | ---: | --- |")
            for rank, row in enumerate(_dict_items(item.get("top_source_lines")), start=1):
                source = str(row.get("source_text", "")).strip() or "source unavailable"
                lines.append(
                    "| "
                    f"{rank} | {row.get('line')} | "
                    f"{_format_ratio(_optional_float(row.get('reported_cycle_ratio')))} | "
                    f"{row.get('total_cycles')} | "
                    f"`{_escape_markdown_cell(source)}` |"
                )
            lines.extend(["", "### Top ISA Ranges", ""])
            lines.append("| Rank | Line | Opcode | ISA | Share | Cycles | Categories |")
            lines.append("| ---: | ---: | --- | --- | ---: | ---: | --- |")
            for rank, row in enumerate(_dict_items(item.get("top_isa_ranges")), start=1):
                lines.append(
                    "| "
                    f"{rank} | {row.get('line')} | "
                    f"`{_escape_markdown_cell(str(row.get('opcode', '')))}` | "
                    f"`{_escape_markdown_cell(str(row.get('isa_text', '')))}` | "
                    f"{_format_ratio(_optional_float(row.get('reported_cycle_ratio')))} | "
                    f"{row.get('total_cycles')} | "
                    f"{_escape_markdown_cell(_format_pairs(row.get('categories')))} |"
                )

    inventory = report.get("artifact_inventory")
    if isinstance(inventory, list) and inventory:
        lines.extend(["", "## Artifact Inventory", ""])
        lines.append("| Path | Role | Size |")
        lines.append("| --- | --- | ---: |")
        for item in _dict_items(inventory):
            lines.append(
                "| "
                f"`{_escape_markdown_cell(str(item.get('path', '')))}` | "
                f"{_escape_markdown_cell(str(item.get('role', '')))} | "
                f"{_format_bytes(_optional_int(item.get('bytes')))} |"
            )
    return "\n".join(lines) + "\n"


def _append_source_isa_status(lines: list[str], reports: object) -> None:
    rows = list(_dict_items(reports))
    if not rows:
        return
    lines.extend(["", "## Source/ISA Status", ""])
    lines.append("| Submit | Availability | Matched | Total | Zero PC | Reason |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| "
            f"{row.get('submit_ordinal', '')} | "
            f"{_escape_markdown_cell(str(row.get('availability', '')))} | "
            f"{row.get('matched_instruction_event_count', '')} | "
            f"{row.get('total_instruction_event_count', '')} | "
            f"{row.get('zero_pc_instruction_event_count', '')} | "
            f"{_escape_markdown_cell(str(row.get('unavailable_reason') or ''))} |"
        )


def _compact_attribution_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "submit_ordinal",
        "dispatch_index",
        "frame",
        "shader",
        "output_op",
        "pipeline_debug_name",
        "pipeline_hash_hex",
        "dispatch_size",
        "symbols",
        "reads",
        "writes",
        "rgp_path",
        "pipeline_debug_path",
        "disasm_path",
    )
    return {
        key: row.get(key)
        for key in keys
        if key in row
    }


def _compact_source_line(row: dict[str, Any], reported_cycles: int) -> dict[str, Any]:
    total_cycles = _optional_int(row.get("total_cycles"))
    return {
        "line": row.get("line"),
        "source_text": row.get("source_text"),
        "total_cycles": total_cycles,
        "reported_cycle_ratio": _ratio(total_cycles, reported_cycles),
        "hit_count": row.get("hit_count"),
        "hottest_categories": _top_pairs(row.get("hottest_categories"), limit=4),
    }


def _compact_hot_range(row: dict[str, Any], reported_cycles: int) -> dict[str, Any]:
    disasm = _expect_list(row.get("disasm"), "disasm")
    first_instruction = next((item for item in disasm if isinstance(item, dict)), {})
    total_cycles = _optional_int(row.get("total_cycles"))
    return {
        "line": row.get("line"),
        "source_text": row.get("source_text"),
        "opcode": first_instruction.get("opcode"),
        "isa_text": first_instruction.get("text"),
        "total_cycles": total_cycles,
        "reported_cycle_ratio": _ratio(total_cycles, reported_cycles),
        "hit_count": row.get("hit_count"),
        "categories": _top_pairs(row.get("categories"), limit=4),
    }


def _resource_usage(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sgpr_count": row.get("driver_sgpr_count"),
        "vgpr_count": row.get("driver_vgpr_count"),
        "lds_size": row.get("driver_lds_size"),
        "scratch_memory_size": row.get("driver_scratch_memory_size"),
        "wave_size": row.get("driver_wave_size"),
        "code_size": row.get("driver_code_size"),
        "exec_size": row.get("driver_exec_size"),
    }


def _artifact_inventory(*, root: Path, attribution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    _append_file(entries, root=root, relative="attribution.jsonl", role="join table")
    _append_file(entries, root=root, relative="dispatches.jsonl", role="runtime manifest")
    _append_file(entries, root=root, relative="summary.json", role="runtime profiler summary")
    rgp_dir = root / "rgp"
    if rgp_dir.is_dir():
        entries.append({
            "path": "rgp/",
            "role": "raw RGP captures",
            "bytes": _path_size(rgp_dir),
        })
    focused_hashes = sorted(
        {
            row["pipeline_hash_hex"]
            for row in attribution_rows
            if isinstance(row.get("pipeline_hash_hex"), str)
        }
    )
    for pipeline_hash in focused_hashes:
        relative = f"driver/{pipeline_hash}/"
        entries.append({
            "path": relative,
            "role": "focused driver pipeline artifact",
            "bytes": _path_size(root / relative),
        })
    driver_dir = root / "driver"
    if driver_dir.is_dir():
        entries.append({
            "path": "driver/",
            "role": "all driver artifacts for the process",
            "bytes": _path_size(driver_dir),
        })
    cache_dir = root / "mesa-shader-cache"
    if cache_dir.is_dir():
        entries.append({
            "path": "mesa-shader-cache/",
            "role": "Mesa shader cache",
            "bytes": _path_size(cache_dir),
        })
    entries.sort(key=lambda item: int(item.get("bytes", 0)), reverse=True)
    return entries


def _append_file(entries: list[dict[str, Any]], *, root: Path, relative: str, role: str) -> None:
    path = root / relative
    if not path.is_file():
        return
    entries.append({
        "path": relative,
        "role": role,
        "bytes": path.stat().st_size,
    })


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _expect_list(value: object, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a list or tuple, got {type(value).__name__}")
    return list(value)


def _dict_items(value: object) -> Iterable[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _sum_int_field(rows: Iterable[Any], key: str) -> int:
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        total += _optional_int(row.get(key))
    return total


def _optional_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _optional_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _format_ratio(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    current = float(value)
    unit = units[0]
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            break
        current /= 1024.0
    if unit == "B":
        return f"{int(current)} B"
    return f"{current:.1f} {unit}"


def _top_pairs(value: object, *, limit: int) -> list[list[Any]]:
    if not isinstance(value, list | tuple):
        return []
    result: list[list[Any]] = []
    for item in value[:limit]:
        if (
            isinstance(item, list | tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], int)
        ):
            result.append([item[0], item[1]])
    return result


def _first_str(values: Iterable[object]) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_pairs(value: object) -> str:
    pairs = _top_pairs(value, limit=4)
    return ", ".join(f"{name}:{count}" for name, count in pairs)


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
