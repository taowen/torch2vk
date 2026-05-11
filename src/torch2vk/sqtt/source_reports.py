"""Build source/ISA SQTT hotspot summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .postprocess_common import expect_int, expect_str, relative_or_str
from .report import compact_hotspot_focus
from .rgp_capture_parser import parse_rgp_capture


def build_source_isa_reports(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    debug_artifacts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .pipeline_attribution import build_pipeline_attribution_from_attribution_jsonl
    from .source_isa_hotspot_report import build_source_isa_hotspot_report

    debug_outputs: dict[str, Any] = {}
    debug_dir = root / "debug"
    try:
        pipeline_attribution = build_pipeline_attribution_from_attribution_jsonl(root=root)
    except Exception as exc:
        return (
            [{
                "stage": "pipeline_attribution",
                "availability": "error",
                "unavailable_reason": str(exc),
            }],
            [],
            debug_outputs,
        )

    if debug_artifacts:
        debug_dir.mkdir(parents=True, exist_ok=True)
        pipeline_attribution_path = debug_dir / "pipeline-attribution.json"
        pipeline_attribution_path.write_text(
            json.dumps(pipeline_attribution.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        debug_outputs["pipeline_attribution_path"] = relative_or_str(
            pipeline_attribution_path,
            root,
        )

    reports: list[dict[str, Any]] = []
    focus: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    unique_rgps = {
        (expect_int(item, "submit_ordinal"), expect_str(item, "rgp_path"))
        for item in rows
    }
    rgp_rows = sorted(rows, key=lambda row: (expect_int(row, "submit_ordinal"), expect_str(row, "rgp_path")))
    for row in rgp_rows:
        submit_ordinal = expect_int(row, "submit_ordinal")
        rgp_path_text = expect_str(row, "rgp_path")
        key = (submit_ordinal, rgp_path_text)
        if key in seen:
            continue
        seen.add(key)

        rgp_path = Path(rgp_path_text)
        if not rgp_path.is_absolute():
            rgp_path = root / rgp_path
        try:
            capture = parse_rgp_capture(rgp_path)
            report = build_source_isa_hotspot_report(
                capture_path=rgp_path,
                pipeline_attribution=pipeline_attribution,
                capture=capture,
            )
            payload = report.to_dict()
            hotspot_path: str | None = None
            if debug_artifacts:
                if len(unique_rgps) == 1:
                    output_name = "source-isa-sqtt-hotspots.json"
                else:
                    output_name = f"source-isa-sqtt-hotspots-submit{submit_ordinal}.json"
                output_path = debug_dir / output_name
                output_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                hotspot_path = relative_or_str(output_path, root)
            report_summary = {
                "submit_ordinal": submit_ordinal,
                "capture_path": relative_or_str(rgp_path, root),
                "availability": payload["availability"],
                "unavailable_reason": payload["unavailable_reason"],
                "decoded_stream_count": payload["decoded_stream_count"],
                "total_instruction_event_count": payload["total_instruction_event_count"],
                "matched_instruction_event_count": payload["matched_instruction_event_count"],
                "unmatched_instruction_event_count": payload["unmatched_instruction_event_count"],
                "zero_pc_instruction_event_count": payload["zero_pc_instruction_event_count"],
            }
            if hotspot_path is not None:
                report_summary["debug_hotspot_path"] = hotspot_path
                debug_outputs.setdefault("source_isa_hotspot_paths", []).append(hotspot_path)
            try:
                focus.append(
                    compact_hotspot_focus(
                        attribution_row=row,
                        hotspot_payload=payload,
                        hotspot_path=hotspot_path,
                        root=root,
                    )
                )
            except Exception as focus_exc:
                report_summary["focus_error"] = str(focus_exc)
            reports.append(report_summary)
        except Exception as exc:
            reports.append({
                "submit_ordinal": submit_ordinal,
                "capture_path": relative_or_str(rgp_path, root),
                "availability": "error",
                "unavailable_reason": str(exc),
            })

    return reports, focus, debug_outputs
