#!/usr/bin/env python3
"""Join torch2vk runtime dispatches with RADV SQTT driver artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ParsedProfileTag(TypedDict):
    frame: str
    shader: str
    dispatch: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="SQTT run root")
    args = parser.parse_args()

    try:
        postprocess(args.root.expanduser().resolve())
    except Exception as exc:
        print(f"postprocess-sqtt: {exc}", file=sys.stderr)
        return 1
    return 0


def postprocess(root: Path) -> None:
    driver_dir = root / "driver"
    dispatches_path = root / "dispatches.jsonl"
    driver_dispatch_path = driver_dir / "dispatch-sequence.jsonl"
    capture_path = driver_dir / "capture-sequence.jsonl"

    manifest_rows = _load_jsonl(dispatches_path)
    driver_dispatch_rows = _load_jsonl(driver_dispatch_path)
    capture_rows = _load_jsonl(capture_path)
    if not manifest_rows:
        raise RuntimeError(f"{dispatches_path} is empty")
    if not driver_dispatch_rows:
        raise RuntimeError(f"{driver_dispatch_path} is empty")
    if not capture_rows:
        raise RuntimeError(f"{capture_path} is empty")

    non_record = [row for row in manifest_rows if row.get("phase") != "record"]
    if non_record:
        raise RuntimeError(
            "SQTT only supports record-mode dispatches; "
            f"found {len(non_record)} non-record manifest row(s)"
        )

    manifest_by_key = _manifest_by_key(manifest_rows)
    captures_by_ordinal = _captures_by_ordinal(capture_rows)
    copied_rgps = _copy_rgp_captures(root=root, capture_rows=capture_rows)

    rows: list[dict[str, Any]] = []
    for raw_driver_row in driver_dispatch_rows:
        tag = _expect_str(raw_driver_row, "profile_tag")
        parsed_tag = _parse_profile_tag(tag)
        key = (parsed_tag["frame"], parsed_tag["shader"], parsed_tag["dispatch"])
        manifest = manifest_by_key.get(key)
        if manifest is None:
            raise RuntimeError(
                "driver dispatch did not join to runtime manifest: "
                f"profile_tag={tag!r}"
            )

        submit_ordinal = _expect_int(raw_driver_row, "submit_ordinal")
        pipeline_hash = _expect_int(raw_driver_row, "pipeline_hash")
        pipeline_hash_hex = f"{pipeline_hash:016x}"
        driver_pipeline_name = _expect_str(raw_driver_row, "pipeline_name")
        runtime_pipeline_name = _expect_str(manifest, "pipeline_debug_name")
        if driver_pipeline_name != runtime_pipeline_name:
            raise RuntimeError(
                "pipeline name mismatch for "
                f"profile_tag={tag!r}: driver={driver_pipeline_name!r} "
                f"runtime={runtime_pipeline_name!r}"
            )

        capture = captures_by_ordinal.get(submit_ordinal)
        if capture is None:
            raise RuntimeError(f"missing capture-sequence row for submit_ordinal={submit_ordinal}")

        pipeline_debug_rel = Path("driver") / pipeline_hash_hex / "pipeline-debug.json"
        disasm_rel = Path("driver") / pipeline_hash_hex / "compiler-native-disasm.s"
        pipeline_debug = _load_optional_json(root / pipeline_debug_rel)
        shader_stats = _shader_stats(pipeline_debug)

        rows.append({
            "dispatch_index": parsed_tag["dispatch"],
            "frame": parsed_tag["frame"],
            "phase": "record",
            "shader": parsed_tag["shader"],
            "submit_ordinal": submit_ordinal,
            "driver_dispatch_index": _expect_int(raw_driver_row, "dispatch_index"),
            "pipeline_hash": pipeline_hash,
            "pipeline_hash_hex": pipeline_hash_hex,
            "pipeline_debug_name": runtime_pipeline_name,
            "pipeline_identity_sha256": _expect_str(manifest, "pipeline_identity_sha256"),
            "shader_spv_sha256": _expect_str(manifest, "shader_spv_sha256"),
            "shader_spv_path": manifest.get("shader_spv_path"),
            "dispatch_size": manifest.get("dispatch_size"),
            "symbols": manifest.get("symbols"),
            "reads": manifest.get("reads"),
            "writes": manifest.get("writes"),
            "output_op": _output_op_name(manifest.get("writes")),
            "profile_tag": tag,
            "rgp_path": _relative_or_str(copied_rgps[submit_ordinal], root),
            "driver_capture_path": capture.get("capture_path"),
            "pipeline_debug_path": _relative_or_str(root / pipeline_debug_rel, root),
            "disasm_path": _relative_or_str(root / disasm_rel, root),
            **shader_stats,
        })

    if not rows:
        raise RuntimeError("no SQTT attribution rows were produced")

    attribution_path = root / "attribution.jsonl"
    _write_jsonl(attribution_path, rows)
    decoder_summary = _write_source_hotspot_reports(root=root, rows=rows)
    summary = _build_summary(rows, capture_rows)
    summary.update(decoder_summary)
    (root / "sqtt-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _update_run_json(root, summary)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    return rows


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _manifest_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            _expect_str(row, "frame"),
            _expect_str(row, "shader"),
            _expect_int(row, "dispatch_index"),
        )
        if key in result:
            raise RuntimeError(f"duplicate runtime dispatch key: {key}")
        result[key] = row
    return result


def _captures_by_ordinal(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        ordinal = _expect_int(row, "submit_ordinal")
        if ordinal in result:
            raise RuntimeError(f"duplicate capture submit_ordinal: {ordinal}")
        result[ordinal] = row
    return result


def _copy_rgp_captures(root: Path, capture_rows: list[dict[str, Any]]) -> dict[int, Path]:
    rgp_dir = root / "rgp"
    rgp_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[int, Path] = {}
    for row in capture_rows:
        submit_ordinal = _expect_int(row, "submit_ordinal")
        source = Path(_expect_str(row, "capture_path"))
        destination = rgp_dir / f"submit{submit_ordinal}.rgp"
        if source.is_file():
            shutil.copy2(source, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(
                f"missing RGP capture for submit_ordinal={submit_ordinal}: {source}"
            )
        copied[submit_ordinal] = destination
    return copied


def _parse_profile_tag(tag: str) -> ParsedProfileTag:
    fields: dict[str, str] = {}
    for part in tag.split(";"):
        if not part:
            continue
        if "=" not in part:
            raise RuntimeError(f"invalid profile_tag segment {part!r} in {tag!r}")
        key, value = part.split("=", 1)
        fields[key] = value
    for required in ("frame", "shader", "dispatch"):
        if required not in fields or not fields[required]:
            raise RuntimeError(f"profile_tag missing {required!r}: {tag!r}")
    return {
        "frame": fields["frame"],
        "shader": fields["shader"],
        "dispatch": int(fields["dispatch"]),
    }


def _shader_stats(pipeline_debug: dict[str, Any] | None) -> dict[str, Any]:
    if pipeline_debug is None:
        return {}
    shaders = pipeline_debug.get("shaders")
    if not isinstance(shaders, list) or not shaders:
        return {}
    shader = shaders[0]
    if not isinstance(shader, dict):
        return {}
    keys = (
        "code_size",
        "exec_size",
        "sgpr_count",
        "vgpr_count",
        "scratch_memory_size",
        "lds_size",
        "wave_size",
        "raw_debug_info_count",
        "instruction_debug_record_count",
    )
    return {
        f"driver_{key}": shader[key]
        for key in keys
        if isinstance(shader.get(key), int)
    }


def _output_op_name(raw_writes: object) -> str:
    if not isinstance(raw_writes, list) or not raw_writes:
        return "<no-write>"
    first = raw_writes[0]
    if not isinstance(first, dict):
        return "<unknown>"
    tensor = first.get("tensor")
    if not isinstance(tensor, str):
        return "<unknown>"
    marker = "text_decode.layers."
    if marker in tensor:
        suffix = tensor.split(marker, 1)[1]
        parts = suffix.split(".", 1)
        if len(parts) == 2:
            return parts[1]
    marker = "text_decode."
    if marker in tensor:
        return tensor.split(marker, 1)[1]
    marker = "token_select."
    if marker in tensor:
        return f"token_select.{tensor.split(marker, 1)[1]}"
    return tensor


def _build_summary(rows: list[dict[str, Any]], capture_rows: list[dict[str, Any]]) -> dict[str, Any]:
    shaders = Counter(str(row["shader"]) for row in rows)
    outputs = Counter(str(row["output_op"]) for row in rows)
    pipelines = Counter(str(row["pipeline_debug_name"]) for row in rows)
    return {
        "captured_dispatches": len(rows),
        "captured_submits": len(capture_rows),
        "shaders": shaders.most_common(),
        "outputs": outputs.most_common(),
        "pipelines": pipelines.most_common(),
    }


def _write_source_hotspot_reports(*, root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from torch2vk.sqtt.pipeline_attribution import build_pipeline_attribution_from_attribution_jsonl
    from torch2vk.sqtt.rgp_capture_parser import parse_rgp_capture
    from torch2vk.sqtt.source_isa_hotspot_report import build_source_isa_hotspot_report

    try:
        pipeline_attribution = build_pipeline_attribution_from_attribution_jsonl(root=root)
        (root / "pipeline-attribution.json").write_text(
            json.dumps(pipeline_attribution.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        return {
            "pipeline_attribution_path": None,
            "source_isa_hotspot_reports": [],
            "source_isa_hotspot_error": f"pipeline attribution failed: {exc}",
        }

    reports: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    unique_rgps = {
        (_expect_int(item, "submit_ordinal"), _expect_str(item, "rgp_path"))
        for item in rows
    }
    rgp_rows = sorted(rows, key=lambda row: (_expect_int(row, "submit_ordinal"), _expect_str(row, "rgp_path")))
    for row in rgp_rows:
        submit_ordinal = _expect_int(row, "submit_ordinal")
        rgp_path_text = _expect_str(row, "rgp_path")
        key = (submit_ordinal, rgp_path_text)
        if key in seen:
            continue
        seen.add(key)

        rgp_path = Path(rgp_path_text)
        if not rgp_path.is_absolute():
            rgp_path = root / rgp_path
        if len(unique_rgps) == 1:
            output_name = "source-isa-sqtt-hotspots.json"
        else:
            output_name = f"source-isa-sqtt-hotspots-submit{submit_ordinal}.json"
        output_path = root / output_name
        try:
            capture = parse_rgp_capture(rgp_path)
            report = build_source_isa_hotspot_report(
                capture_path=rgp_path,
                pipeline_attribution=pipeline_attribution,
                capture=capture,
            )
            payload = report.to_dict()
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            reports.append({
                "submit_ordinal": submit_ordinal,
                "capture_path": _relative_or_str(rgp_path, root),
                "path": _relative_or_str(output_path, root),
                "availability": payload["availability"],
                "unavailable_reason": payload["unavailable_reason"],
                "decoded_stream_count": payload["decoded_stream_count"],
                "total_instruction_event_count": payload["total_instruction_event_count"],
                "matched_instruction_event_count": payload["matched_instruction_event_count"],
                "unmatched_instruction_event_count": payload["unmatched_instruction_event_count"],
                "zero_pc_instruction_event_count": payload["zero_pc_instruction_event_count"],
            })
        except Exception as exc:
            reports.append({
                "submit_ordinal": submit_ordinal,
                "capture_path": _relative_or_str(rgp_path, root),
                "path": None,
                "availability": "error",
                "unavailable_reason": str(exc),
            })

    return {
        "pipeline_attribution_path": "pipeline-attribution.json",
        "source_isa_hotspot_reports": reports,
    }


def _update_run_json(root: Path, sqtt_summary: dict[str, Any]) -> None:
    path = root / "run.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            payload = raw
    payload["sqtt"] = {
        "attribution_path": "attribution.jsonl",
        "summary_path": "sqtt-summary.json",
        **sqtt_summary,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True))
            output.write("\n")


def _expect_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _expect_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
