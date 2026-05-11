"""Join runtime dispatch rows with RADV SQTT driver capture rows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, TypedDict

from .postprocess_common import (
    expect_int,
    expect_str,
    load_optional_json,
    relative_or_str,
)


class ParsedProfileTag(TypedDict):
    frame: str
    shader: str
    dispatch: int


def build_attribution_rows(
    *,
    root: Path,
    manifest_rows: list[dict[str, Any]],
    driver_dispatch_rows: list[dict[str, Any]],
    capture_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_key = _manifest_by_key(manifest_rows)
    captures_by_ordinal = _captures_by_ordinal(capture_rows)
    copied_rgps = _copy_rgp_captures(root=root, capture_rows=capture_rows)

    rows: list[dict[str, Any]] = []
    for raw_driver_row in driver_dispatch_rows:
        tag = expect_str(raw_driver_row, "profile_tag")
        parsed_tag = _parse_profile_tag(tag)
        key = (parsed_tag["frame"], parsed_tag["shader"], parsed_tag["dispatch"])
        manifest = manifest_by_key.get(key)
        if manifest is None:
            raise RuntimeError(
                "driver dispatch did not join to runtime manifest: "
                f"profile_tag={tag!r}"
            )

        submit_ordinal = expect_int(raw_driver_row, "submit_ordinal")
        pipeline_hash = expect_int(raw_driver_row, "pipeline_hash")
        pipeline_hash_hex = f"{pipeline_hash:016x}"
        driver_pipeline_name = expect_str(raw_driver_row, "pipeline_name")
        runtime_pipeline_name = expect_str(manifest, "pipeline_debug_name")
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
        pipeline_debug = load_optional_json(root / pipeline_debug_rel)
        shader_stats = _shader_stats(pipeline_debug)

        rows.append({
            "dispatch_index": parsed_tag["dispatch"],
            "frame": parsed_tag["frame"],
            "phase": "record",
            "shader": parsed_tag["shader"],
            "submit_ordinal": submit_ordinal,
            "driver_dispatch_index": expect_int(raw_driver_row, "dispatch_index"),
            "pipeline_hash": pipeline_hash,
            "pipeline_hash_hex": pipeline_hash_hex,
            "pipeline_debug_name": runtime_pipeline_name,
            "pipeline_identity_sha256": expect_str(manifest, "pipeline_identity_sha256"),
            "shader_spv_sha256": expect_str(manifest, "shader_spv_sha256"),
            "shader_spv_path": manifest.get("shader_spv_path"),
            "shader_glsl_path": manifest.get("shader_glsl_path"),
            "dispatch_size": manifest.get("dispatch_size"),
            "symbols": manifest.get("symbols"),
            "reads": manifest.get("reads"),
            "writes": manifest.get("writes"),
            "output_op": _output_op_name(manifest.get("writes")),
            "profile_tag": tag,
            "rgp_path": relative_or_str(copied_rgps[submit_ordinal], root),
            "driver_capture_path": capture.get("capture_path"),
            "pipeline_debug_path": relative_or_str(root / pipeline_debug_rel, root),
            "disasm_path": relative_or_str(root / disasm_rel, root),
            **shader_stats,
        })
    if not rows:
        raise RuntimeError("no SQTT attribution rows were produced")
    return rows


def _manifest_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            expect_str(row, "frame"),
            expect_str(row, "shader"),
            expect_int(row, "dispatch_index"),
        )
        if key in result:
            raise RuntimeError(f"duplicate runtime dispatch key: {key}")
        result[key] = row
    return result


def _captures_by_ordinal(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        ordinal = expect_int(row, "submit_ordinal")
        if ordinal in result:
            raise RuntimeError(f"duplicate capture submit_ordinal: {ordinal}")
        result[ordinal] = row
    return result


def _copy_rgp_captures(root: Path, capture_rows: list[dict[str, Any]]) -> dict[int, Path]:
    rgp_dir = root / "rgp"
    rgp_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[int, Path] = {}
    for row in capture_rows:
        submit_ordinal = expect_int(row, "submit_ordinal")
        source = Path(expect_str(row, "capture_path"))
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
