"""Build SQTT decoder attribution from torch2vk postprocess rows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .driver_artifacts import DriverPipelineArtifact, DriverShaderArtifact, load_driver_artifacts


@dataclass(frozen=True, slots=True)
class RuntimePipelineRecord:
    pipeline_debug_name: str
    shader_variant_name: str
    shader_family: str
    glsl_path: str
    spv_path: str
    dispatch_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttributedCodeObject:
    internal_pipeline_hash: tuple[int, int]
    runtime_pipeline: RuntimePipelineRecord
    driver_artifact: DriverPipelineArtifact
    driver_shader: DriverShaderArtifact


@dataclass(frozen=True, slots=True)
class PipelineAttribution:
    attributed_code_objects: tuple[AttributedCodeObject, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline_attribution_from_attribution_jsonl(
    *,
    root: Path,
    attribution_path: Path | None = None,
) -> PipelineAttribution:
    rows = _load_jsonl(root / "attribution.jsonl" if attribution_path is None else attribution_path)
    driver_artifacts = load_driver_artifacts(root / "driver")
    by_pipeline: dict[str, dict[str, Any]] = {}
    labels_by_pipeline: dict[str, list[str]] = {}

    for row in rows:
        pipeline_name = _expect_str(row, "pipeline_debug_name")
        existing = by_pipeline.get(pipeline_name)
        if existing is None:
            by_pipeline[pipeline_name] = row
            labels_by_pipeline[pipeline_name] = []
        else:
            _assert_same_pipeline(existing, row, pipeline_name=pipeline_name)
        label = _expect_str(row, "profile_tag")
        if label not in labels_by_pipeline[pipeline_name]:
            labels_by_pipeline[pipeline_name].append(label)

    attributed: list[AttributedCodeObject] = []
    for pipeline_name, row in sorted(by_pipeline.items()):
        driver_artifact = driver_artifacts[pipeline_name]
        driver_shader = _require_single_shader(driver_artifact)
        glsl_path = _glsl_path_for_row(row)
        pipeline_hash = driver_artifact.pipeline_hash
        attributed.append(
            AttributedCodeObject(
                internal_pipeline_hash=pipeline_hash,
                runtime_pipeline=RuntimePipelineRecord(
                    pipeline_debug_name=pipeline_name,
                    shader_variant_name=_expect_str(row, "shader"),
                    shader_family=_expect_str(row, "shader"),
                    glsl_path=str(glsl_path),
                    spv_path=_expect_str(row, "shader_spv_path"),
                    dispatch_labels=tuple(labels_by_pipeline[pipeline_name]),
                ),
                driver_artifact=driver_artifact,
                driver_shader=driver_shader,
            )
        )

    return PipelineAttribution(attributed_code_objects=tuple(attributed))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def _assert_same_pipeline(existing: dict[str, Any], row: dict[str, Any], *, pipeline_name: str) -> None:
    keys = ("shader", "shader_spv_path", "pipeline_hash_hex")
    for key in keys:
        if existing.get(key) != row.get(key):
            raise RuntimeError(
                f"attribution rows for pipeline {pipeline_name!r} disagree on {key}: "
                f"{existing.get(key)!r} != {row.get(key)!r}"
            )


def _require_single_shader(artifact: DriverPipelineArtifact) -> DriverShaderArtifact:
    if len(artifact.shaders) != 1:
        raise RuntimeError(
            f"Expected one compute shader in pipeline artifact {artifact.pipeline_name!r}, "
            f"got {len(artifact.shaders)}"
        )
    return artifact.shaders[0]


def _glsl_path_for_row(row: dict[str, Any]) -> Path:
    profiled_source = row.get("shader_glsl_path")
    if isinstance(profiled_source, str) and profiled_source:
        path = Path(profiled_source)
        if path.is_file():
            return path
    spv_path = Path(_expect_str(row, "shader_spv_path"))
    candidates = [
        spv_path.with_suffix(".comp"),
        spv_path.with_suffix(".glsl"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _expect_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or value == "":
        raise RuntimeError(f"{key} must be a non-empty string, got {value!r}")
    return value
