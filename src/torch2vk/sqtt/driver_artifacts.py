"""Load driver-emitted pipeline artifact bundles exported by vendored RADV."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeGuard


PIPELINE_DEBUG_FILENAME = "pipeline-debug.json"
COMPILER_NATIVE_DISASM_FILENAME = "compiler-native-disasm.s"
PIPELINE_ARTIFACT_FORMAT_V6 = "agentorch-radv-pipeline-artifact-v6"
SUPPORTED_PIPELINE_ARTIFACT_FORMATS = frozenset({PIPELINE_ARTIFACT_FORMAT_V6})


@dataclass(frozen=True, slots=True)
class DriverCompileRegime:
    driver_artifacts_enabled: bool
    keep_shader_info: bool
    capture_shaders: bool
    skip_shaders_cache: bool
    dump_shaders_debug_flag: bool
    nir_debug_info_debug_flag: bool
    effective_nir_debug_info: bool
    rgp_trace_enabled: bool


@dataclass(frozen=True, slots=True)
class DriverSourceSpan:
    pc_start: int
    pc_end: int
    isa_offset: int
    isa_end_offset: int
    source_kind: Literal["glsl", "compiler_generated"]
    file: str | None
    line: int | None
    column: int | None
    spirv_offset: int | None


@dataclass(frozen=True, slots=True)
class DriverInstructionDebugRecord(DriverSourceSpan):
    pass


@dataclass(frozen=True, slots=True)
class DriverShaderArtifact:
    stage: int
    stage_name: str
    code_size: int
    exec_size: int
    sgpr_count: int
    vgpr_count: int
    scratch_memory_size: int
    lds_size: int
    wave_size: int
    shader_va: int
    entry_pc: int
    entry_pc_end: int
    raw_debug_info_count: int
    instruction_debug_record_count: int
    instruction_debug_map: tuple[DriverInstructionDebugRecord, ...]


@dataclass(frozen=True, slots=True)
class DriverPipelineArtifact:
    pipeline_hash: tuple[int, int]
    pipeline_name: str
    pipeline_type: int
    compile_regime: DriverCompileRegime
    bundle_dir: str
    pipeline_debug_path: str
    compiler_native_disasm_path: str
    shaders: tuple[DriverShaderArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DriverInstructionDebugMatch:
    pipeline_name: str
    pipeline_hash: tuple[int, int]
    shader_stage_name: str
    shader_va: int
    instruction_debug_record: DriverInstructionDebugRecord


@dataclass(frozen=True, slots=True)
class DriverInstructionDebugIndexEntry:
    pc_start: int
    pc_end: int
    match: DriverInstructionDebugMatch


@dataclass(frozen=True, slots=True)
class DriverInstructionDebugIndex:
    starts: tuple[int, ...]
    entries: tuple[DriverInstructionDebugIndexEntry, ...]


def load_driver_artifacts(root_dir: Path) -> dict[str, DriverPipelineArtifact]:
    if not root_dir.exists():
        raise RuntimeError(f"Driver artifacts root does not exist: {root_dir}")
    artifacts: dict[str, DriverPipelineArtifact] = {}
    for bundle_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        debug_path = bundle_dir / PIPELINE_DEBUG_FILENAME
        disasm_path = bundle_dir / COMPILER_NATIVE_DISASM_FILENAME
        if not debug_path.exists():
            raise RuntimeError(f"Driver artifact bundle is missing {PIPELINE_DEBUG_FILENAME}: {bundle_dir}")
        if not disasm_path.exists():
            raise RuntimeError(f"Driver artifact bundle is missing {COMPILER_NATIVE_DISASM_FILENAME}: {bundle_dir}")
        if disasm_path.stat().st_size == 0:
            raise RuntimeError(f"Driver artifact disasm is empty: {disasm_path}")
        payload_object: object = json.loads(debug_path.read_text(encoding="utf-8"))
        if not _is_object_map(payload_object):
            raise RuntimeError(f"{debug_path} must contain a JSON object")
        artifact = _parse_driver_pipeline_artifact(
            payload=payload_object,
            bundle_dir=bundle_dir,
            debug_path=debug_path,
        )
        existing = artifacts.get(artifact.pipeline_name)
        if existing is not None:
            raise RuntimeError(
                f"Duplicate driver artifact pipeline name {artifact.pipeline_name!r} in {root_dir}: "
                f"{existing.pipeline_hash} and {artifact.pipeline_hash}"
            )
        artifacts[artifact.pipeline_name] = artifact
    return artifacts


def build_driver_instruction_debug_index(
    artifacts: dict[str, DriverPipelineArtifact],
) -> DriverInstructionDebugIndex:
    entries: list[DriverInstructionDebugIndexEntry] = []
    for pipeline in artifacts.values():
        for shader in pipeline.shaders:
            for record in shader.instruction_debug_map:
                entries.append(
                    DriverInstructionDebugIndexEntry(
                        pc_start=record.pc_start,
                        pc_end=record.pc_end,
                        match=DriverInstructionDebugMatch(
                            pipeline_name=pipeline.pipeline_name,
                            pipeline_hash=pipeline.pipeline_hash,
                            shader_stage_name=shader.stage_name,
                            shader_va=shader.shader_va,
                            instruction_debug_record=record,
                        ),
                    )
                )
    entries.sort(key=lambda item: item.pc_start)
    return DriverInstructionDebugIndex(
        starts=tuple(item.pc_start for item in entries),
        entries=tuple(entries),
    )


def find_driver_instruction_debug_record_in_index(
    index: DriverInstructionDebugIndex,
    *,
    pc: int,
) -> DriverInstructionDebugMatch | None:
    position = bisect_right(index.starts, pc) - 1
    if position < 0:
        return None
    entry = index.entries[position]
    if not (entry.pc_start <= pc < entry.pc_end):
        return None
    return entry.match


def _parse_driver_pipeline_artifact(
    *,
    payload: dict[str, object],
    bundle_dir: Path,
    debug_path: Path,
) -> DriverPipelineArtifact:
    format_name = payload.get("format")
    if format_name not in SUPPORTED_PIPELINE_ARTIFACT_FORMATS:
        raise RuntimeError(f"{debug_path} has unexpected driver artifact format: {format_name!r}")
    pipeline_hash_payload = _require_list(payload, "pipeline_hash", path=debug_path)
    pipeline_hash = _require_int_pair(pipeline_hash_payload, "pipeline_hash", path=debug_path)
    if pipeline_hash is None:
        raise RuntimeError(f"{debug_path} field 'pipeline_hash' must be [int, int], got {pipeline_hash_payload!r}")
    disasm_file = _require_str(payload, "compiler_native_disasm_file", path=debug_path)
    if disasm_file != COMPILER_NATIVE_DISASM_FILENAME:
        raise RuntimeError(
            f"{debug_path} field 'compiler_native_disasm_file' must be {COMPILER_NATIVE_DISASM_FILENAME!r}, "
            f"got {disasm_file!r}"
        )
    shaders_payload = _require_list(payload, "shaders", path=debug_path)
    compile_regime_payload = _require_dict(payload, "compile_regime", path=debug_path)
    shaders: list[DriverShaderArtifact] = []
    for shader_payload in shaders_payload:
        if not _is_object_map(shader_payload):
            raise RuntimeError(f"{debug_path} contains non-object shader artifact: {shader_payload!r}")
        instruction_debug_map_payload = _require_list(shader_payload, "instruction_debug_map", path=debug_path)
        instruction_debug_map: list[DriverInstructionDebugRecord] = []
        for record_payload in instruction_debug_map_payload:
            if not _is_object_map(record_payload):
                raise RuntimeError(
                    f"{debug_path} contains non-object driver instruction debug record: {record_payload!r}"
                )
            instruction_debug_map.append(_parse_driver_source_span(record=record_payload, path=debug_path))
        shaders.append(
            DriverShaderArtifact(
                stage=_require_int(shader_payload, "stage", path=debug_path),
                stage_name=_require_str(shader_payload, "stage_name", path=debug_path),
                code_size=_require_int(shader_payload, "code_size", path=debug_path),
                exec_size=_require_int(shader_payload, "exec_size", path=debug_path),
                sgpr_count=_require_int(shader_payload, "sgpr_count", path=debug_path),
                vgpr_count=_require_int(shader_payload, "vgpr_count", path=debug_path),
                scratch_memory_size=_require_int(shader_payload, "scratch_memory_size", path=debug_path),
                lds_size=_require_int(shader_payload, "lds_size", path=debug_path),
                wave_size=_require_int(shader_payload, "wave_size", path=debug_path),
                shader_va=_require_int(shader_payload, "shader_va", path=debug_path),
                entry_pc=_require_int(shader_payload, "entry_pc", path=debug_path),
                entry_pc_end=_require_int(shader_payload, "entry_pc_end", path=debug_path),
                raw_debug_info_count=_require_int(shader_payload, "raw_debug_info_count", path=debug_path),
                instruction_debug_record_count=_require_int_with_default(
                    shader_payload,
                    "instruction_debug_record_count",
                    default=len(instruction_debug_map),
                    path=debug_path,
                ),
                instruction_debug_map=tuple(instruction_debug_map),
            )
        )
    return DriverPipelineArtifact(
        pipeline_hash=pipeline_hash,
        pipeline_name=_require_str(payload, "pipeline_name", path=debug_path),
        pipeline_type=_require_int(payload, "pipeline_type", path=debug_path),
        compile_regime=DriverCompileRegime(
            driver_artifacts_enabled=_require_bool(compile_regime_payload, "driver_artifacts_enabled", path=debug_path),
            keep_shader_info=_require_bool(compile_regime_payload, "keep_shader_info", path=debug_path),
            capture_shaders=_require_bool(compile_regime_payload, "capture_shaders", path=debug_path),
            skip_shaders_cache=_require_bool(compile_regime_payload, "skip_shaders_cache", path=debug_path),
            dump_shaders_debug_flag=_require_bool(compile_regime_payload, "dump_shaders_debug_flag", path=debug_path),
            nir_debug_info_debug_flag=_require_bool(
                compile_regime_payload, "nir_debug_info_debug_flag", path=debug_path
            ),
            effective_nir_debug_info=_require_bool(compile_regime_payload, "effective_nir_debug_info", path=debug_path),
            rgp_trace_enabled=_require_bool(compile_regime_payload, "rgp_trace_enabled", path=debug_path),
        ),
        bundle_dir=str(bundle_dir),
        pipeline_debug_path=str(debug_path),
        compiler_native_disasm_path=str(bundle_dir / COMPILER_NATIVE_DISASM_FILENAME),
        shaders=tuple(shaders),
    )


def _require_str(record: dict[str, object], key: str, *, path: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or value == "":
        raise RuntimeError(f"{path} field {key!r} must be a non-empty string, got {value!r}")
    return value


def _require_optional_str(record: dict[str, object], key: str, *, path: Path) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"{path} field {key!r} must be string|null, got {value!r}")
    return value


def _require_int(record: dict[str, object], key: str, *, path: Path) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"{path} field {key!r} must be int, got {value!r}")
    return value


def _require_optional_int(record: dict[str, object], key: str, *, path: Path) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise RuntimeError(f"{path} field {key!r} must be int|null, got {value!r}")
    return value


def _require_bool(record: dict[str, object], key: str, *, path: Path) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"{path} field {key!r} must be bool, got {value!r}")
    return value


def _require_dict(record: dict[str, object], key: str, *, path: Path) -> dict[str, object]:
    value = record.get(key)
    if not _is_object_map(value):
        raise RuntimeError(f"{path} field {key!r} must be object, got {value!r}")
    return value


def _require_list(record: dict[str, object], key: str, *, path: Path) -> list[object]:
    value = record.get(key)
    if not _is_object_list(value):
        raise RuntimeError(f"{path} field {key!r} must be list, got {value!r}")
    return value


def _require_int_pair(value: list[object], field_name: str, *, path: Path) -> tuple[int, int] | None:
    if len(value) != 2:
        return None
    first = value[0]
    second = value[1]
    if not isinstance(first, int) or not isinstance(second, int):
        return None
    return (first, second)


def _is_object_map(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _require_source_kind(
    record: dict[str, object],
    *,
    path: Path,
) -> Literal["glsl", "compiler_generated"]:
    value = record.get("source_kind")
    if value not in ("glsl", "compiler_generated"):
        raise RuntimeError(
            f"{path} field 'source_kind' must be 'glsl' or 'compiler_generated', got {value!r}"
        )
    return value


def _parse_driver_source_span(
    *,
    record: dict[str, object],
    path: Path,
) -> DriverInstructionDebugRecord:
    return DriverInstructionDebugRecord(
        pc_start=_require_int(record, "pc_start", path=path),
        pc_end=_require_int(record, "pc_end", path=path),
        isa_offset=_require_int(record, "isa_offset", path=path),
        isa_end_offset=_require_int(record, "isa_end_offset", path=path),
        source_kind=_require_source_kind(record, path=path),
        file=_require_optional_str(record, "file", path=path),
        line=_require_optional_int(record, "line", path=path),
        column=_require_optional_int(record, "column", path=path),
        spirv_offset=_require_optional_int(record, "spirv_offset", path=path),
    )


def _require_int_with_default(record: dict[str, object], key: str, *, default: int, path: Path) -> int:
    value = record.get(key)
    if value is None:
        return default
    if not isinstance(value, int):
        raise RuntimeError(f"{path} field {key!r} must be int, got {value!r}")
    return value
