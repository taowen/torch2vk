"""Compare torch2vk profiler output with llama.cpp Vulkan GLSL dumps.

Example:
    uv run python -m torch2vk.profile_diff.llama_glsl \
        --torch2vk-profile /tmp/torch2vk_qwen3_profile \
        --llama-cpp-root ~/projects/agentorch/third_party/llama.cpp \
        --llama-model dist/llama_cpp_qwen3/qwen3-0.6b-q4_k_m.gguf \
        --prompt-tokens 33 \
        --decode-tokens 1 \
        --out /tmp/qwen3_glsl_diff
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from torch2vk.runtime.profile_features import (
    ShaderFeatures,
    classify_op_group,
    empty_shader_features,
    scan_shader_features,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LLAMA_CPP_ROOT = Path.home() / "projects" / "agentorch" / "third_party" / "llama.cpp"
DEFAULT_DISPATCH_FILTER = ""
LLAMA_SHADER_SOURCES = (
    "mul_mm.comp",
    "mul_mat_vecq.comp",
    "mul_mat_vec_q4_k.comp",
    "mul_mat_vec_q6_k.comp",
    "flash_attn.comp",
    "rms_norm.comp",
    "rope_norm.comp",
)


@dataclass(frozen=True, slots=True)
class TorchShaderHotspot:
    shader: str
    op_group: str
    total_ns: int
    dispatch_count: int
    source_path: str | None
    features: ShaderFeatures
    shape_hint: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LlamaDispatchGroup:
    pipeline: str
    op_group: str
    dispatch_count: int
    workgroups: tuple[str, ...]
    elements: tuple[str, ...]
    source_path: str | None
    original_spv_hash: str | None
    runtime_spv_hash: str | None
    runtime_spv_path: str | None
    features: ShaderFeatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch2vk-profile", required=True, type=Path)
    parser.add_argument("--llama-cpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--llama-model", required=True, type=Path)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--dispatch-filter", default=DEFAULT_DISPATCH_FILTER)
    parser.add_argument("--max-dispatches", type=int, default=64)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch_profile = args.torch2vk_profile.expanduser().resolve()
    llama_root = args.llama_cpp_root.expanduser().resolve()
    llama_model = args.llama_model.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    glsl_dir = out / "llama-glsl"
    spv_dir = out / "llama-spv"
    dispatch_dir = out / "llama-dispatch"
    glsl_dir.mkdir(parents=True, exist_ok=True)
    spv_dir.mkdir(parents=True, exist_ok=True)
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    _generate_llama_glsl(llama_root=llama_root, glsl_dir=glsl_dir, spv_dir=spv_dir)
    llama_cases = _run_llama_bench(
        llama_root=llama_root,
        model=llama_model,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        repetitions=args.repetitions,
        dispatch_filter=args.dispatch_filter,
        max_dispatches=args.max_dispatches,
        glsl_dir=glsl_dir,
        dispatch_dir=dispatch_dir,
        out=out,
    )

    torch_rows = _load_jsonl(torch_profile / "dispatches.jsonl")
    torch_summary = _load_json(torch_profile / "summary.json")
    torch_hotspots = _torch_hotspots(torch_rows)
    llama_glsl_by_hash = _llama_glsl_by_original_spv_hash(glsl_dir)
    llama_groups = _llama_dispatch_groups(dispatch_dir, llama_glsl_by_hash)

    _write_json(out / "torch2vk-dispatches.json", [asdict(hotspot) for hotspot in torch_hotspots])
    _write_json(out / "llama-dispatches.json", [asdict(group) for group in llama_groups])
    (out / "report.md").write_text(
        _render_report(
            torch_profile=torch_profile,
            torch_summary=torch_summary,
            torch_hotspots=torch_hotspots,
            llama_cases=llama_cases,
            llama_groups=llama_groups,
        ),
        encoding="utf-8",
    )
    print(f"report written: {out / 'report.md'}")


def _generate_llama_glsl(*, llama_root: Path, glsl_dir: Path, spv_dir: Path) -> None:
    generator = llama_root / "build" / "Release" / "vulkan-shaders-gen"
    if not generator.is_file():
        raise FileNotFoundError(f"vulkan-shaders-gen not found: {generator}")
    shader_root = llama_root / "ggml" / "src" / "ggml-vulkan" / "vulkan-shaders"
    env = os.environ.copy()
    env["GGML_VULKAN_DUMP_GLSL_DIR"] = str(glsl_dir)
    for source_name in LLAMA_SHADER_SOURCES:
        source = shader_root / source_name
        if not source.is_file():
            raise FileNotFoundError(f"llama.cpp shader source not found: {source}")
        source_out = spv_dir / source.stem
        source_out.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(generator),
                "--source",
                str(source),
                "--output-dir",
                str(source_out),
                "--target-hpp",
                str(source_out / "generated.hpp"),
                "--target-cpp",
                str(source_out / "generated.cpp"),
            ],
            env=env,
        )


def _run_llama_bench(
    *,
    llama_root: Path,
    model: Path,
    prompt_tokens: int,
    decode_tokens: int,
    repetitions: int,
    dispatch_filter: str,
    max_dispatches: int,
    glsl_dir: Path,
    dispatch_dir: Path,
    out: Path,
) -> list[dict[str, Any]]:
    bench = llama_root / "build" / "bin" / "llama-bench"
    if not bench.is_file():
        raise FileNotFoundError(f"llama-bench not found: {bench}")
    env = os.environ.copy()
    env["GGML_VULKAN_DUMP_DISPATCH_DIR"] = str(dispatch_dir)
    if dispatch_filter:
        env["GGML_VULKAN_DUMP_DISPATCH_FILTER"] = dispatch_filter
    env["GGML_VULKAN_DUMP_DISPATCH_VALUES"] = "0"
    env["GGML_VULKAN_DUMP_DISPATCH_START"] = "0"
    env["GGML_VULKAN_DUMP_MAX_DISPATCHES"] = str(max_dispatches)
    completed = _run(
        [
            str(bench),
            "-m",
            str(model),
            "-p",
            str(prompt_tokens),
            "-n",
            str(decode_tokens),
            "-ngl",
            "99",
            "-fa",
            "1",
            "-r",
            str(repetitions),
            "-o",
            "json",
        ],
        env=env,
    )
    (out / "llama-bench.json").write_text(completed.stdout, encoding="utf-8")
    return _parse_llama_json(completed.stdout)


def _torch_hotspots(rows: Sequence[dict[str, Any]]) -> list[TorchShaderHotspot]:
    timing_field = "elapsed_ns" if any(row.get("phase") == "replay" for row in rows) else "elapsed_wall_ns"
    timing_rows = [
        row
        for row in rows
        if row.get(timing_field) is not None
        and (timing_field == "elapsed_wall_ns" or row.get("phase") == "replay")
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in timing_rows:
        shader = str(row.get("shader", ""))
        op_group = str(row.get("op_group", classify_op_group(shader=shader)))
        grouped[(shader, op_group)].append(row)

    hotspots = []
    for (shader, op_group), group_rows in grouped.items():
        total_ns = sum(int(row[timing_field]) for row in group_rows)
        first = group_rows[0]
        features = first.get("shader_features")
        if not isinstance(features, dict):
            features = empty_shader_features()
        hotspots.append(
            TorchShaderHotspot(
                shader=shader,
                op_group=op_group,
                total_ns=total_ns,
                dispatch_count=len(group_rows),
                source_path=_optional_str(first.get("shader_glsl_path")),
                features=cast(ShaderFeatures, features),
                shape_hint=cast(dict[str, Any], first.get("shape_hint", {})),
            )
        )
    hotspots.sort(key=lambda item: item.total_ns, reverse=True)
    return hotspots


def _llama_glsl_by_original_spv_hash(glsl_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for path in sorted(glsl_dir.glob("*.artifact.txt")):
        fields = _parse_key_value_file(path)
        compiled_hash = fields.get("compiled_spv_hash_fnv1a64")
        glsl_path = fields.get("glsl_path")
        if compiled_hash and glsl_path:
            paths[compiled_hash] = glsl_path
    return paths


def _llama_dispatch_groups(
    dispatch_dir: Path,
    glsl_by_original_spv_hash: Mapping[str, str],
) -> list[LlamaDispatchGroup]:
    grouped: dict[tuple[str, str | None], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(dispatch_dir.glob("*-dispatch.txt"), key=_dispatch_sort_key):
        fields = _parse_key_value_file(path)
        pipeline = fields.get("pipeline", path.name)
        source_path = glsl_by_original_spv_hash.get(fields.get("original_spv_hash_fnv1a64", ""))
        grouped[(pipeline, source_path)].append(fields)

    result = []
    for (pipeline, source_path), rows in grouped.items():
        op_group = classify_op_group(shader=pipeline)
        features = (
            scan_shader_features(source_path)
            if source_path is not None and Path(source_path).is_file()
            else empty_shader_features()
        )
        result.append(
            LlamaDispatchGroup(
                pipeline=pipeline,
                op_group=op_group,
                dispatch_count=len(rows),
                workgroups=tuple(_unique(row.get("workgroups", "") for row in rows)),
                elements=tuple(_unique(row.get("elements", "") for row in rows)),
                source_path=source_path,
                original_spv_hash=rows[0].get("original_spv_hash_fnv1a64"),
                runtime_spv_hash=rows[0].get("runtime_spv_hash_fnv1a64"),
                runtime_spv_path=rows[0].get("runtime_spv_path"),
                features=features,
            )
        )
    result.sort(key=lambda item: (item.op_group, item.pipeline))
    return result


def _render_report(
    *,
    torch_profile: Path,
    torch_summary: dict[str, Any],
    torch_hotspots: Sequence[TorchShaderHotspot],
    llama_cases: Sequence[dict[str, Any]],
    llama_groups: Sequence[LlamaDispatchGroup],
) -> str:
    llama_by_group: dict[str, list[LlamaDispatchGroup]] = defaultdict(list)
    for group in llama_groups:
        llama_by_group[group.op_group].append(group)

    lines = [
        "# GLSL Diff Report",
        "",
        "## Performance Summary",
        "",
        f"- Torch2VK profile: `{torch_profile}`",
        f"- Torch2VK replay plans: {_format_replay_plans(torch_summary)}",
        f"- llama.cpp cases: {_format_llama_cases(llama_cases)}",
        "",
        "## Top Torch2VK Hotspots",
        "",
    ]
    for hotspot in torch_hotspots[:20]:
        lines.extend(
            [
                f"### {hotspot.shader}",
                "",
                f"- op_group: `{hotspot.op_group}`",
                f"- total: `{hotspot.total_ns / 1_000_000:.3f} ms` across "
                f"`{hotspot.dispatch_count}` dispatches",
                f"- features: {_format_features(hotspot.features)}",
                f"- source: `{hotspot.source_path or '<missing>'}`",
                f"- shape_hint: `{_short_json(hotspot.shape_hint)}`",
                "",
                "**Likely llama.cpp matches**",
                "",
            ]
        )
        candidates = llama_by_group.get(hotspot.op_group, [])
        if not candidates:
            lines.extend(["- none in captured llama.cpp dispatches", ""])
            continue
        for candidate in candidates[:5]:
            lines.extend(
                [
                    f"- `{candidate.pipeline}`: `{candidate.dispatch_count}` dispatches, "
                    f"workgroups `{', '.join(candidate.workgroups)}`, "
                    f"features {_format_features(candidate.features)}, "
                    f"generated GLSL `{candidate.source_path or '<missing>'}`, "
                    f"runtime SPV `{candidate.runtime_spv_path or '<missing>'}`",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Captured llama.cpp Dispatch Groups",
            "",
        ]
    )
    for group in llama_groups:
        lines.append(
            f"- `{group.pipeline}` ({group.op_group}): `{group.dispatch_count}` dispatches, "
            f"elements `{', '.join(group.elements)}`, "
            f"original SPV `{group.original_spv_hash or '<missing>'}`, "
            f"runtime SPV `{group.runtime_spv_hash or '<missing>'}`, "
            f"generated GLSL `{group.source_path or '<missing>'}`"
        )
    lines.append("")
    return "\n".join(lines)


def _format_replay_plans(summary: dict[str, Any]) -> str:
    replay = summary.get("replay")
    if not isinstance(replay, dict):
        return "`<missing>`"
    plans = replay.get("plans")
    if not isinstance(plans, dict) or not plans:
        return "`<none>`"
    parts = []
    for name, stats in plans.items():
        if isinstance(stats, dict):
            median = int(stats.get("median_ns", 0))
            parts.append(f"`{name}` {median / 1_000_000:.3f} ms")
    return ", ".join(parts)


def _format_llama_cases(cases: Sequence[dict[str, Any]]) -> str:
    parts = []
    for case in cases:
        prompt = case.get("n_prompt")
        generated = case.get("n_gen")
        avg_ns = case.get("avg_ns")
        avg_ts = case.get("avg_ts")
        if isinstance(avg_ns, int | float):
            parts.append(
                f"`p={prompt} n={generated}` {float(avg_ns) / 1_000_000:.3f} ms, "
                f"{float(avg_ts or 0.0):.1f} tok/s"
            )
    return ", ".join(parts) if parts else "`<missing>`"


def _format_features(features: ShaderFeatures) -> str:
    enabled = [name for name, value in features.items() if value]
    return "`" + ", ".join(enabled or ["none"]) + "`"


def _short_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:500]


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return cast(dict[str, Any], loaded)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected JSON object line in {path}")
        rows.append(cast(dict[str, Any], loaded))
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_llama_json(output: str) -> list[dict[str, Any]]:
    start = output.find("[")
    if start < 0:
        raise ValueError(f"llama-bench did not print a JSON array:\n{output}")
    loaded = json.loads(output[start:])
    if not isinstance(loaded, list):
        raise TypeError(f"Expected llama-bench JSON list, got {type(loaded).__name__}")
    return cast(list[dict[str, Any]], loaded)


def _parse_key_value_file(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _dispatch_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.name.split("-", 1)[0]
    return (int(prefix) if prefix.isdigit() else 0, path.name)


def _unique(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _run(command: Sequence[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


if __name__ == "__main__":
    main()
