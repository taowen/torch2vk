#!/usr/bin/env python3
"""Compile every Qwen3 safetensor shader module discovered in its shader directory."""

from __future__ import annotations

import importlib
import pkgutil
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from torch2vk.shader import ShaderVariant


PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"


def main() -> int:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        print("glslangValidator is required to compile Qwen3 shaders", file=sys.stderr)
        return 127

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "build" / "shaders" / "qwen3_safetensor"
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for variant in _shader_variants():
        result = _compile_variant(
            compiler=compiler,
            root=root,
            output_dir=output_dir,
            variant=variant,
        )
        if result.returncode == 0:
            print(f"compiled {variant.name}")
        else:
            failures.append(variant.name)
            print(f"failed {variant.name}", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
    if failures:
        print("shader compilation failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def _shader_variants() -> tuple[ShaderVariant, ...]:
    package = importlib.import_module(PACKAGE)
    variants: list[ShaderVariant] = []
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, ShaderVariant):
                variants.append(value)
    return tuple(sorted(variants, key=lambda item: item.name))


def _compile_variant(
    *,
    compiler: str,
    root: Path,
    output_dir: Path,
    variant: ShaderVariant,
) -> subprocess.CompletedProcess[str]:
    output_path = output_dir / f"{variant.name}.spv"
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / f"{variant.name}.comp"
        source_path.write_text(_source_for_compile(variant.source), encoding="utf-8")
        command = [
            compiler,
            "-V",
            "--target-env",
            "vulkan1.2",
            "-S",
            "comp",
            "-o",
            str(output_path),
        ]
        for include_dir in variant.include_dirs:
            command.append(f"-I{root / 'src' / 'torch2vk' / include_dir}")
        for define in variant.compile_defines:
            command.append(f"-D{define}")
        command.append(str(source_path))
        return subprocess.run(command, capture_output=True, text=True, check=False)


def _source_for_compile(source: str) -> str:
    if "#include" not in source or "GL_GOOGLE_include_directive" in source:
        return source
    lines = source.splitlines()
    if lines and lines[0].startswith("#version"):
        return "\n".join(
            (
                lines[0],
                "#extension GL_GOOGLE_include_directive : enable",
                *lines[1:],
            )
        )
    return "#extension GL_GOOGLE_include_directive : enable\n" + source


if __name__ == "__main__":
    raise SystemExit(main())
