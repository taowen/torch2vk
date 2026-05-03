#!/usr/bin/env python3
"""Verify Qwen3 shader modules inline their GLSL sources."""

from __future__ import annotations

import ast
from pathlib import Path


SHADER_DIR = Path("src/torch2vk/models/qwen3_safetensor/shaders")
REMOVED_SOURCE_NAME = "agent" + "orch"


def main() -> int:
    checked = 0
    for path in sorted(SHADER_DIR.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        _verify_shader_module(path)
        checked += 1
    print(f"qwen3_shader_sources=ok modules={checked}")
    return 0


def _verify_shader_module(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "copied_shader_source" in source or f"{REMOVED_SOURCE_NAME}_shader_source" in source:
        raise ValueError(f"{path} still references removed copied shader helpers")
    module = ast.parse(source, filename=str(path))
    source_names = _inline_source_assignments(module)
    variants = 0
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not _is_shader_variant_call(node.value):
            continue
        variants += 1
        source_expr = _keyword_value(node.value, "source")
        if source_expr is None:
            raise ValueError(f"{path} ShaderVariant is missing source=")
        if not isinstance(source_expr, ast.Name) or source_expr.id not in source_names:
            raise ValueError(f"{path} ShaderVariant.source must reference an inline _SOURCE string")
    if variants != 1:
        raise ValueError(f"{path} must define exactly one ShaderVariant, got {variants}")


def _inline_source_assignments(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        if "#version" not in node.value.value:
            continue
        if REMOVED_SOURCE_NAME in node.value.value:
            raise ValueError("inline GLSL source must not mention removed source paths or symbols")
        if "#include" in node.value.value:
            raise ValueError("inline GLSL source must not depend on external includes")
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_shader_variant_call(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "ShaderVariant"


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


if __name__ == "__main__":
    raise SystemExit(main())
