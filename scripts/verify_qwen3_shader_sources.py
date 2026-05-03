#!/usr/bin/env python3
"""Verify Qwen3 shader wrappers use copied GLSL sources, not inline GLSL."""

from __future__ import annotations

import ast
from pathlib import Path


SHADER_DIR = Path("src/torch2vk/models/qwen3_safetensor/shaders")
COPIED_HELPERS = {"copied_assignment_string", "copied_shader_variant_source"}


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
    if "#version" in source:
        raise ValueError(f"{path} contains inline GLSL #version; copy source instead")
    module = ast.parse(source, filename=str(path))
    source_names = _copied_source_assignments(module)
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
        if not _is_copied_source_expression(source_expr, source_names):
            raise ValueError(f"{path} ShaderVariant.source must come from copied shader source")
    if variants != 1:
        raise ValueError(f"{path} must define exactly one ShaderVariant, got {variants}")


def _copied_source_assignments(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if _is_copied_source_expression(node.value, names):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_copied_source_expression(node: ast.expr, source_names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in source_names
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in COPIED_HELPERS:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "replace":
            return _is_copied_source_expression(node.func.value, source_names)
    return False


def _is_shader_variant_call(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "ShaderVariant"


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


if __name__ == "__main__":
    raise SystemExit(main())
