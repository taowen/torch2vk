#!/usr/bin/env python3
"""Verify OmniVoice shader modules inline their migrated GLSL sources."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

from torch2vk.shader import ShaderVariant, validate_shader_source_bindings

SHADER_DIR = Path("src/torch2vk/models/omnivoice_safetensor/shaders")
PACKAGE = "torch2vk.models.omnivoice_safetensor.shaders"
REMOVED_SOURCE_NAME = "agent" + "orch"
REMOVED_AUDIO_HEAD_ABI = (
    "omnivoice_audio_head_mat_vec_f16_f32_f32",
    "omnivoice_audio_head_round_f32_to_f16_f32",
    "omnivoice_audio_head_scalar_f16_f32_f32",
)


def main() -> int:
    checked = 0
    for path in sorted(SHADER_DIR.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        _verify_shader_module(path)
        checked += 1
    _verify_source_binding_contracts()
    print(f"omnivoice_shader_sources=ok modules={checked}")
    return 0


def _verify_shader_module(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if REMOVED_SOURCE_NAME in source:
        raise ValueError(f"{path} still references removed agentorch symbols or paths")
    for removed_name in REMOVED_AUDIO_HEAD_ABI:
        if removed_name in path.name or removed_name in source:
            raise ValueError(f"{path} still references removed audio-head ABI {removed_name}")
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
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_shader_variant_call(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "ShaderVariant"


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _verify_source_binding_contracts() -> None:
    package = importlib.import_module(PACKAGE)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, ShaderVariant):
                _verify_binding_fields(value)
                validate_shader_source_bindings(value)


def _verify_binding_fields(variant: ShaderVariant) -> None:
    fields = set(variant.contract.inputs) | set(variant.contract.outputs)
    for binding in variant.contract.bindings:
        if binding.field not in fields:
            raise ValueError(
                f"{variant.name}.{binding.field} binding has no input/output tensor contract"
            )


if __name__ == "__main__":
    raise SystemExit(main())
