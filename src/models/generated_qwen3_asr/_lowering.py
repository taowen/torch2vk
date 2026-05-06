"""Shared local lowering helpers for generated Qwen3-ASR frames."""

from __future__ import annotations

from collections.abc import Mapping


def resolve_local_shader(*, target: str, mapping: Mapping[str, str], frame: str) -> str:
    shader = mapping.get(target)
    if shader is None:
        raise NotImplementedError(f"Unsupported generated {frame} op: {target}")
    return shader

