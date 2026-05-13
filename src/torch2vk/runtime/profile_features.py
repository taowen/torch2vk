"""Small helpers for profiler rows and shader-source feature summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ShaderFeatures = dict[str, bool]


def classify_op_group(*, shader: str, frame: str = "", output_op: str = "") -> str:
    haystack = f"{shader} {frame} {output_op}".lower()
    if "q4_k" in haystack or "q4" in haystack:
        return "q4_linear"
    if "q6_k" in haystack or "q6" in haystack:
        return "q6_linear"
    if "sdpa" in haystack or "attention" in haystack or "attn" in haystack:
        return "attention"
    if "cache_write" in haystack or "token_major" in haystack or "kv_cache_write" in haystack:
        return "cache_write"
    if "norm" in haystack:
        return "norm"
    if "embedding" in haystack or "embed" in haystack:
        return "embedding"
    if "token" in haystack or "argmax" in haystack:
        return "token"
    return "other"


def empty_shader_features() -> ShaderFeatures:
    return {
        "coop_matrix": False,
        "integer_dot": False,
        "shared_memory": False,
        "subgroup": False,
        "f16": False,
        "f32": False,
    }


def scan_shader_features(source_path: str | Path | None) -> ShaderFeatures:
    features = empty_shader_features()
    if source_path is None:
        return features
    text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
    features["coop_matrix"] = (
        "GL_KHR_cooperative_matrix" in text or "coopMatMulAdd" in text
    )
    features["integer_dot"] = (
        "GL_EXT_integer_dot_product" in text or "dotPacked4x8EXT" in text
    )
    features["shared_memory"] = "shared " in text
    features["subgroup"] = "subgroup" in text
    features["f16"] = "float16_t" in text or "f16" in text
    features["f32"] = "float" in text
    return features


def build_shape_hint(row: dict[str, Any]) -> dict[str, Any]:
    tensors = row.get("tensors")
    tensor_shapes: dict[str, list[int]] = {}
    if isinstance(tensors, list):
        for item in tensors:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            shape = item.get("shape")
            if isinstance(field, str) and isinstance(shape, list):
                tensor_shapes[field] = [int(value) for value in shape]
    return {
        "dispatch_size": row.get("dispatch_size", []),
        "symbols": row.get("symbols", {}),
        "push_constants": row.get("push_constants", {}),
        "tensor_shapes": tensor_shapes,
    }
