#!/usr/bin/env python3
"""Verify Qwen3 safetensor shader contracts against source and execution records."""

from __future__ import annotations

import importlib
import pkgutil

from torch2vk.models.qwen3_safetensor.debug import qwen3_prefill_initial_tensors
from torch2vk.models.qwen3_safetensor.execution import (
    qwen3_execution_tensors,
    record_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.shader import DispatchTarget, ShaderVariant, validate_shader_source_bindings
from torch2vk.validation import validate_dispatch_read_write_chain


PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"


def main() -> int:
    variants = _shader_variants()
    for variant in variants:
        validate_shader_source_bindings(variant)
    print(f"shader_source_bindings=ok variants={len(variants)}")

    spec = Qwen3Spec(
        model_type="qwen3",
        vocab_size=16,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    tensors = qwen3_execution_tensors(batch=1, steps=3, spec=spec, max_seq_len=8)
    target = DispatchTarget()
    record_qwen3_prefill(target, spec=spec, tensors=tensors)
    report = validate_dispatch_read_write_chain(
        target.records,
        initial_tensors=qwen3_prefill_initial_tensors(spec=spec, tensors=tensors),
    )
    report.raise_for_issues()
    print(f"dispatch_read_write=ok dispatches={len(target.records)}")
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


if __name__ == "__main__":
    raise SystemExit(main())
