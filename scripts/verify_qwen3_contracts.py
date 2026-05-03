#!/usr/bin/env python3
"""Verify Qwen3 safetensor shader contracts against source and execution records."""

from __future__ import annotations

import importlib
import pkgutil

from torch2vk.models.qwen3_safetensor.debug import (
    qwen3_decode_initial_tensors,
    qwen3_prefill_initial_tensors,
)
from torch2vk.models.qwen3_safetensor.execution import (
    run_qwen3_decode_step,
    run_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.models.qwen3_safetensor.tensors.decode import qwen3_decode_tensors
from torch2vk.models.qwen3_safetensor.tensors.prefill import qwen3_prefill_tensors
from torch2vk.models.qwen3_safetensor.tensors.weights import qwen3_weights
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
    weights = qwen3_weights(spec)

    prefill_tensors = qwen3_prefill_tensors(batch=1, steps=3, spec=spec, max_seq_len=8)
    prefill_target = DispatchTarget()
    run_qwen3_prefill(
        prefill_target,
        spec=spec,
        tensors=prefill_tensors,
        weights=weights,
    )
    validate_dispatch_read_write_chain(
        prefill_target.records,
        initial_tensors=qwen3_prefill_initial_tensors(
            tensors=prefill_tensors,
            weights=weights,
        ),
    ).raise_for_issues()

    decode_tensors = qwen3_decode_tensors(batch=1, spec=spec, max_seq_len=8, step_index=3)
    decode_target = DispatchTarget()
    run_qwen3_decode_step(
        decode_target,
        spec=spec,
        tensors=decode_tensors,
        weights=weights,
    )
    validate_dispatch_read_write_chain(
        decode_target.records,
        initial_tensors=qwen3_decode_initial_tensors(tensors=decode_tensors, weights=weights),
    ).raise_for_issues()
    print(
        "dispatch_read_write=ok "
        f"prefill_dispatches={len(prefill_target.records)} "
        f"decode_dispatches={len(decode_target.records)}"
    )
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
