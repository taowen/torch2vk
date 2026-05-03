#!/usr/bin/env python3
"""Verify Qwen3 debug boundaries produce useful mismatch drilldown."""

from __future__ import annotations

from torch2vk.models.qwen3_safetensor.debug import (
    qwen3_prefill_debug_boundaries,
    qwen3_prefill_initial_tensors,
)
from torch2vk.models.qwen3_safetensor.execution import (
    qwen3_execution_tensors,
    record_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.shader import DispatchTarget
from torch2vk.validation import (
    compare_declared_boundaries,
    debug_readback_plan,
    validate_dispatch_read_write_chain,
)


def main() -> int:
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
    tensors = qwen3_execution_tensors(batch=1, steps=2, spec=spec)
    target = DispatchTarget()
    record_qwen3_prefill(target, spec=spec, tensors=tensors)
    initial = qwen3_prefill_initial_tensors(spec=spec, tensors=tensors)
    validate_dispatch_read_write_chain(target.records, initial_tensors=initial).raise_for_issues()

    boundaries = qwen3_prefill_debug_boundaries(tensors)
    plan = debug_readback_plan(boundaries, dispatch_records=target.records)
    if tensors.hidden.name not in plan:
        raise AssertionError("debug readback plan must include the first boundary tensor")
    if tensors.input_ids.name not in plan:
        raise AssertionError("debug readback plan must include writer inputs")
    if "weights.embed_tokens" not in plan:
        raise AssertionError("debug readback plan must include writer weights")

    reference = {name: 0 for name in plan.tensor_names}
    candidate = dict(reference)
    candidate[tensors.hidden.name] = 1
    report = compare_declared_boundaries(
        boundaries,
        dispatch_records=target.records,
        reference=reference,
        candidate=candidate,
    )
    if report.mismatch is None:
        raise AssertionError("debug compare must report the injected mismatch")
    if report.mismatch.tensor != tensors.hidden.name:
        raise AssertionError(f"unexpected mismatch tensor {report.mismatch.tensor}")
    if report.mismatch.writer is None:
        raise AssertionError("debug mismatch must include writer drilldown")
    if report.mismatch.writer.shader != "embedding_lookup_bf16_f32_sequence":
        raise AssertionError(f"unexpected writer shader {report.mismatch.writer.shader}")
    if report.mismatch.writer.divergent_inputs:
        raise AssertionError("writer inputs should match for the injected output mismatch")
    if report.mismatch.writer.divergent_output is None:
        raise AssertionError("writer drilldown must identify the divergent output")

    print(
        "qwen3_debug=ok "
        f"boundaries={len(boundaries)} "
        f"readback_tensors={len(plan.tensor_names)} "
        f"first_writer={report.mismatch.writer.shader}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
