#!/usr/bin/env python3
"""Verify Qwen3 safetensor logical tensors can be bound to physical storage."""

from __future__ import annotations

from collections.abc import Mapping

from torch2vk.logical import BufferSlice, LogicalTensor
from torch2vk.models.qwen3_safetensor.runtime import qwen3_collect_logical_tensors
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.models.qwen3_safetensor.tensors.decode import qwen3_decode_tensors
from torch2vk.models.qwen3_safetensor.tensors.prefill import qwen3_prefill_tensors
from torch2vk.models.qwen3_safetensor.tensors.weights import qwen3_weights
from torch2vk.replay import RecordedSequence, ReplayRegime, storage_fingerprints
from torch2vk.storage import bind_storage, plan_storage, tensor_nbytes


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
    weights = qwen3_weights(spec)
    prefill_tensors = qwen3_prefill_tensors(batch=1, steps=3, spec=spec, max_seq_len=8)
    prefill_plan = _verify_storage_plan(
        (
            *qwen3_collect_logical_tensors(prefill_tensors),
            *qwen3_collect_logical_tensors(weights),
        ),
        phase="prefill",
    )

    decode_tensors = qwen3_decode_tensors(batch=1, spec=spec, max_seq_len=8, step_index=3)
    decode_plan = _verify_storage_plan(
        (
            *qwen3_collect_logical_tensors(decode_tensors),
            *qwen3_collect_logical_tensors(weights),
        ),
        phase="decode",
    )
    print(
        "storage_plan=ok "
        f"prefill_tensors={prefill_plan[0]} prefill_unique={prefill_plan[1]} "
        f"decode_tensors={decode_plan[0]} decode_unique={decode_plan[1]}"
    )
    return 0


def _verify_storage_plan(
    tensors: tuple[LogicalTensor, ...],
    *,
    phase: str,
) -> tuple[int, int]:
    plan = plan_storage(tensors, allocation_id=f"qwen3-test-{phase}")
    _validate_plan_covers_tensors(tensors, plan.slices)
    bound_tensors = bind_storage(tensors, plan)
    fingerprints = storage_fingerprints(bound_tensors)
    RecordedSequence(
        regime=ReplayRegime(model="qwen3_safetensor", phase="prefill", values={}),
        dispatches=(),
        storage=fingerprints,
    ).validate_storage(bound_tensors)
    return len(tensors), len(plan.slices)


def _validate_plan_covers_tensors(
    tensors: tuple[LogicalTensor, ...],
    slices: Mapping[str, BufferSlice],
) -> None:
    by_name: dict[str, int] = {}
    for tensor in tensors:
        by_name[tensor.name] = max(by_name.get(tensor.name, 0), tensor_nbytes(tensor))
    for name, nbytes in by_name.items():
        storage = slices.get(name)
        if storage is None:
            raise ValueError(f"missing storage for {name}")
        if storage.nbytes < nbytes:
            raise ValueError(f"{name} storage too small: {storage.nbytes} < {nbytes}")


if __name__ == "__main__":
    raise SystemExit(main())
