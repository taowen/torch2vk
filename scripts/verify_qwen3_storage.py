#!/usr/bin/env python3
"""Verify Qwen3 safetensor logical tensors can be bound to physical storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from torch2vk.logical import BufferSlice, LogicalTensor
from torch2vk.models.qwen3_safetensor.execution import qwen3_execution_tensors
from torch2vk.models.qwen3_safetensor.schema import qwen3_weight_tensors
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.storage import plan_storage, tensor_nbytes


def main() -> int:
    spec = Qwen3Spec(
        model_type="qwen3",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    execution_tensors = qwen3_execution_tensors(batch=1, steps=3, spec=spec, max_seq_len=8)
    tensors = (*_collect_logical_tensors(execution_tensors), *qwen3_weight_tensors(spec))
    plan = plan_storage(tensors, allocation_id="qwen3-test")
    _validate_plan_covers_tensors(tensors, plan.slices)
    print(f"storage_plan=ok tensors={len(tensors)} unique={len(plan.slices)}")
    return 0


def _collect_logical_tensors(value: object) -> tuple[LogicalTensor, ...]:
    found: list[LogicalTensor] = []
    _collect(value, found)
    return tuple(found)


def _collect(value: object, found: list[LogicalTensor]) -> None:
    if isinstance(value, LogicalTensor):
        found.append(value)
        return
    if isinstance(value, tuple):
        for item in value:
            _collect(item, found)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect(getattr(value, field.name), found)


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
