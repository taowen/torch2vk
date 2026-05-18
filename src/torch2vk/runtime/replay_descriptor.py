"""Replay descriptor tensor helpers."""

from __future__ import annotations

from collections.abc import Mapping

from torch2vk.runtime.logical import LogicalTensor, MemoryClass
from torch2vk.vulkan.types import tensor_nbytes


def canonical_replay_descriptor_tensor(
    *,
    tensor: LogicalTensor,
    logical_tensors: Mapping[str, LogicalTensor],
) -> LogicalTensor:
    if tensor.memory is not MemoryClass.FRAME_WORKSPACE:
        return tensor
    alias_source = tensor.alias_source
    if alias_source is not None:
        alias_nbytes = tensor.alias_nbytes or tensor_nbytes(tensor.spec)
        if tensor.alias_byte_offset == 0 and alias_nbytes == tensor_nbytes(alias_source.spec):
            return canonical_replay_descriptor_tensor(
                tensor=alias_source,
                logical_tensors=logical_tensors,
            )
        return tensor
    alias_owner = _live_non_frame_alias_owner(tensor, logical_tensors)
    if alias_owner is not None:
        return alias_owner
    return tensor


def replay_descriptor_rebindable(tensor: LogicalTensor) -> bool:
    return tensor.memory not in {
        MemoryClass.FRAME_WORKSPACE,
        MemoryClass.MODEL_WEIGHT,
        MemoryClass.SESSION_TENSOR,
        MemoryClass.OP_SCRATCH,
    }


def has_live_buffer(tensor: LogicalTensor) -> bool:
    return tensor.buffer is not None and not tensor.buffer.allocation.released


def _live_non_frame_alias_owner(
    tensor: LogicalTensor,
    logical_tensors: Mapping[str, LogicalTensor],
) -> LogicalTensor | None:
    if not has_live_buffer(tensor):
        return None
    for candidate in logical_tensors.values():
        if candidate is tensor or candidate.memory is MemoryClass.FRAME_WORKSPACE:
            continue
        if has_live_buffer(candidate) and candidate.buffer == tensor.buffer:
            return candidate
    return None
