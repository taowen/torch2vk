"""Generated tensor declarations."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    bind_logical_tensor_names,
)
from torch2vk.vulkan.types import (
    TensorSpec,
    q6_k_halfwords_layout,
)


@dataclass(frozen=True, slots=True)
class LmHeadTensors:
    p_weight: LogicalTensor


def create_lm_head(
    prefix: str,
    *,
    p_weight: LogicalTensor | None = None,
) -> LmHeadTensors:
    tensors = LmHeadTensors(
        p_weight=_bind_tensor(
            p_weight,
            LogicalTensor(
                spec=TensorSpec(dtype="uint16", shape=(151936, 420)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                checkpoint_key="lm_head.weight",
                layout=q6_k_halfwords_layout(logical_k=1024),
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    return tensors


def _bind_tensor(
    bound: LogicalTensor | None,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        bound_name = bound.name or "<bound>"
        tensor_name = tensor.name or "<declared>"
        raise ValueError(f"{bound_name} spec {bound.spec} does not match {tensor_name} spec {tensor.spec}")
    return bound
