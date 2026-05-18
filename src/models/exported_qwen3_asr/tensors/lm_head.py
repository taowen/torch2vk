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
    CONTIGUOUS_LAYOUT,
    TensorLayout,
    TensorSpec,
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
            _declare_tensor(
                checkpoint=None,
                checkpoint_key='thinker.lm_head.weight',
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='bfloat16', shape=(151936, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state=False,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    return tensors


def _declare_tensor(
    *,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    layout: TensorLayout = CONTIGUOUS_LAYOUT,
    checkpoint: str | None = None,
    checkpoint_key: str | None = None,
    reference_key: str | None = None,
    layer: str | None = None,
    request_state: bool = False,
) -> LogicalTensor:
    if request_state:
        role = TensorRole.OUTPUT
        memory = MemoryClass.REQUEST_STATE
        lifetime = TensorLifetime.REQUEST
    return LogicalTensor(
        spec=spec,
        role=role,
        memory=memory,
        lifetime=lifetime,
        checkpoint=checkpoint,
        checkpoint_key=checkpoint_key,
        reference_key=reference_key,
        layer=layer,
        layout=layout,
    )


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
