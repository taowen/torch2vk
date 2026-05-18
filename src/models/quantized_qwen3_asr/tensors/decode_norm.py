"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection
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
class DecodeNormTensors:
    p_weight: LogicalTensor
    hidden_states: LogicalTensor
    rms_norm: LogicalTensor


DECODE_NORM_OUTPUT: str = 'rms_norm'


def create_decode_norm(
    prefix: str,
    *,
    p_weight: LogicalTensor | None = None,
    hidden_states: LogicalTensor | None = None,
    rms_norm: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> DecodeNormTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('rms_norm',)))
    tensors = DecodeNormTensors(
        p_weight=_bind_tensor(
            p_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key="thinker.model.norm.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='hidden_states' in request_state_outputs,
            ),
        ),
        rms_norm=_bind_tensor(
            rms_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm' in request_state_outputs,
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


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
