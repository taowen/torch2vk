"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    bind_logical_tensor_alias,
    bind_logical_tensor_names,
)
from torch2vk.vulkan.types import (
    CONTIGUOUS_LAYOUT,
    TensorLayout,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class EulerUpdateTensors:
    x: LogicalTensor
    pred: LogicalTensor
    dt: LogicalTensor
    view: LogicalTensor
    mul: LogicalTensor
    add: LogicalTensor


EULER_UPDATE_OUTPUT: str = 'add'


def create_euler_update(
    prefix: str,
    *,
    image_seq_len: int,
    x: LogicalTensor | None = None,
    pred: LogicalTensor | None = None,
    dt: LogicalTensor | None = None,
    view: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> EulerUpdateTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add',)))
    tensors = EulerUpdateTensors(
        x=_bind_tensor(
            x,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='x' in request_state_outputs,
            ),
        ),
        pred=_bind_tensor(
            pred,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='pred' in request_state_outputs,
            ),
        ),
        dt=_bind_tensor(
            dt,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='dt' in request_state_outputs,
            ),
        ),
        view=_bind_tensor(
            view,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.dt, tensors.view)
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


def _bind_alias_source(
    src: LogicalTensor,
    dst: LogicalTensor,
    *,
    byte_offset: int = 0,
    nbytes: int | None = None,
) -> None:
    bind_logical_tensor_alias(src, dst, byte_offset=byte_offset, nbytes=nbytes)


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
