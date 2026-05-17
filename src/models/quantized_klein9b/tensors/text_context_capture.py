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
class TextContextCaptureTensors:
    layer_9: LogicalTensor
    layer_18: LogicalTensor
    layer_27: LogicalTensor
    to: LogicalTensor
    to_1: LogicalTensor
    to_2: LogicalTensor
    cat: LogicalTensor


TEXT_CONTEXT_CAPTURE_OUTPUT: str = 'cat'


def create_text_context_capture(
    prefix: str,
    *,
    sequence_length: int,
    layer_9: LogicalTensor | None = None,
    layer_18: LogicalTensor | None = None,
    layer_27: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    to_1: LogicalTensor | None = None,
    to_2: LogicalTensor | None = None,
    cat: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> TextContextCaptureTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('cat',)))
    tensors = TextContextCaptureTensors(
        layer_9=_bind_tensor(
            layer_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=None,
                spec=TensorSpec(dtype='float16', shape=(1, sequence_length, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_9' in request_state_outputs,
            ),
        ),
        layer_18=_bind_tensor(
            layer_18,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=None,
                spec=TensorSpec(dtype='float16', shape=(1, sequence_length, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_18' in request_state_outputs,
            ),
        ),
        layer_27=_bind_tensor(
            layer_27,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=None,
                spec=TensorSpec(dtype='float16', shape=(1, sequence_length, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_27' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, sequence_length, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to' in request_state_outputs,
            ),
        ),
        to_1=_bind_tensor(
            to_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_1',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, sequence_length, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_1' in request_state_outputs,
            ),
        ),
        to_2=_bind_tensor(
            to_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_2',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, sequence_length, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_2' in request_state_outputs,
            ),
        ),
        cat=_bind_tensor(
            cat,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, sequence_length, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat' in request_state_outputs,
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
