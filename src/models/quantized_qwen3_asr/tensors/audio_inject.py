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
class AudioInjectTensors:
    audio_positions: LogicalTensor
    audio_features: LogicalTensor
    unsqueeze: LogicalTensor
    index_copy: LogicalTensor


AUDIO_INJECT_OUTPUT: str = 'index_copy'


def create_audio_inject(
    prefix: str,
    *,
    audio_sequence_length: int,
    sequence_length: int,
    audio_positions: LogicalTensor | None = None,
    audio_features: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    index_copy: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> AudioInjectTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('index_copy',)))
    tensors = AudioInjectTensors(
        audio_positions=_bind_tensor(
            audio_positions,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='int64', shape=(audio_sequence_length,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='audio_positions' in request_state_outputs,
            ),
        ),
        audio_features=_bind_tensor(
            audio_features,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float32', shape=(audio_sequence_length, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='audio_features' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='unsqueeze',
                spec=TensorSpec(dtype='float32', shape=(1, audio_sequence_length, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        index_copy=_bind_tensor(
            index_copy,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='index_copy',
                spec=TensorSpec(dtype='float32', shape=(1, sequence_length, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='index_copy' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.audio_features, tensors.unsqueeze)
    return tensors


def _declare_tensor(
    *,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    layout: TensorLayout = CONTIGUOUS_LAYOUT,
    checkpoint_key: str | None = None,
    reference_key: str | None = None,
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
        checkpoint_key=checkpoint_key,
        reference_key=reference_key,
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


def _bind_alias_source(src: LogicalTensor, dst: LogicalTensor) -> None:
    bind_logical_tensor_alias(src, dst)


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
