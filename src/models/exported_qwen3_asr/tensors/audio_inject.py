"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
)
from torch2vk.vulkan.types import TensorSpec


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
    audio_positions: LogicalTensor | None = None,
    audio_features: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    index_copy: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> AudioInjectTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('index_copy',)))
    return AudioInjectTensors(
        audio_positions=_bind_tensor(
            audio_positions,
            _declare_tensor(
            name=f"{prefix}.audio_positions",
            spec=TensorSpec(dtype='int64', shape=(133,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='audio_positions' in request_state_outputs,
            ),
        ),
        audio_features=_bind_tensor(
            audio_features,
            _declare_tensor(
            name=f"{prefix}.audio_features",
            spec=TensorSpec(dtype='float32', shape=(133, 1024)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='audio_features' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
            name=f"{prefix}.unsqueeze",
            spec=TensorSpec(dtype='float32', shape=(1, 133, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        index_copy=_bind_tensor(
            index_copy,
            _declare_tensor(
            name=f"{prefix}.index_copy",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='index_copy' in request_state_outputs,
            ),
        ),
    )


def _declare_tensor(
    *,
    name: str,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    request_state: bool = False,
    compare: ComparePolicy | None = None,
    pytorch_probe: PyTorchProbe | None = None,
) -> LogicalTensor:
    if request_state:
        role = TensorRole.OUTPUT
        memory = MemoryClass.REQUEST_STATE
        lifetime = TensorLifetime.REQUEST
    return LogicalTensor(
        name=name,
        spec=spec,
        role=role,
        memory=memory,
        lifetime=lifetime,
        compare=compare,
        pytorch_probe=pytorch_probe,
    )


def _bind_tensor(
    bound: LogicalTensor | None,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        raise ValueError(f"{bound.name} spec {bound.spec} does not match {tensor.name} spec {tensor.spec}")
    return bound


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
