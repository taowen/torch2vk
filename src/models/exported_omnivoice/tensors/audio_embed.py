"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection, Mapping
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
class AudioEmbedTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    embedding: LogicalTensor


AUDIO_EMBED_OUTPUT: str = 'embedding'


def create_audio_embed(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> AudioEmbedTensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'input', 'embedding')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('embedding',)))
    return AudioEmbedTensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="audio_embeddings.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(8200, 1024)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        input=_bind_tensor(
            bindings,
            'input',
            _declare_tensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='int64', shape=(2, 300)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='input' in request_state_outputs,
            ),
        ),
        embedding=_bind_tensor(
            bindings,
            'embedding',
            _declare_tensor(
            name=f"{prefix}.embedding",
            spec=TensorSpec(dtype='float32', shape=(2, 300, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='embedding' in request_state_outputs,
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
    bindings: Mapping[str, LogicalTensor] | None,
    field: str,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bindings is None:
        return tensor
    bound = bindings.get(field)
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        raise ValueError(f"{field} binding spec {bound.spec} does not match {tensor.spec}")
    return bound


def _validate_bindings(
    bindings: Mapping[str, LogicalTensor] | None,
    tensor_names: frozenset[str],
) -> None:
    if bindings is None:
        return
    unknown = frozenset(bindings) - tensor_names
    if unknown:
        raise ValueError(f"unknown tensor bindings: {sorted(unknown)}")


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
