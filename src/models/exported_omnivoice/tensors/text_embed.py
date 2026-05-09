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
    bind_logical_tensor_names,
)
from torch2vk.vulkan.types import TensorSpec


@dataclass(frozen=True, slots=True)
class TextEmbedTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    embedding: LogicalTensor


TEXT_EMBED_OUTPUT: str = 'embedding'


def create_text_embed(
    prefix: str,
    *,
    p_weight: LogicalTensor | None = None,
    input: LogicalTensor | None = None,
    embedding: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> TextEmbedTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('embedding',)))
    tensors = TextEmbedTensors(
        p_weight=_bind_tensor(
            p_weight,
            _declare_tensor(
                checkpoint_key="llm.embed_tokens.weight",
                spec=TensorSpec(dtype='bfloat16', shape=(151676, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_weight' in request_state_outputs,
            ),
        ),
        input=_bind_tensor(
            input,
            _declare_tensor(
                checkpoint_key=None,
                spec=TensorSpec(dtype='int64', shape=(2, 300)),
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='input' in request_state_outputs,
            ),
        ),
        embedding=_bind_tensor(
            embedding,
            _declare_tensor(
                checkpoint_key=None,
                spec=TensorSpec(dtype='float32', shape=(2, 300, 1024)),
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='embedding' in request_state_outputs,
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
    checkpoint_key: str | None = None,
    request_state: bool = False,
    compare: ComparePolicy | None = None,
    pytorch_probe: PyTorchProbe | None = None,
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
        compare=compare,
        pytorch_probe=pytorch_probe,
        checkpoint_key=checkpoint_key,
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
