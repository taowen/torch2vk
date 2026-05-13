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
class DecodeEmbedTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    embedding: LogicalTensor


DECODE_EMBED_OUTPUT: str = 'embedding'


def create_decode_embed(
    prefix: str,
    *,
    p_weight: LogicalTensor | None = None,
    input: LogicalTensor | None = None,
    embedding: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> DecodeEmbedTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('embedding',)))
    tensors = DecodeEmbedTensors(
        p_weight=_bind_tensor(
            p_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key="thinker.model.embed_tokens.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(151936, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_weight' in request_state_outputs,
            ),
        ),
        input=_bind_tensor(
            input,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='int64', shape=(1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='input' in request_state_outputs,
            ),
        ),
        embedding=_bind_tensor(
            embedding,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='embedding',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
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
    layout: TensorLayout = CONTIGUOUS_LAYOUT,
    checkpoint: str | None = None,
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
        checkpoint=checkpoint,
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


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
