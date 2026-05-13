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
    q4_k_words_layout,
)


@dataclass(frozen=True, slots=True)
class EmbedTokensTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    embedding: LogicalTensor


EMBED_TOKENS_OUTPUT: str = 'embedding'


def create_embed_tokens(
    prefix: str,
    *,
    sequence_length: int,
    p_weight: LogicalTensor | None = None,
    input: LogicalTensor | None = None,
    embedding: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> EmbedTokensTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('embedding',)))
    tensors = EmbedTokensTensors(
        p_weight=_bind_tensor(
            p_weight,
            _declare_tensor(
                checkpoint_key="model.embed_tokens.weight",
                reference_key=None,
                spec=TensorSpec(dtype='uint32', shape=(151936, 144)),
                layout=q4_k_words_layout(logical_k=1024),
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
                reference_key=None,
                spec=TensorSpec(dtype='int64', shape=(1, sequence_length)),
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
                checkpoint_key=None,
                reference_key='embedding',
                spec=TensorSpec(dtype='float32', shape=(1, sequence_length, 1024)),
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
