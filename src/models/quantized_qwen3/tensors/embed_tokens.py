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
    q6_k_halfwords_layout,
    q8_0_halfwords_layout,
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
                checkpoint=None,
                checkpoint_key="model.embed_tokens.weight",
                reference_key=None,
                spec=_quantized_weight_spec("model.embed_tokens.weight", dtype='float32', shape=(151936, 1024)),
                layout=_quantized_weight_layout("model.embed_tokens.weight", dtype='float32', shape=(151936, 1024)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='embedding',
                spec=TensorSpec(dtype='float16', shape=(1, sequence_length, 1024)),
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


_Q6_TENSOR_NAMES = frozenset(('lm_head.weight', 'model.layers.0.mlp.down_proj.weight', 'model.layers.0.self_attn.v_proj.weight', 'model.layers.1.mlp.down_proj.weight', 'model.layers.1.self_attn.v_proj.weight', 'model.layers.11.mlp.down_proj.weight', 'model.layers.11.self_attn.v_proj.weight', 'model.layers.14.mlp.down_proj.weight', 'model.layers.14.self_attn.v_proj.weight', 'model.layers.17.mlp.down_proj.weight', 'model.layers.17.self_attn.v_proj.weight', 'model.layers.2.mlp.down_proj.weight', 'model.layers.2.self_attn.v_proj.weight', 'model.layers.20.mlp.down_proj.weight', 'model.layers.20.self_attn.v_proj.weight', 'model.layers.23.mlp.down_proj.weight', 'model.layers.23.self_attn.v_proj.weight', 'model.layers.24.mlp.down_proj.weight', 'model.layers.24.self_attn.v_proj.weight', 'model.layers.25.mlp.down_proj.weight', 'model.layers.25.self_attn.v_proj.weight', 'model.layers.26.mlp.down_proj.weight', 'model.layers.26.self_attn.v_proj.weight', 'model.layers.27.mlp.down_proj.weight', 'model.layers.27.self_attn.v_proj.weight', 'model.layers.5.mlp.down_proj.weight', 'model.layers.5.self_attn.v_proj.weight', 'model.layers.8.mlp.down_proj.weight', 'model.layers.8.self_attn.v_proj.weight'))
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(())
_Q8_TENSOR_PREFIXES = ()


def _quantized_weight_spec(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorSpec:
    if dtype not in ("float32", "float16", "bfloat16"):
        return TensorSpec(dtype=dtype, shape=shape)
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    if force_q6 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return TensorSpec(dtype="uint16", shape=(n, k // 256 * 105))
    if force_q8 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        padded_k = _round_up(k, 32)
        return TensorSpec(dtype="uint16", shape=(n, padded_k // 32 * 17))
    if len(shape) != 2:
        return TensorSpec(dtype=dtype, shape=shape)
    n, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return TensorSpec(dtype="float32", shape=shape)
        return TensorSpec(dtype="uint16", shape=(n, k // 32 * 17))
    return TensorSpec(dtype="uint32", shape=(n, k // 256 * 36))


def _quantized_weight_layout(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorLayout:
    if dtype not in ("float32", "float16", "bfloat16"):
        return CONTIGUOUS_LAYOUT
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    if force_q6 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return q6_k_halfwords_layout(logical_k=k)
    if force_q8 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        return q8_0_halfwords_layout(logical_k=k)
    if len(shape) != 2:
        return CONTIGUOUS_LAYOUT
    _, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return CONTIGUOUS_LAYOUT
        return q8_0_halfwords_layout(logical_k=k)
    return q4_k_words_layout(logical_k=k)


def _quantized_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    rows = shape[0]
    cols = 1
    for dim in shape[1:]:
        cols *= dim
    return rows, cols


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


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


def _bind_alias_source(src: LogicalTensor, dst: LogicalTensor) -> None:
    bind_logical_tensor_alias(src, dst)


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
