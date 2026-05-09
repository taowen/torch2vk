"""Logical tensors for one Qwen3-ASR text decoder layer."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrTextLayerTensors:
    input_layernorm_weight: LogicalTensor
    post_attention_layernorm_weight: LogicalTensor
    q_norm_weight: LogicalTensor
    k_norm_weight: LogicalTensor
    q_proj_weight: LogicalTensor
    k_proj_weight: LogicalTensor
    v_proj_weight: LogicalTensor
    o_proj_weight: LogicalTensor
    gate_proj_weight: LogicalTensor
    up_proj_weight: LogicalTensor
    down_proj_weight: LogicalTensor
    input_layernorm: LogicalTensor
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    q_normed: LogicalTensor
    k_normed: LogicalTensor
    q_roped: LogicalTensor
    k_roped: LogicalTensor
    key_cache: LogicalTensor
    value_cache: LogicalTensor
    attention: LogicalTensor
    o_proj: LogicalTensor
    attn_residual: LogicalTensor
    post_attention_layernorm: LogicalTensor
    gate_proj: LogicalTensor
    up_proj: LogicalTensor
    swiglu: LogicalTensor
    down_proj: LogicalTensor
    output: LogicalTensor


def declare_qwen3_asr_text_layer_tensors(
    *,
    frame_prefix: str,
    layer: int,
    sequence_length: int,
    max_sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    key_cache: LogicalTensor | None = None,
    value_cache: LogicalTensor | None = None,
    weights_from: Qwen3AsrTextLayerTensors | None = None,
) -> Qwen3AsrTextLayerTensors:
    del max_sequence_length
    del frame_prefix
    hidden_shape = (1, sequence_length, hidden_size)
    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    q_shape = (1, sequence_length, q_width)
    kv_shape = (1, sequence_length, kv_width)
    cache_shape = (1, num_key_value_heads, sequence_length, head_dim)
    if weights_from is not None:
        input_layernorm_weight = weights_from.input_layernorm_weight
        post_attention_layernorm_weight = weights_from.post_attention_layernorm_weight
        q_norm_weight = weights_from.q_norm_weight
        k_norm_weight = weights_from.k_norm_weight
        q_proj_weight = weights_from.q_proj_weight
        k_proj_weight = weights_from.k_proj_weight
        v_proj_weight = weights_from.v_proj_weight
        o_proj_weight = weights_from.o_proj_weight
        gate_proj_weight = weights_from.gate_proj_weight
        up_proj_weight = weights_from.up_proj_weight
        down_proj_weight = weights_from.down_proj_weight
    else:
        weight_prefix = f"thinker.model.layers.{layer}"
        input_layernorm_weight = _weight(f"{weight_prefix}.input_layernorm.weight", (hidden_size,))
        post_attention_layernorm_weight = _weight(
            f"{weight_prefix}.post_attention_layernorm.weight", (hidden_size,)
        )
        q_norm_weight = _weight(f"{weight_prefix}.self_attn.q_norm.weight", (head_dim,))
        k_norm_weight = _weight(f"{weight_prefix}.self_attn.k_norm.weight", (head_dim,))
        q_proj_weight = _weight(
            f"{weight_prefix}.self_attn.q_proj.weight", (q_width, hidden_size)
        )
        k_proj_weight = _weight(
            f"{weight_prefix}.self_attn.k_proj.weight", (kv_width, hidden_size),
        )
        v_proj_weight = _weight(
            f"{weight_prefix}.self_attn.v_proj.weight", (kv_width, hidden_size),
        )
        o_proj_weight = _weight(
            f"{weight_prefix}.self_attn.o_proj.weight", (hidden_size, q_width)
        )
        gate_proj_weight = _weight(
            f"{weight_prefix}.mlp.gate_proj.weight", (intermediate_size, hidden_size)
        )
        up_proj_weight = _weight(
            f"{weight_prefix}.mlp.up_proj.weight", (intermediate_size, hidden_size)
        )
        down_proj_weight = _weight(
            f"{weight_prefix}.mlp.down_proj.weight", (hidden_size, intermediate_size)
        )
    return Qwen3AsrTextLayerTensors(
        input_layernorm_weight=input_layernorm_weight,
        post_attention_layernorm_weight=post_attention_layernorm_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        gate_proj_weight=gate_proj_weight,
        up_proj_weight=up_proj_weight,
        down_proj_weight=down_proj_weight,
        input_layernorm=_probed(
            hidden_shape,
            target=f"model.layers.{layer}.input_layernorm",
        ),
        q_proj=_probed(
            q_shape,
            target=f"model.layers.{layer}.self_attn.q_proj",
        ),
        k_proj=_probed(
            kv_shape,
            target=f"model.layers.{layer}.self_attn.k_proj",
        ),
        v_proj=_probed(
            kv_shape,
            target=f"model.layers.{layer}.self_attn.v_proj",
        ),
        q_normed=_activation(q_shape),
        k_normed=_activation(kv_shape),
        q_roped=_activation(q_shape),
        k_roped=_activation(kv_shape),
        key_cache=key_cache or _state(
            cache_shape,
            semantic=TensorSemantic.KV_CACHE,
        ),
        value_cache=value_cache or _state(
            cache_shape,
            semantic=TensorSemantic.KV_CACHE,
        ),
        attention=LogicalTensor(
            spec=TensorSpec(dtype="float32", shape=(1, sequence_length, q_width)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(
                kind="module_input",
                target=f"model.layers.{layer}.self_attn.o_proj",
                index=0,
            ),
        ),
        o_proj=_probed(
            hidden_shape,
            target=f"model.layers.{layer}.self_attn.o_proj",
        ),
        attn_residual=_activation(hidden_shape),
        post_attention_layernorm=_probed(
            hidden_shape,
            target=f"model.layers.{layer}.post_attention_layernorm",
        ),
        gate_proj=_probed(
            (1, sequence_length, intermediate_size),
            target=f"model.layers.{layer}.mlp.gate_proj",
        ),
        up_proj=_probed(
            (1, sequence_length, intermediate_size),
            target=f"model.layers.{layer}.mlp.up_proj",
        ),
        swiglu=_activation((1, sequence_length, intermediate_size)),
        down_proj=_probed(
            hidden_shape,
            target=f"model.layers.{layer}.mlp.down_proj",
        ),
        output=LogicalTensor(
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target=f"model.layers.{layer}"),
        ),
    )


def _weight(name: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype="bfloat16", shape=shape),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
        checkpoint_key=name,
    )


def _probed(shape: tuple[int, ...], *, target: str) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype="float32", shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
        compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
        pytorch_probe=PyTorchProbe(kind="module_output", target=target),
    )


def _activation(shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype="float32", shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
    )


def _state(
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic,
) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype="float32", shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=semantic,
    )
