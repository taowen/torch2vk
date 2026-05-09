"""Logical tensors for one Qwen3-ASR audio encoder layer."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioEncoderLayerTensors:
    self_attn_layer_norm_weight: LogicalTensor
    self_attn_layer_norm_bias: LogicalTensor
    q_proj_weight: LogicalTensor
    q_proj_bias: LogicalTensor
    k_proj_weight: LogicalTensor
    k_proj_bias: LogicalTensor
    v_proj_weight: LogicalTensor
    v_proj_bias: LogicalTensor
    out_proj_weight: LogicalTensor
    out_proj_bias: LogicalTensor
    final_layer_norm_weight: LogicalTensor
    final_layer_norm_bias: LogicalTensor
    fc1_weight: LogicalTensor
    fc1_bias: LogicalTensor
    fc2_weight: LogicalTensor
    fc2_bias: LogicalTensor
    self_attn_layer_norm: LogicalTensor
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    self_attn: LogicalTensor
    out_proj: LogicalTensor
    self_attn_residual: LogicalTensor
    final_layer_norm: LogicalTensor
    fc1_gelu: LogicalTensor
    fc2: LogicalTensor
    output: LogicalTensor


def declare_qwen3_asr_audio_encoder_layer_tensors(
    *,
    layer: int,
    hidden_shape: tuple[int, int],
    hidden_size: int,
    encoder_ffn_dim: int,
) -> Qwen3AsrAudioEncoderLayerTensors:
    weight_prefix = f"thinker.audio_tower.layers.{layer}"
    def weight(checkpoint_key: str, shape: tuple[int, ...]) -> LogicalTensor:
        return LogicalTensor(
            spec=TensorSpec(dtype="bfloat16", shape=shape),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            checkpoint_key=checkpoint_key,
        )

    def activation(shape: tuple[int, ...]) -> LogicalTensor:
        return LogicalTensor(
            spec=TensorSpec(dtype="float32", shape=shape),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
        )

    def comparable_activation(
        shape: tuple[int, ...],
        *,
        rtol: float = 2e-3,
        atol: float = 2e-2,
    ) -> LogicalTensor:
        return LogicalTensor(
            spec=TensorSpec(dtype="float32", shape=shape),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
        )

    return Qwen3AsrAudioEncoderLayerTensors(
        self_attn_layer_norm_weight=weight(
            f"{weight_prefix}.self_attn_layer_norm.weight", (hidden_size,)
        ),
        self_attn_layer_norm_bias=weight(
            f"{weight_prefix}.self_attn_layer_norm.bias", (hidden_size,)
        ),
        q_proj_weight=weight(
            f"{weight_prefix}.self_attn.q_proj.weight", (hidden_size, hidden_size)
        ),
        q_proj_bias=weight(f"{weight_prefix}.self_attn.q_proj.bias", (hidden_size,)),
        k_proj_weight=weight(
            f"{weight_prefix}.self_attn.k_proj.weight", (hidden_size, hidden_size)
        ),
        k_proj_bias=weight(f"{weight_prefix}.self_attn.k_proj.bias", (hidden_size,)),
        v_proj_weight=weight(
            f"{weight_prefix}.self_attn.v_proj.weight", (hidden_size, hidden_size)
        ),
        v_proj_bias=weight(f"{weight_prefix}.self_attn.v_proj.bias", (hidden_size,)),
        out_proj_weight=weight(
            f"{weight_prefix}.self_attn.out_proj.weight", (hidden_size, hidden_size)
        ),
        out_proj_bias=weight(f"{weight_prefix}.self_attn.out_proj.bias", (hidden_size,)),
        final_layer_norm_weight=weight(f"{weight_prefix}.final_layer_norm.weight", (hidden_size,)),
        final_layer_norm_bias=weight(f"{weight_prefix}.final_layer_norm.bias", (hidden_size,)),
        fc1_weight=weight(f"{weight_prefix}.fc1.weight", (encoder_ffn_dim, hidden_size)),
        fc1_bias=weight(f"{weight_prefix}.fc1.bias", (encoder_ffn_dim,)),
        fc2_weight=weight(f"{weight_prefix}.fc2.weight", (hidden_size, encoder_ffn_dim)),
        fc2_bias=weight(f"{weight_prefix}.fc2.bias", (hidden_size,)),
        self_attn_layer_norm=comparable_activation(hidden_shape),
        q_proj=comparable_activation(hidden_shape),
        k_proj=comparable_activation(hidden_shape),
        v_proj=comparable_activation(hidden_shape),
        self_attn=comparable_activation(hidden_shape),
        out_proj=comparable_activation(hidden_shape),
        self_attn_residual=comparable_activation(hidden_shape),
        final_layer_norm=comparable_activation(hidden_shape),
        fc1_gelu=activation((hidden_shape[0], encoder_ffn_dim)),
        fc2=comparable_activation(hidden_shape),
        output=comparable_activation(hidden_shape),
    )
