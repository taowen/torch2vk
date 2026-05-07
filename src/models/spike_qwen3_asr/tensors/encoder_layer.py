"""Generated tensor declarations."""

from __future__ import annotations

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
class EncoderLayerTensors:
    p_attn_k_proj_weight: LogicalTensor
    p_attn_k_proj_bias: LogicalTensor
    p_attn_v_proj_weight: LogicalTensor
    p_attn_v_proj_bias: LogicalTensor
    p_attn_q_proj_weight: LogicalTensor
    p_attn_q_proj_bias: LogicalTensor
    p_attn_out_proj_weight: LogicalTensor
    p_attn_out_proj_bias: LogicalTensor
    p_attn_layer_norm_weight: LogicalTensor
    p_attn_layer_norm_bias: LogicalTensor
    p_fc1_weight: LogicalTensor
    p_fc1_bias: LogicalTensor
    p_fc2_weight: LogicalTensor
    p_fc2_bias: LogicalTensor
    p_final_layer_norm_weight: LogicalTensor
    p_final_layer_norm_bias: LogicalTensor
    hidden_states: LogicalTensor
    cu_seqlens: LogicalTensor
    attention_mask: LogicalTensor
    layer_norm: LogicalTensor
    linear: LogicalTensor
    reshape: LogicalTensor
    linear_1: LogicalTensor
    reshape_1: LogicalTensor
    linear_2: LogicalTensor
    reshape_2: LogicalTensor
    transpose: LogicalTensor
    unsqueeze: LogicalTensor
    transpose_1: LogicalTensor
    unsqueeze_1: LogicalTensor
    transpose_2: LogicalTensor
    unsqueeze_2: LogicalTensor
    slice_1: LogicalTensor
    slice_2: LogicalTensor
    sub: LogicalTensor
    max_1: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    transpose_3: LogicalTensor
    contiguous: LogicalTensor
    reshape_3: LogicalTensor
    linear_3: LogicalTensor
    add: LogicalTensor
    layer_norm_1: LogicalTensor
    linear_4: LogicalTensor
    gelu: LogicalTensor
    linear_5: LogicalTensor
    add_1: LogicalTensor


ENCODER_LAYER_WEIGHT_MAP: dict[str, str] = {
    'p_attn_k_proj_weight': 'thinker.audio_tower.layers.{i}.self_attn.k_proj.weight',
    'p_attn_k_proj_bias': 'thinker.audio_tower.layers.{i}.self_attn.k_proj.bias',
    'p_attn_v_proj_weight': 'thinker.audio_tower.layers.{i}.self_attn.v_proj.weight',
    'p_attn_v_proj_bias': 'thinker.audio_tower.layers.{i}.self_attn.v_proj.bias',
    'p_attn_q_proj_weight': 'thinker.audio_tower.layers.{i}.self_attn.q_proj.weight',
    'p_attn_q_proj_bias': 'thinker.audio_tower.layers.{i}.self_attn.q_proj.bias',
    'p_attn_out_proj_weight': 'thinker.audio_tower.layers.{i}.self_attn.out_proj.weight',
    'p_attn_out_proj_bias': 'thinker.audio_tower.layers.{i}.self_attn.out_proj.bias',
    'p_attn_layer_norm_weight': 'thinker.audio_tower.layers.{i}.self_attn_layer_norm.weight',
    'p_attn_layer_norm_bias': 'thinker.audio_tower.layers.{i}.self_attn_layer_norm.bias',
    'p_fc1_weight': 'thinker.audio_tower.layers.{i}.fc1.weight',
    'p_fc1_bias': 'thinker.audio_tower.layers.{i}.fc1.bias',
    'p_fc2_weight': 'thinker.audio_tower.layers.{i}.fc2.weight',
    'p_fc2_bias': 'thinker.audio_tower.layers.{i}.fc2.bias',
    'p_final_layer_norm_weight': 'thinker.audio_tower.layers.{i}.final_layer_norm.weight',
    'p_final_layer_norm_bias': 'thinker.audio_tower.layers.{i}.final_layer_norm.bias',
}

ENCODER_LAYER_OUTPUT: str = 'add_1'


def create_encoder_layer(prefix: str) -> EncoderLayerTensors:
    return EncoderLayerTensors(
        p_attn_k_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_k_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_k_proj_bias=LogicalTensor(
            name=f"{prefix}.p_attn_k_proj_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_v_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_v_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_v_proj_bias=LogicalTensor(
            name=f"{prefix}.p_attn_v_proj_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_q_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_q_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_q_proj_bias=LogicalTensor(
            name=f"{prefix}.p_attn_q_proj_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_out_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_out_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_out_proj_bias=LogicalTensor(
            name=f"{prefix}.p_attn_out_proj_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_layer_norm_weight=LogicalTensor(
            name=f"{prefix}.p_attn_layer_norm_weight",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_layer_norm_bias=LogicalTensor(
            name=f"{prefix}.p_attn_layer_norm_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_fc1_weight=LogicalTensor(
            name=f"{prefix}.p_fc1_weight",
            spec=TensorSpec(dtype='float32', shape=(3584, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_fc1_bias=LogicalTensor(
            name=f"{prefix}.p_fc1_bias",
            spec=TensorSpec(dtype='float32', shape=(3584,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_fc2_weight=LogicalTensor(
            name=f"{prefix}.p_fc2_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 3584)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_fc2_bias=LogicalTensor(
            name=f"{prefix}.p_fc2_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_final_layer_norm_weight=LogicalTensor(
            name=f"{prefix}.p_final_layer_norm_weight",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_final_layer_norm_bias=LogicalTensor(
            name=f"{prefix}.p_final_layer_norm_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        hidden_states=LogicalTensor(
            name=f"{prefix}.hidden_states",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        cu_seqlens=LogicalTensor(
            name=f"{prefix}.cu_seqlens",
            spec=TensorSpec(dtype='int32', shape=(3,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        attention_mask=LogicalTensor(
            name=f"{prefix}.attention_mask",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 133, 133)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        layer_norm=LogicalTensor(
            name=f"{prefix}.layer_norm",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear=LogicalTensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        reshape=LogicalTensor(
            name=f"{prefix}.reshape",
            spec=TensorSpec(dtype='float32', shape=(133, 14, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_1=LogicalTensor(
            name=f"{prefix}.linear_1",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        reshape_1=LogicalTensor(
            name=f"{prefix}.reshape_1",
            spec=TensorSpec(dtype='float32', shape=(133, 14, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_2=LogicalTensor(
            name=f"{prefix}.linear_2",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        reshape_2=LogicalTensor(
            name=f"{prefix}.reshape_2",
            spec=TensorSpec(dtype='float32', shape=(133, 14, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose=LogicalTensor(
            name=f"{prefix}.transpose",
            spec=TensorSpec(dtype='float32', shape=(14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        unsqueeze=LogicalTensor(
            name=f"{prefix}.unsqueeze",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose_1=LogicalTensor(
            name=f"{prefix}.transpose_1",
            spec=TensorSpec(dtype='float32', shape=(14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        unsqueeze_1=LogicalTensor(
            name=f"{prefix}.unsqueeze_1",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose_2=LogicalTensor(
            name=f"{prefix}.transpose_2",
            spec=TensorSpec(dtype='float32', shape=(14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        unsqueeze_2=LogicalTensor(
            name=f"{prefix}.unsqueeze_2",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        slice_1=LogicalTensor(
            name=f"{prefix}.slice_1",
            spec=TensorSpec(dtype='int32', shape=(2,)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        slice_2=LogicalTensor(
            name=f"{prefix}.slice_2",
            spec=TensorSpec(dtype='int32', shape=(2,)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        sub=LogicalTensor(
            name=f"{prefix}.sub",
            spec=TensorSpec(dtype='int32', shape=(2,)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        max_1=LogicalTensor(
            name=f"{prefix}.max_1",
            spec=TensorSpec(dtype='int32', shape=()),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        scaled_dot_product_attention=LogicalTensor(
            name=f"{prefix}.scaled_dot_product_attention",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose_3=LogicalTensor(
            name=f"{prefix}.transpose_3",
            spec=TensorSpec(dtype='float32', shape=(1, 133, 14, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        contiguous=LogicalTensor(
            name=f"{prefix}.contiguous",
            spec=TensorSpec(dtype='float32', shape=(1, 133, 14, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        reshape_3=LogicalTensor(
            name=f"{prefix}.reshape_3",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_3=LogicalTensor(
            name=f"{prefix}.linear_3",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add=LogicalTensor(
            name=f"{prefix}.add",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        layer_norm_1=LogicalTensor(
            name=f"{prefix}.layer_norm_1",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_4=LogicalTensor(
            name=f"{prefix}.linear_4",
            spec=TensorSpec(dtype='float32', shape=(133, 3584)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        gelu=LogicalTensor(
            name=f"{prefix}.gelu",
            spec=TensorSpec(dtype='float32', shape=(133, 3584)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_5=LogicalTensor(
            name=f"{prefix}.linear_5",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_1=LogicalTensor(
            name=f"{prefix}.add_1",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )

