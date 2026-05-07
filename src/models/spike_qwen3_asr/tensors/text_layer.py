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
class TextLayerTensors:
    p_attn_q_proj_weight: LogicalTensor
    p_attn_k_proj_weight: LogicalTensor
    p_attn_v_proj_weight: LogicalTensor
    p_attn_o_proj_weight: LogicalTensor
    p_attn_q_norm_weight: LogicalTensor
    p_attn_k_norm_weight: LogicalTensor
    p_mlp_gate_proj_weight: LogicalTensor
    p_mlp_up_proj_weight: LogicalTensor
    p_mlp_down_proj_weight: LogicalTensor
    p_input_layernorm_weight: LogicalTensor
    p_post_attention_layernorm_weight: LogicalTensor
    hidden_states: LogicalTensor
    position_embeddings_0: LogicalTensor
    position_embeddings_1: LogicalTensor
    to: LogicalTensor
    pow_1: LogicalTensor
    mean: LogicalTensor
    add: LogicalTensor
    rsqrt: LogicalTensor
    mul: LogicalTensor
    to_1: LogicalTensor
    mul_1: LogicalTensor
    linear: LogicalTensor
    view: LogicalTensor
    to_2: LogicalTensor
    pow_2: LogicalTensor
    mean_1: LogicalTensor
    add_1: LogicalTensor
    rsqrt_1: LogicalTensor
    mul_2: LogicalTensor
    to_3: LogicalTensor
    mul_3: LogicalTensor
    transpose: LogicalTensor
    linear_1: LogicalTensor
    view_1: LogicalTensor
    to_4: LogicalTensor
    pow_3: LogicalTensor
    mean_2: LogicalTensor
    add_2: LogicalTensor
    rsqrt_2: LogicalTensor
    mul_4: LogicalTensor
    to_5: LogicalTensor
    mul_5: LogicalTensor
    transpose_1: LogicalTensor
    linear_2: LogicalTensor
    view_2: LogicalTensor
    transpose_2: LogicalTensor
    unsqueeze: LogicalTensor
    unsqueeze_1: LogicalTensor
    mul_6: LogicalTensor
    slice_1: LogicalTensor
    slice_2: LogicalTensor
    neg: LogicalTensor
    cat: LogicalTensor
    mul_7: LogicalTensor
    add_3: LogicalTensor
    mul_8: LogicalTensor
    slice_3: LogicalTensor
    slice_4: LogicalTensor
    neg_1: LogicalTensor
    cat_1: LogicalTensor
    mul_9: LogicalTensor
    add_4: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    transpose_3: LogicalTensor
    contiguous: LogicalTensor
    reshape: LogicalTensor
    linear_3: LogicalTensor
    add_5: LogicalTensor
    to_6: LogicalTensor
    pow_4: LogicalTensor
    mean_3: LogicalTensor
    add_6: LogicalTensor
    rsqrt_3: LogicalTensor
    mul_10: LogicalTensor
    to_7: LogicalTensor
    mul_11: LogicalTensor
    linear_4: LogicalTensor
    silu: LogicalTensor
    linear_5: LogicalTensor
    mul_12: LogicalTensor
    linear_6: LogicalTensor
    add_7: LogicalTensor


TEXT_LAYER_WEIGHT_MAP: dict[str, str] = {
    'p_attn_q_proj_weight': 'thinker.model.layers.{i}.self_attn.q_proj.weight',
    'p_attn_k_proj_weight': 'thinker.model.layers.{i}.self_attn.k_proj.weight',
    'p_attn_v_proj_weight': 'thinker.model.layers.{i}.self_attn.v_proj.weight',
    'p_attn_o_proj_weight': 'thinker.model.layers.{i}.self_attn.o_proj.weight',
    'p_attn_q_norm_weight': 'thinker.model.layers.{i}.self_attn.q_norm.weight',
    'p_attn_k_norm_weight': 'thinker.model.layers.{i}.self_attn.k_norm.weight',
    'p_mlp_gate_proj_weight': 'thinker.model.layers.{i}.mlp.gate_proj.weight',
    'p_mlp_up_proj_weight': 'thinker.model.layers.{i}.mlp.up_proj.weight',
    'p_mlp_down_proj_weight': 'thinker.model.layers.{i}.mlp.down_proj.weight',
    'p_input_layernorm_weight': 'thinker.model.layers.{i}.input_layernorm.weight',
    'p_post_attention_layernorm_weight': 'thinker.model.layers.{i}.post_attention_layernorm.weight',
}

TEXT_LAYER_OUTPUT: str = 'add_7'


def create_text_layer(prefix: str) -> TextLayerTensors:
    return TextLayerTensors(
        p_attn_q_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_q_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(2048, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_k_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_k_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(1024, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_v_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_v_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(1024, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_o_proj_weight=LogicalTensor(
            name=f"{prefix}.p_attn_o_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(1024, 2048)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_q_norm_weight=LogicalTensor(
            name=f"{prefix}.p_attn_q_norm_weight",
            spec=TensorSpec(dtype='float32', shape=(128,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_attn_k_norm_weight=LogicalTensor(
            name=f"{prefix}.p_attn_k_norm_weight",
            spec=TensorSpec(dtype='float32', shape=(128,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_mlp_gate_proj_weight=LogicalTensor(
            name=f"{prefix}.p_mlp_gate_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(3072, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_mlp_up_proj_weight=LogicalTensor(
            name=f"{prefix}.p_mlp_up_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(3072, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_mlp_down_proj_weight=LogicalTensor(
            name=f"{prefix}.p_mlp_down_proj_weight",
            spec=TensorSpec(dtype='float32', shape=(1024, 3072)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_input_layernorm_weight=LogicalTensor(
            name=f"{prefix}.p_input_layernorm_weight",
            spec=TensorSpec(dtype='float32', shape=(1024,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_post_attention_layernorm_weight=LogicalTensor(
            name=f"{prefix}.p_post_attention_layernorm_weight",
            spec=TensorSpec(dtype='float32', shape=(1024,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        hidden_states=LogicalTensor(
            name=f"{prefix}.hidden_states",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        position_embeddings_0=LogicalTensor(
            name=f"{prefix}.position_embeddings_0",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 128)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        position_embeddings_1=LogicalTensor(
            name=f"{prefix}.position_embeddings_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 128)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        to=LogicalTensor(
            name=f"{prefix}.to",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        pow_1=LogicalTensor(
            name=f"{prefix}.pow_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mean=LogicalTensor(
            name=f"{prefix}.mean",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add=LogicalTensor(
            name=f"{prefix}.add",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        rsqrt=LogicalTensor(
            name=f"{prefix}.rsqrt",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul=LogicalTensor(
            name=f"{prefix}.mul",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_1=LogicalTensor(
            name=f"{prefix}.to_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_1=LogicalTensor(
            name=f"{prefix}.mul_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear=LogicalTensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 2048)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        view=LogicalTensor(
            name=f"{prefix}.view",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_2=LogicalTensor(
            name=f"{prefix}.to_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        pow_2=LogicalTensor(
            name=f"{prefix}.pow_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mean_1=LogicalTensor(
            name=f"{prefix}.mean_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_1=LogicalTensor(
            name=f"{prefix}.add_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        rsqrt_1=LogicalTensor(
            name=f"{prefix}.rsqrt_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_2=LogicalTensor(
            name=f"{prefix}.mul_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_3=LogicalTensor(
            name=f"{prefix}.to_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_3=LogicalTensor(
            name=f"{prefix}.mul_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose=LogicalTensor(
            name=f"{prefix}.transpose",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_1=LogicalTensor(
            name=f"{prefix}.linear_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        view_1=LogicalTensor(
            name=f"{prefix}.view_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_4=LogicalTensor(
            name=f"{prefix}.to_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        pow_3=LogicalTensor(
            name=f"{prefix}.pow_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mean_2=LogicalTensor(
            name=f"{prefix}.mean_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_2=LogicalTensor(
            name=f"{prefix}.add_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        rsqrt_2=LogicalTensor(
            name=f"{prefix}.rsqrt_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_4=LogicalTensor(
            name=f"{prefix}.mul_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_5=LogicalTensor(
            name=f"{prefix}.to_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_5=LogicalTensor(
            name=f"{prefix}.mul_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose_1=LogicalTensor(
            name=f"{prefix}.transpose_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_2=LogicalTensor(
            name=f"{prefix}.linear_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        view_2=LogicalTensor(
            name=f"{prefix}.view_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose_2=LogicalTensor(
            name=f"{prefix}.transpose_2",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        unsqueeze=LogicalTensor(
            name=f"{prefix}.unsqueeze",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        unsqueeze_1=LogicalTensor(
            name=f"{prefix}.unsqueeze_1",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_6=LogicalTensor(
            name=f"{prefix}.mul_6",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        slice_1=LogicalTensor(
            name=f"{prefix}.slice_1",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        slice_2=LogicalTensor(
            name=f"{prefix}.slice_2",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        neg=LogicalTensor(
            name=f"{prefix}.neg",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        cat=LogicalTensor(
            name=f"{prefix}.cat",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_7=LogicalTensor(
            name=f"{prefix}.mul_7",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_3=LogicalTensor(
            name=f"{prefix}.add_3",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_8=LogicalTensor(
            name=f"{prefix}.mul_8",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        slice_3=LogicalTensor(
            name=f"{prefix}.slice_3",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        slice_4=LogicalTensor(
            name=f"{prefix}.slice_4",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        neg_1=LogicalTensor(
            name=f"{prefix}.neg_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 64)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        cat_1=LogicalTensor(
            name=f"{prefix}.cat_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_9=LogicalTensor(
            name=f"{prefix}.mul_9",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_4=LogicalTensor(
            name=f"{prefix}.add_4",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        scaled_dot_product_attention=LogicalTensor(
            name=f"{prefix}.scaled_dot_product_attention",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        transpose_3=LogicalTensor(
            name=f"{prefix}.transpose_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        contiguous=LogicalTensor(
            name=f"{prefix}.contiguous",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        reshape=LogicalTensor(
            name=f"{prefix}.reshape",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 2048)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_3=LogicalTensor(
            name=f"{prefix}.linear_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_5=LogicalTensor(
            name=f"{prefix}.add_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_6=LogicalTensor(
            name=f"{prefix}.to_6",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        pow_4=LogicalTensor(
            name=f"{prefix}.pow_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mean_3=LogicalTensor(
            name=f"{prefix}.mean_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_6=LogicalTensor(
            name=f"{prefix}.add_6",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        rsqrt_3=LogicalTensor(
            name=f"{prefix}.rsqrt_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_10=LogicalTensor(
            name=f"{prefix}.mul_10",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_7=LogicalTensor(
            name=f"{prefix}.to_7",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_11=LogicalTensor(
            name=f"{prefix}.mul_11",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_4=LogicalTensor(
            name=f"{prefix}.linear_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        silu=LogicalTensor(
            name=f"{prefix}.silu",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_5=LogicalTensor(
            name=f"{prefix}.linear_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_12=LogicalTensor(
            name=f"{prefix}.mul_12",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        linear_6=LogicalTensor(
            name=f"{prefix}.linear_6",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add_7=LogicalTensor(
            name=f"{prefix}.add_7",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )

