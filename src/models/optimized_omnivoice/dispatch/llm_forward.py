"""Optimized dispatch function for run_llm_forward."""

from __future__ import annotations

from models.optimized_omnivoice.tensors.model import model_tensors
from models.optimized_omnivoice.shaders.add_f32_32 import ADD_F32_32
from models.optimized_omnivoice.shaders.add_f32_35 import ADD_F32_35
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_pair_f32 import (
    LINEAR_NOBIAS_Q4_K_PAIR_F32,
)
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_swiglu_f32 import (
    LINEAR_NOBIAS_Q4_K_SWIGLU_F32,
)
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_triple_f32 import (
    LINEAR_NOBIAS_Q4_K_TRIPLE_F32,
)
from models.optimized_omnivoice.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.optimized_omnivoice.shaders.omnivoice_sdpa_masked_f32 import (
    OMNIVOICE_SDPA_MASKED_F32,
)
from models.optimized_omnivoice.shaders.rms_norm_mul_f16_f32 import RMS_NORM_MUL_F16_F32
from models.optimized_omnivoice.shaders.rms_norm_rope_transpose_f16 import (
    RMS_NORM_ROPE_TRANSPOSE_F16,
)
from models.optimized_omnivoice.shaders.transpose_f32_e6f353739d import (
    TRANSPOSE_F32_E6F353739D,
)
from models.optimized_omnivoice.tensors.llm_forward import LlmForwardTensors, LlmLayerTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.types import Q4KWordsLayout, Q6KHalfwordsLayout


def _linear_q4_or_q6(
    rt: RuntimeSession,
    *,
    q4: ShaderVariant,
    q6: ShaderVariant,
    x: LogicalTensor,
    weight: LogicalTensor,
    output: LogicalTensor,
) -> None:
    if isinstance(weight.layout, Q6KHalfwordsLayout):
        q6(rt, x=x, weight=weight, output=output)
        return
    if not isinstance(weight.layout, Q4KWordsLayout):
        raise ValueError(f"{weight.name} expected Q4_K or Q6_K layout, got {weight.layout}")
    q4(rt, x=x, weight=weight, output=output)


def _attn_qkv(rt: RuntimeSession, tensors: LlmLayerTensors) -> None:
    if isinstance(tensors.p_layers_0_self_attn_v_proj_weight.layout, Q6KHalfwordsLayout):
        LINEAR_NOBIAS_Q4_K_PAIR_F32(
            rt,
            x=tensors.mul_1,
            weight0=tensors.p_layers_0_self_attn_q_proj_weight,
            weight1=tensors.p_layers_0_self_attn_k_proj_weight,
            output0=tensors.linear,
            output1=tensors.linear_1,
        )
        LINEAR_NOBIAS_Q6_K_F32(
            rt,
            x=tensors.mul_1,
            weight=tensors.p_layers_0_self_attn_v_proj_weight,
            output=tensors.linear_2,
        )
        return
    if not isinstance(tensors.p_layers_0_self_attn_v_proj_weight.layout, Q4KWordsLayout):
        raise ValueError(
            f"{tensors.p_layers_0_self_attn_v_proj_weight.name} expected Q4_K or Q6_K "
            f"layout, got {tensors.p_layers_0_self_attn_v_proj_weight.layout}"
        )
    LINEAR_NOBIAS_Q4_K_TRIPLE_F32(
        rt,
        x=tensors.mul_1,
        weight0=tensors.p_layers_0_self_attn_q_proj_weight,
        weight1=tensors.p_layers_0_self_attn_k_proj_weight,
        weight2=tensors.p_layers_0_self_attn_v_proj_weight,
        output0=tensors.linear,
        output1=tensors.linear_1,
        output2=tensors.linear_2,
    )


def _mlp_swiglu(rt: RuntimeSession, tensors: LlmLayerTensors) -> None:
    if not isinstance(tensors.p_layers_0_mlp_gate_proj_weight.layout, Q4KWordsLayout):
        raise ValueError(
            f"{tensors.p_layers_0_mlp_gate_proj_weight.name} expected Q4_K layout, "
            f"got {tensors.p_layers_0_mlp_gate_proj_weight.layout}"
        )
    if not isinstance(tensors.p_layers_0_mlp_up_proj_weight.layout, Q4KWordsLayout):
        raise ValueError(
            f"{tensors.p_layers_0_mlp_up_proj_weight.name} expected Q4_K layout, "
            f"got {tensors.p_layers_0_mlp_up_proj_weight.layout}"
        )
    LINEAR_NOBIAS_Q4_K_SWIGLU_F32(
        rt,
        x=tensors.mul_11,
        gate_weight=tensors.p_layers_0_mlp_gate_proj_weight,
        up_weight=tensors.p_layers_0_mlp_up_proj_weight,
        output=tensors.mul_12,
    )


def _run_llm_forward_with_tensors(rt: RuntimeSession, tensors: LlmForwardTensors) -> None:
    rope_t = model_tensors().rope
    for layer_t in tensors.layers:
        RMS_NORM_MUL_F16_F32(
            rt,
            x=layer_t.to,
            weight=layer_t.p_layers_0_input_layernorm_weight,
            output=layer_t.mul_1,
        )
        _attn_qkv(rt, layer_t)
        RMS_NORM_ROPE_TRANSPOSE_F16(
            rt,
            x=layer_t.to_2,
            weight=layer_t.p_layers_0_self_attn_q_norm_weight,
            cos=rope_t.cos,
            sin=rope_t.sin,
            output=layer_t.add_3,
        )
        RMS_NORM_ROPE_TRANSPOSE_F16(
            rt,
            x=layer_t.to_4,
            weight=layer_t.p_layers_0_self_attn_k_norm_weight,
            cos=rope_t.cos,
            sin=rope_t.sin,
            output=layer_t.add_4,
        )
        TRANSPOSE_F32_E6F353739D(rt, x=layer_t.view_2, output=layer_t.transpose_2)
        OMNIVOICE_SDPA_MASKED_F32(
            rt,
            q=layer_t.add_3,
            k=layer_t.add_4,
            v=layer_t.transpose_2,
            mask=tensors.attention_mask,
            output=layer_t.scaled_dot_product_attention,
        )
        TRANSPOSE_F32_E6F353739D(
            rt,
            x=layer_t.scaled_dot_product_attention,
            output=layer_t.transpose_3,
        )
        _linear_q4_or_q6(
            rt,
            q4=LINEAR_NOBIAS_Q4_K_F32,
            q6=LINEAR_NOBIAS_Q6_K_F32,
            x=layer_t.reshape,
            weight=layer_t.p_layers_0_self_attn_o_proj_weight,
            output=layer_t.linear_3,
        )
        ADD_F32_32(rt, x=layer_t.to, y=layer_t.linear_3, output=layer_t.add_5)
        RMS_NORM_MUL_F16_F32(
            rt,
            x=layer_t.to_6,
            weight=layer_t.p_layers_0_post_attention_layernorm_weight,
            output=layer_t.mul_11,
        )
        _mlp_swiglu(rt, layer_t)
        _linear_q4_or_q6(
            rt,
            q4=LINEAR_NOBIAS_Q4_K_F32,
            q6=LINEAR_NOBIAS_Q6_K_F32,
            x=layer_t.mul_12,
            weight=layer_t.p_layers_0_mlp_down_proj_weight,
            output=layer_t.linear_6,
        )
        ADD_F32_35(rt, x=layer_t.to_6, y=layer_t.linear_6, output=layer_t.add_7)
    RMS_NORM_MUL_F16_F32(
        rt,
        x=tensors.to_224,
        weight=tensors.p_norm_weight,
        output=tensors.mul_365,
    )


def run_llm_forward(rt: RuntimeSession) -> None:
    _run_llm_forward_with_tensors(rt, model_tensors().llm_forward)
