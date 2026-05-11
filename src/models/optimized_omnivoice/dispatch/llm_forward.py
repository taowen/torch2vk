"""Optimized dispatch function for run_llm_forward."""

from __future__ import annotations

from models.optimized_omnivoice.tensors.model import model_tensors
from models.optimized_omnivoice.shaders.add_f32_37 import ADD_F32_37
from models.optimized_omnivoice.shaders.add_f32_43 import ADD_F32_43
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32_14 import LINEAR_NOBIAS_Q4_K_F32_14
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32_22 import LINEAR_NOBIAS_Q4_K_F32_22
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32_36 import LINEAR_NOBIAS_Q4_K_F32_36
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32_38 import LINEAR_NOBIAS_Q4_K_F32_38
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32_40 import LINEAR_NOBIAS_Q4_K_F32_40
from models.optimized_omnivoice.shaders.linear_nobias_q4_k_f32_42 import LINEAR_NOBIAS_Q4_K_F32_42
from models.optimized_omnivoice.shaders.omnivoice_rms_norm_3d_f32 import (
    OMNIVOICE_RMS_NORM_3D_F32,
)
from models.optimized_omnivoice.shaders.omnivoice_rms_norm_4d_f32 import (
    OMNIVOICE_RMS_NORM_4D_F32,
)
from models.optimized_omnivoice.shaders.omnivoice_rotary_fused_f32 import (
    OMNIVOICE_ROTARY_FUSED_F32,
)
from models.optimized_omnivoice.shaders.omnivoice_sdpa_masked_f32 import (
    OMNIVOICE_SDPA_MASKED_F32,
)
from models.optimized_omnivoice.shaders.omnivoice_silu_mul_f32 import OMNIVOICE_SILU_MUL_F32
from models.optimized_omnivoice.shaders.transpose_f32_322f87bdab import (
    TRANSPOSE_F32_322F87BDAB,
)
from models.optimized_omnivoice.shaders.transpose_f32_b4bb3205fc import (
    TRANSPOSE_F32_B4BB3205FC,
)
from models.optimized_omnivoice.shaders.transpose_f32_e6f353739d import (
    TRANSPOSE_F32_E6F353739D,
)
from models.optimized_omnivoice.tensors.llm_forward import LlmForwardTensors
from torch2vk.runtime.session import RuntimeSession


def _run_llm_forward_with_tensors(rt: RuntimeSession, tensors: LlmForwardTensors) -> None:
    for layer_t in tensors.layers:
        OMNIVOICE_RMS_NORM_3D_F32(
            rt,
            x=layer_t.to,
            weight=layer_t.p_layers_0_input_layernorm_weight,
            output=layer_t.mul_1,
        )
        LINEAR_NOBIAS_Q4_K_F32(
            rt,
            x=layer_t.mul_1,
            weight=layer_t.p_layers_0_self_attn_q_proj_weight,
            output=layer_t.linear,
        )
        OMNIVOICE_RMS_NORM_4D_F32(
            rt,
            x=layer_t.to_2,
            weight=layer_t.p_layers_0_self_attn_q_norm_weight,
            output=layer_t.mul_3,
        )
        TRANSPOSE_F32_322F87BDAB(rt, x=layer_t.mul_3, output=layer_t.transpose)
        LINEAR_NOBIAS_Q4_K_F32_14(
            rt,
            x=layer_t.mul_1,
            weight=layer_t.p_layers_0_self_attn_k_proj_weight,
            output=layer_t.linear_1,
        )
        OMNIVOICE_RMS_NORM_4D_F32(
            rt,
            x=layer_t.to_4,
            weight=layer_t.p_layers_0_self_attn_k_norm_weight,
            output=layer_t.mul_5,
        )
        TRANSPOSE_F32_B4BB3205FC(rt, x=layer_t.mul_5, output=layer_t.transpose_1)
        LINEAR_NOBIAS_Q4_K_F32_22(
            rt,
            x=layer_t.mul_1,
            weight=layer_t.p_layers_0_self_attn_v_proj_weight,
            output=layer_t.linear_2,
        )
        TRANSPOSE_F32_B4BB3205FC(rt, x=layer_t.view_2, output=layer_t.transpose_2)
        OMNIVOICE_ROTARY_FUSED_F32(
            rt,
            x=layer_t.transpose,
            cos=layer_t.unsqueeze,
            sin=layer_t.unsqueeze_1,
            output=layer_t.add_3,
        )
        OMNIVOICE_ROTARY_FUSED_F32(
            rt,
            x=layer_t.transpose_1,
            cos=layer_t.unsqueeze,
            sin=layer_t.unsqueeze_1,
            output=layer_t.add_4,
        )
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
        LINEAR_NOBIAS_Q4_K_F32_36(
            rt,
            x=layer_t.reshape,
            weight=layer_t.p_layers_0_self_attn_o_proj_weight,
            output=layer_t.linear_3,
        )
        ADD_F32_37(rt, x=layer_t.to, y=layer_t.linear_3, output=layer_t.add_5)
        OMNIVOICE_RMS_NORM_3D_F32(
            rt,
            x=layer_t.to_6,
            weight=layer_t.p_layers_0_post_attention_layernorm_weight,
            output=layer_t.mul_11,
        )
        LINEAR_NOBIAS_Q4_K_F32_38(
            rt,
            x=layer_t.mul_11,
            weight=layer_t.p_layers_0_mlp_gate_proj_weight,
            output=layer_t.linear_4,
        )
        LINEAR_NOBIAS_Q4_K_F32_40(
            rt,
            x=layer_t.mul_11,
            weight=layer_t.p_layers_0_mlp_up_proj_weight,
            output=layer_t.linear_5,
        )
        OMNIVOICE_SILU_MUL_F32(
            rt,
            x=layer_t.linear_4,
            y=layer_t.linear_5,
            output=layer_t.mul_12,
        )
        LINEAR_NOBIAS_Q4_K_F32_42(
            rt,
            x=layer_t.mul_12,
            weight=layer_t.p_layers_0_mlp_down_proj_weight,
            output=layer_t.linear_6,
        )
        ADD_F32_43(rt, x=layer_t.to_6, y=layer_t.linear_6, output=layer_t.add_7)
    OMNIVOICE_RMS_NORM_3D_F32(
        rt,
        x=tensors.to_224,
        weight=tensors.p_norm_weight,
        output=tensors.mul_365,
    )


def run_llm_forward(rt: RuntimeSession) -> None:
    _run_llm_forward_with_tensors(rt, model_tensors().llm_forward)
