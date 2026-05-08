"""Generated dispatch functions for OmniVoice submodules."""

from __future__ import annotations

from models.exported_omnivoice.shaders.audio_head_export_linear_nobias_f32 import AUDIO_HEAD_EXPORT_LINEAR_NOBIAS_F32
from models.exported_omnivoice.shaders.export_add_f32 import EXPORT_ADD_F32
from models.exported_omnivoice.shaders.export_add_f32_37 import EXPORT_ADD_F32_37
from models.exported_omnivoice.shaders.export_add_f32_43 import EXPORT_ADD_F32_43
from models.exported_omnivoice.shaders.export_add_scalar import EXPORT_ADD_SCALAR
from models.exported_omnivoice.shaders.export_add_scalar_17 import EXPORT_ADD_SCALAR_17
from models.exported_omnivoice.shaders.export_add_scalar_9 import EXPORT_ADD_SCALAR_9
from models.exported_omnivoice.shaders.export_cat_f32 import EXPORT_CAT_F32
from models.exported_omnivoice.shaders.export_cat_f32_32 import EXPORT_CAT_F32_32
from models.exported_omnivoice.shaders.export_embedding_f32 import EXPORT_EMBEDDING_F32
from models.exported_omnivoice.shaders.export_linear_nobias_f32 import EXPORT_LINEAR_NOBIAS_F32
from models.exported_omnivoice.shaders.export_linear_nobias_f32_14 import EXPORT_LINEAR_NOBIAS_F32_14
from models.exported_omnivoice.shaders.export_linear_nobias_f32_22 import EXPORT_LINEAR_NOBIAS_F32_22
from models.exported_omnivoice.shaders.export_linear_nobias_f32_36 import EXPORT_LINEAR_NOBIAS_F32_36
from models.exported_omnivoice.shaders.export_linear_nobias_f32_38 import EXPORT_LINEAR_NOBIAS_F32_38
from models.exported_omnivoice.shaders.export_linear_nobias_f32_40 import EXPORT_LINEAR_NOBIAS_F32_40
from models.exported_omnivoice.shaders.export_linear_nobias_f32_42 import EXPORT_LINEAR_NOBIAS_F32_42
from models.exported_omnivoice.shaders.export_mean_dim_f32 import EXPORT_MEAN_DIM_F32
from models.exported_omnivoice.shaders.export_mean_dim_f32_16 import EXPORT_MEAN_DIM_F32_16
from models.exported_omnivoice.shaders.export_mean_dim_f32_8 import EXPORT_MEAN_DIM_F32_8
from models.exported_omnivoice.shaders.export_mul_broadcast_inner import EXPORT_MUL_BROADCAST_INNER
from models.exported_omnivoice.shaders.export_mul_broadcast_inner_29 import EXPORT_MUL_BROADCAST_INNER_29
from models.exported_omnivoice.shaders.export_mul_broadcast_inner_33 import EXPORT_MUL_BROADCAST_INNER_33
from models.exported_omnivoice.shaders.export_mul_broadcast_last import EXPORT_MUL_BROADCAST_LAST
from models.exported_omnivoice.shaders.export_mul_broadcast_last_11 import EXPORT_MUL_BROADCAST_LAST_11
from models.exported_omnivoice.shaders.export_mul_broadcast_last_19 import EXPORT_MUL_BROADCAST_LAST_19
from models.exported_omnivoice.shaders.export_mul_f32 import EXPORT_MUL_F32
from models.exported_omnivoice.shaders.export_mul_left_broadcast import EXPORT_MUL_LEFT_BROADCAST
from models.exported_omnivoice.shaders.export_mul_left_broadcast_12 import EXPORT_MUL_LEFT_BROADCAST_12
from models.exported_omnivoice.shaders.export_mul_left_broadcast_20 import EXPORT_MUL_LEFT_BROADCAST_20
from models.exported_omnivoice.shaders.export_neg_f32 import EXPORT_NEG_F32
from models.exported_omnivoice.shaders.export_pow_scalar_f32 import EXPORT_POW_SCALAR_F32
from models.exported_omnivoice.shaders.export_pow_scalar_f32_15 import EXPORT_POW_SCALAR_F32_15
from models.exported_omnivoice.shaders.export_pow_scalar_f32_7 import EXPORT_POW_SCALAR_F32_7
from models.exported_omnivoice.shaders.export_rsqrt_f32 import EXPORT_RSQRT_F32
from models.exported_omnivoice.shaders.export_rsqrt_f32_10 import EXPORT_RSQRT_F32_10
from models.exported_omnivoice.shaders.export_rsqrt_f32_18 import EXPORT_RSQRT_F32_18
from models.exported_omnivoice.shaders.export_sdpa_masked_f32 import EXPORT_SDPA_MASKED_F32
from models.exported_omnivoice.shaders.export_silu_f32 import EXPORT_SILU_F32
from models.exported_omnivoice.shaders.export_slice_f32 import EXPORT_SLICE_F32
from models.exported_omnivoice.shaders.export_slice_f32_25 import EXPORT_SLICE_F32_25
from models.exported_omnivoice.shaders.export_slice_f32_30 import EXPORT_SLICE_F32_30
from models.exported_omnivoice.shaders.export_slice_f32_31 import EXPORT_SLICE_F32_31
from models.exported_omnivoice.shaders.export_transpose_f32_45de1e4f84 import EXPORT_TRANSPOSE_F32_45DE1E4F84
from models.exported_omnivoice.shaders.export_transpose_f32_c943282b28 import EXPORT_TRANSPOSE_F32_C943282B28
from models.exported_omnivoice.shaders.export_transpose_f32_f3e8fdf2d4 import EXPORT_TRANSPOSE_F32_F3E8FDF2D4

from models.exported_omnivoice.tensors.audio_embed import AudioEmbedTensors
from models.exported_omnivoice.tensors.audio_head import AudioHeadTensors
from models.exported_omnivoice.tensors.llm_forward import LlmForwardTensors
from models.exported_omnivoice.tensors.text_embed import TextEmbedTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession


SHADER_VARIANTS_BY_NAME: dict[str, ShaderVariant] = {
    'audio_head_export_linear_nobias_f32': AUDIO_HEAD_EXPORT_LINEAR_NOBIAS_F32,
    'export_add_f32': EXPORT_ADD_F32,
    'export_add_f32_37': EXPORT_ADD_F32_37,
    'export_add_f32_43': EXPORT_ADD_F32_43,
    'export_add_scalar': EXPORT_ADD_SCALAR,
    'export_add_scalar_17': EXPORT_ADD_SCALAR_17,
    'export_add_scalar_9': EXPORT_ADD_SCALAR_9,
    'export_cat_f32': EXPORT_CAT_F32,
    'export_cat_f32_32': EXPORT_CAT_F32_32,
    'export_embedding_f32': EXPORT_EMBEDDING_F32,
    'export_linear_nobias_f32': EXPORT_LINEAR_NOBIAS_F32,
    'export_linear_nobias_f32_14': EXPORT_LINEAR_NOBIAS_F32_14,
    'export_linear_nobias_f32_22': EXPORT_LINEAR_NOBIAS_F32_22,
    'export_linear_nobias_f32_36': EXPORT_LINEAR_NOBIAS_F32_36,
    'export_linear_nobias_f32_38': EXPORT_LINEAR_NOBIAS_F32_38,
    'export_linear_nobias_f32_40': EXPORT_LINEAR_NOBIAS_F32_40,
    'export_linear_nobias_f32_42': EXPORT_LINEAR_NOBIAS_F32_42,
    'export_mean_dim_f32': EXPORT_MEAN_DIM_F32,
    'export_mean_dim_f32_16': EXPORT_MEAN_DIM_F32_16,
    'export_mean_dim_f32_8': EXPORT_MEAN_DIM_F32_8,
    'export_mul_broadcast_inner': EXPORT_MUL_BROADCAST_INNER,
    'export_mul_broadcast_inner_29': EXPORT_MUL_BROADCAST_INNER_29,
    'export_mul_broadcast_inner_33': EXPORT_MUL_BROADCAST_INNER_33,
    'export_mul_broadcast_last': EXPORT_MUL_BROADCAST_LAST,
    'export_mul_broadcast_last_11': EXPORT_MUL_BROADCAST_LAST_11,
    'export_mul_broadcast_last_19': EXPORT_MUL_BROADCAST_LAST_19,
    'export_mul_f32': EXPORT_MUL_F32,
    'export_mul_left_broadcast': EXPORT_MUL_LEFT_BROADCAST,
    'export_mul_left_broadcast_12': EXPORT_MUL_LEFT_BROADCAST_12,
    'export_mul_left_broadcast_20': EXPORT_MUL_LEFT_BROADCAST_20,
    'export_neg_f32': EXPORT_NEG_F32,
    'export_pow_scalar_f32': EXPORT_POW_SCALAR_F32,
    'export_pow_scalar_f32_15': EXPORT_POW_SCALAR_F32_15,
    'export_pow_scalar_f32_7': EXPORT_POW_SCALAR_F32_7,
    'export_rsqrt_f32': EXPORT_RSQRT_F32,
    'export_rsqrt_f32_10': EXPORT_RSQRT_F32_10,
    'export_rsqrt_f32_18': EXPORT_RSQRT_F32_18,
    'export_sdpa_masked_f32': EXPORT_SDPA_MASKED_F32,
    'export_silu_f32': EXPORT_SILU_F32,
    'export_slice_f32': EXPORT_SLICE_F32,
    'export_slice_f32_25': EXPORT_SLICE_F32_25,
    'export_slice_f32_30': EXPORT_SLICE_F32_30,
    'export_slice_f32_31': EXPORT_SLICE_F32_31,
    'export_transpose_f32_45de1e4f84': EXPORT_TRANSPOSE_F32_45DE1E4F84,
    'export_transpose_f32_c943282b28': EXPORT_TRANSPOSE_F32_C943282B28,
    'export_transpose_f32_f3e8fdf2d4': EXPORT_TRANSPOSE_F32_F3E8FDF2D4,
}


def run_text_embed(rt: RuntimeSession, tensors: TextEmbedTensors) -> None:
    EXPORT_EMBEDDING_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_audio_embed(rt: RuntimeSession, tensors: AudioEmbedTensors) -> None:
    EXPORT_EMBEDDING_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_llm_forward(rt: RuntimeSession, tensors: LlmForwardTensors) -> None:
    carry = tensors.hidden_states
    for layer_t in tensors.layers:
        _alias(rt, carry, layer_t.to)
        EXPORT_POW_SCALAR_F32(rt, x=layer_t.to, output=layer_t.pow_1)
        EXPORT_MEAN_DIM_F32(rt, x=layer_t.pow_1, output=layer_t.mean)
        EXPORT_ADD_SCALAR(rt, x=layer_t.mean, output=layer_t.add)
        EXPORT_RSQRT_F32(rt, x=layer_t.add, output=layer_t.rsqrt)
        EXPORT_MUL_BROADCAST_LAST(rt, x=layer_t.to, y=layer_t.rsqrt, output=layer_t.mul)
        _alias(rt, layer_t.mul, layer_t.to_1)
        EXPORT_MUL_LEFT_BROADCAST(rt, x=layer_t.p_layers_0_input_layernorm_weight, y=layer_t.to_1, output=layer_t.mul_1)
        EXPORT_LINEAR_NOBIAS_F32(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_q_proj_weight, output=layer_t.linear)
        _alias(rt, layer_t.linear, layer_t.view)
        _alias(rt, layer_t.view, layer_t.to_2)
        EXPORT_POW_SCALAR_F32_7(rt, x=layer_t.to_2, output=layer_t.pow_2)
        EXPORT_MEAN_DIM_F32_8(rt, x=layer_t.pow_2, output=layer_t.mean_1)
        EXPORT_ADD_SCALAR_9(rt, x=layer_t.mean_1, output=layer_t.add_1)
        EXPORT_RSQRT_F32_10(rt, x=layer_t.add_1, output=layer_t.rsqrt_1)
        EXPORT_MUL_BROADCAST_LAST_11(rt, x=layer_t.to_2, y=layer_t.rsqrt_1, output=layer_t.mul_2)
        _alias(rt, layer_t.mul_2, layer_t.to_3)
        EXPORT_MUL_LEFT_BROADCAST_12(rt, x=layer_t.p_layers_0_self_attn_q_norm_weight, y=layer_t.to_3, output=layer_t.mul_3)
        EXPORT_TRANSPOSE_F32_F3E8FDF2D4(rt, x=layer_t.mul_3, output=layer_t.transpose)
        EXPORT_LINEAR_NOBIAS_F32_14(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_k_proj_weight, output=layer_t.linear_1)
        _alias(rt, layer_t.linear_1, layer_t.view_1)
        _alias(rt, layer_t.view_1, layer_t.to_4)
        EXPORT_POW_SCALAR_F32_15(rt, x=layer_t.to_4, output=layer_t.pow_3)
        EXPORT_MEAN_DIM_F32_16(rt, x=layer_t.pow_3, output=layer_t.mean_2)
        EXPORT_ADD_SCALAR_17(rt, x=layer_t.mean_2, output=layer_t.add_2)
        EXPORT_RSQRT_F32_18(rt, x=layer_t.add_2, output=layer_t.rsqrt_2)
        EXPORT_MUL_BROADCAST_LAST_19(rt, x=layer_t.to_4, y=layer_t.rsqrt_2, output=layer_t.mul_4)
        _alias(rt, layer_t.mul_4, layer_t.to_5)
        EXPORT_MUL_LEFT_BROADCAST_20(rt, x=layer_t.p_layers_0_self_attn_k_norm_weight, y=layer_t.to_5, output=layer_t.mul_5)
        EXPORT_TRANSPOSE_F32_C943282B28(rt, x=layer_t.mul_5, output=layer_t.transpose_1)
        EXPORT_LINEAR_NOBIAS_F32_22(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_v_proj_weight, output=layer_t.linear_2)
        _alias(rt, layer_t.linear_2, layer_t.view_2)
        EXPORT_TRANSPOSE_F32_C943282B28(rt, x=layer_t.view_2, output=layer_t.transpose_2)
        _alias(rt, tensors.cos, layer_t.unsqueeze)
        _alias(rt, tensors.sin, layer_t.unsqueeze_1)
        EXPORT_MUL_BROADCAST_INNER(rt, x=layer_t.transpose, y=layer_t.unsqueeze, output=layer_t.mul_6)
        EXPORT_SLICE_F32(rt, x=layer_t.transpose, output=layer_t.slice_1)
        EXPORT_SLICE_F32_25(rt, x=layer_t.transpose, output=layer_t.slice_2)
        EXPORT_NEG_F32(rt, x=layer_t.slice_2, output=layer_t.neg)
        EXPORT_CAT_F32(rt, a=layer_t.neg, b=layer_t.slice_1, output=layer_t.cat)
        EXPORT_MUL_BROADCAST_INNER(rt, x=layer_t.cat, y=layer_t.unsqueeze_1, output=layer_t.mul_7)
        EXPORT_ADD_F32(rt, x=layer_t.mul_6, y=layer_t.mul_7, output=layer_t.add_3)
        EXPORT_MUL_BROADCAST_INNER_29(rt, x=layer_t.transpose_1, y=layer_t.unsqueeze, output=layer_t.mul_8)
        EXPORT_SLICE_F32_30(rt, x=layer_t.transpose_1, output=layer_t.slice_3)
        EXPORT_SLICE_F32_31(rt, x=layer_t.transpose_1, output=layer_t.slice_4)
        EXPORT_NEG_F32(rt, x=layer_t.slice_4, output=layer_t.neg_1)
        EXPORT_CAT_F32_32(rt, a=layer_t.neg_1, b=layer_t.slice_3, output=layer_t.cat_1)
        EXPORT_MUL_BROADCAST_INNER_33(rt, x=layer_t.cat_1, y=layer_t.unsqueeze_1, output=layer_t.mul_9)
        EXPORT_ADD_F32(rt, x=layer_t.mul_8, y=layer_t.mul_9, output=layer_t.add_4)
        EXPORT_SDPA_MASKED_F32(rt, q=layer_t.add_3, k=layer_t.add_4, v=layer_t.transpose_2, mask=tensors.attention_mask, output=layer_t.scaled_dot_product_attention)
        EXPORT_TRANSPOSE_F32_45DE1E4F84(rt, x=layer_t.scaled_dot_product_attention, output=layer_t.transpose_3)
        _alias(rt, layer_t.transpose_3, layer_t.contiguous)
        _alias(rt, layer_t.contiguous, layer_t.reshape)
        EXPORT_LINEAR_NOBIAS_F32_36(rt, x=layer_t.reshape, weight=layer_t.p_layers_0_self_attn_o_proj_weight, output=layer_t.linear_3)
        EXPORT_ADD_F32_37(rt, x=layer_t.to, y=layer_t.linear_3, output=layer_t.add_5)
        _alias(rt, layer_t.add_5, layer_t.to_6)
        EXPORT_POW_SCALAR_F32(rt, x=layer_t.to_6, output=layer_t.pow_4)
        EXPORT_MEAN_DIM_F32(rt, x=layer_t.pow_4, output=layer_t.mean_3)
        EXPORT_ADD_SCALAR(rt, x=layer_t.mean_3, output=layer_t.add_6)
        EXPORT_RSQRT_F32(rt, x=layer_t.add_6, output=layer_t.rsqrt_3)
        EXPORT_MUL_BROADCAST_LAST(rt, x=layer_t.to_6, y=layer_t.rsqrt_3, output=layer_t.mul_10)
        _alias(rt, layer_t.mul_10, layer_t.to_7)
        EXPORT_MUL_LEFT_BROADCAST(rt, x=layer_t.p_layers_0_post_attention_layernorm_weight, y=layer_t.to_7, output=layer_t.mul_11)
        EXPORT_LINEAR_NOBIAS_F32_38(rt, x=layer_t.mul_11, weight=layer_t.p_layers_0_mlp_gate_proj_weight, output=layer_t.linear_4)
        EXPORT_SILU_F32(rt, x=layer_t.linear_4, output=layer_t.silu)
        EXPORT_LINEAR_NOBIAS_F32_40(rt, x=layer_t.mul_11, weight=layer_t.p_layers_0_mlp_up_proj_weight, output=layer_t.linear_5)
        EXPORT_MUL_F32(rt, x=layer_t.silu, y=layer_t.linear_5, output=layer_t.mul_12)
        EXPORT_LINEAR_NOBIAS_F32_42(rt, x=layer_t.mul_12, weight=layer_t.p_layers_0_mlp_down_proj_weight, output=layer_t.linear_6)
        EXPORT_ADD_F32_43(rt, x=layer_t.to_6, y=layer_t.linear_6, output=layer_t.add_7)
        carry = layer_t.add_7
    _alias(rt, carry, tensors.to_224)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to_224, output=tensors.pow_113)
    EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_113, output=tensors.mean_112)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean_112, output=tensors.add_224)
    EXPORT_RSQRT_F32(rt, x=tensors.add_224, output=tensors.rsqrt_112)
    EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to_224, y=tensors.rsqrt_112, output=tensors.mul_364)
    _alias(rt, tensors.mul_364, tensors.to_225)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_norm_weight, y=tensors.to_225, output=tensors.mul_365)


def run_audio_head(rt: RuntimeSession, tensors: AudioHeadTensors) -> None:
    AUDIO_HEAD_EXPORT_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:
    rt._materialize_read(src)
    with dst.runtime_write_scope():
        dst.buffer = src.buffer
        dst.descriptor_nbytes = src.descriptor_nbytes
        dst.version = src.version
        dst.writer = src.writer
    rt._current_frame().written_tensors.append(dst)
