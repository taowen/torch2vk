"""Aten op shader factories for torch2vk.export."""

from torch2vk.export.shaders.add_f32 import make_add_variant
from torch2vk.export.shaders.argmax_f32 import make_argmax_variant
from torch2vk.export.shaders.cat_f32 import make_cat_variant
from torch2vk.export.shaders.conv1d_f32 import make_conv1d_variant
from torch2vk.export.shaders.conv1d_quantized import make_conv1d_q8_0_variant
from torch2vk.export.shaders.conv2d_f32 import make_conv2d_variant
from torch2vk.export.shaders.conv2d_quantized import make_conv2d_q8_0_variant
from torch2vk.export.shaders.conv_transpose1d_f32 import make_conv_transpose1d_variant
from torch2vk.export.shaders.conv_transpose1d_quantized import (
    make_conv_transpose1d_q8_0_variant,
)
from torch2vk.export.shaders.embedding_f32 import make_embedding_variant
from torch2vk.export.shaders.embedding_quantized import (
    make_embedding_q4_k_m_variant,
    make_embedding_q8_0_variant,
)
from torch2vk.export.shaders.gelu_f32 import make_gelu_variant
from torch2vk.export.shaders.index_copy_f32 import make_index_copy_variant
from torch2vk.export.shaders.index_select_f32 import make_index_select_variant
from torch2vk.export.shaders.layer_norm_f32 import make_layer_norm_variant
from torch2vk.export.shaders.linear_bias_f32 import make_linear_bias_variant
from torch2vk.export.shaders.linear_bias_quantized import make_linear_bias_q8_0_variant
from torch2vk.export.shaders.linear_nobias_f32 import make_linear_nobias_variant
from torch2vk.export.shaders.linear_nobias_quantized import (
    make_linear_nobias_q4_k_m_variant,
    make_linear_nobias_q6_k_variant,
    make_linear_nobias_q8_0_variant,
)
from torch2vk.export.shaders.max_f32 import make_max_variant
from torch2vk.export.shaders.mean_dim_f32 import make_mean_dim_variant
from torch2vk.export.shaders.mul_f32 import make_mul_variant
from torch2vk.export.shaders.neg_f32 import make_neg_variant
from torch2vk.export.shaders.pow_scalar_f32 import make_pow_scalar_variant
from torch2vk.export.shaders.permute_f32 import make_permute_variant
from torch2vk.export.shaders.qwen3_asr_token_select_f32 import (
    QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
)
from torch2vk.export.shaders.qwen3_asr_token_store_f32 import (
    QWEN3_ASR_TOKEN_STORE_EOS_F32,
    QWEN3_ASR_TOKEN_STORE_F32,
)
from torch2vk.export.shaders.reciprocal_f32 import make_reciprocal_variant
from torch2vk.export.shaders.rsqrt_f32 import make_rsqrt_variant
from torch2vk.export.shaders.sdpa_f32 import make_sdpa_variant
from torch2vk.export.shaders.select_int import make_select_variant
from torch2vk.export.shaders.silu_f32 import make_silu_variant
from torch2vk.export.shaders.sin_f32 import make_sin_variant
from torch2vk.export.shaders.slice_f32 import make_slice_variant
from torch2vk.export.shaders.sub_f32 import make_sub_variant
from torch2vk.export.shaders.transpose_f32 import make_transpose_variant

__all__ = [
    "make_add_variant",
    "make_argmax_variant",
    "make_cat_variant",
    "make_conv1d_q8_0_variant",
    "make_conv1d_variant",
    "make_conv_transpose1d_q8_0_variant",
    "make_conv_transpose1d_variant",
    "make_conv2d_q8_0_variant",
    "make_conv2d_variant",
    "make_embedding_q4_k_m_variant",
    "make_embedding_q8_0_variant",
    "make_embedding_variant",
    "make_gelu_variant",
    "make_index_copy_variant",
    "make_index_select_variant",
    "make_layer_norm_variant",
    "make_linear_bias_q8_0_variant",
    "make_linear_bias_variant",
    "make_linear_nobias_variant",
    "make_linear_nobias_q4_k_m_variant",
    "make_linear_nobias_q6_k_variant",
    "make_linear_nobias_q8_0_variant",
    "make_max_variant",
    "make_mean_dim_variant",
    "make_mul_variant",
    "make_neg_variant",
    "make_pow_scalar_variant",
    "make_permute_variant",
    "QWEN3_ASR_TOKEN_SELECT_GREEDY_F32",
    "QWEN3_ASR_TOKEN_STORE_EOS_F32",
    "QWEN3_ASR_TOKEN_STORE_F32",
    "make_reciprocal_variant",
    "make_rsqrt_variant",
    "make_sdpa_variant",
    "make_select_variant",
    "make_silu_variant",
    "make_sin_variant",
    "make_slice_variant",
    "make_sub_variant",
    "make_transpose_variant",
]
