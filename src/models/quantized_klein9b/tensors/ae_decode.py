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
class AeDecodeTensors:
    p_decoder_post_quant_conv_weight: LogicalTensor
    p_decoder_post_quant_conv_bias: LogicalTensor
    p_decoder_conv_in_weight: LogicalTensor
    p_decoder_conv_in_bias: LogicalTensor
    p_decoder_mid_block_1_norm1_weight: LogicalTensor
    p_decoder_mid_block_1_norm1_bias: LogicalTensor
    p_decoder_mid_block_1_conv1_weight: LogicalTensor
    p_decoder_mid_block_1_conv1_bias: LogicalTensor
    p_decoder_mid_block_1_norm2_weight: LogicalTensor
    p_decoder_mid_block_1_norm2_bias: LogicalTensor
    p_decoder_mid_block_1_conv2_weight: LogicalTensor
    p_decoder_mid_block_1_conv2_bias: LogicalTensor
    p_decoder_mid_attn_1_norm_weight: LogicalTensor
    p_decoder_mid_attn_1_norm_bias: LogicalTensor
    p_decoder_mid_attn_1_q_weight: LogicalTensor
    p_decoder_mid_attn_1_q_bias: LogicalTensor
    p_decoder_mid_attn_1_k_weight: LogicalTensor
    p_decoder_mid_attn_1_k_bias: LogicalTensor
    p_decoder_mid_attn_1_v_weight: LogicalTensor
    p_decoder_mid_attn_1_v_bias: LogicalTensor
    p_decoder_mid_attn_1_proj_out_weight: LogicalTensor
    p_decoder_mid_attn_1_proj_out_bias: LogicalTensor
    p_decoder_mid_block_2_norm1_weight: LogicalTensor
    p_decoder_mid_block_2_norm1_bias: LogicalTensor
    p_decoder_mid_block_2_conv1_weight: LogicalTensor
    p_decoder_mid_block_2_conv1_bias: LogicalTensor
    p_decoder_mid_block_2_norm2_weight: LogicalTensor
    p_decoder_mid_block_2_norm2_bias: LogicalTensor
    p_decoder_mid_block_2_conv2_weight: LogicalTensor
    p_decoder_mid_block_2_conv2_bias: LogicalTensor
    p_decoder_up_0_block_0_norm1_weight: LogicalTensor
    p_decoder_up_0_block_0_norm1_bias: LogicalTensor
    p_decoder_up_0_block_0_conv1_weight: LogicalTensor
    p_decoder_up_0_block_0_conv1_bias: LogicalTensor
    p_decoder_up_0_block_0_norm2_weight: LogicalTensor
    p_decoder_up_0_block_0_norm2_bias: LogicalTensor
    p_decoder_up_0_block_0_conv2_weight: LogicalTensor
    p_decoder_up_0_block_0_conv2_bias: LogicalTensor
    p_decoder_up_0_block_0_nin_shortcut_weight: LogicalTensor
    p_decoder_up_0_block_0_nin_shortcut_bias: LogicalTensor
    p_decoder_up_0_block_1_norm1_weight: LogicalTensor
    p_decoder_up_0_block_1_norm1_bias: LogicalTensor
    p_decoder_up_0_block_1_conv1_weight: LogicalTensor
    p_decoder_up_0_block_1_conv1_bias: LogicalTensor
    p_decoder_up_0_block_1_norm2_weight: LogicalTensor
    p_decoder_up_0_block_1_norm2_bias: LogicalTensor
    p_decoder_up_0_block_1_conv2_weight: LogicalTensor
    p_decoder_up_0_block_1_conv2_bias: LogicalTensor
    p_decoder_up_0_block_2_norm1_weight: LogicalTensor
    p_decoder_up_0_block_2_norm1_bias: LogicalTensor
    p_decoder_up_0_block_2_conv1_weight: LogicalTensor
    p_decoder_up_0_block_2_conv1_bias: LogicalTensor
    p_decoder_up_0_block_2_norm2_weight: LogicalTensor
    p_decoder_up_0_block_2_norm2_bias: LogicalTensor
    p_decoder_up_0_block_2_conv2_weight: LogicalTensor
    p_decoder_up_0_block_2_conv2_bias: LogicalTensor
    p_decoder_up_1_block_0_norm1_weight: LogicalTensor
    p_decoder_up_1_block_0_norm1_bias: LogicalTensor
    p_decoder_up_1_block_0_conv1_weight: LogicalTensor
    p_decoder_up_1_block_0_conv1_bias: LogicalTensor
    p_decoder_up_1_block_0_norm2_weight: LogicalTensor
    p_decoder_up_1_block_0_norm2_bias: LogicalTensor
    p_decoder_up_1_block_0_conv2_weight: LogicalTensor
    p_decoder_up_1_block_0_conv2_bias: LogicalTensor
    p_decoder_up_1_block_0_nin_shortcut_weight: LogicalTensor
    p_decoder_up_1_block_0_nin_shortcut_bias: LogicalTensor
    p_decoder_up_1_block_1_norm1_weight: LogicalTensor
    p_decoder_up_1_block_1_norm1_bias: LogicalTensor
    p_decoder_up_1_block_1_conv1_weight: LogicalTensor
    p_decoder_up_1_block_1_conv1_bias: LogicalTensor
    p_decoder_up_1_block_1_norm2_weight: LogicalTensor
    p_decoder_up_1_block_1_norm2_bias: LogicalTensor
    p_decoder_up_1_block_1_conv2_weight: LogicalTensor
    p_decoder_up_1_block_1_conv2_bias: LogicalTensor
    p_decoder_up_1_block_2_norm1_weight: LogicalTensor
    p_decoder_up_1_block_2_norm1_bias: LogicalTensor
    p_decoder_up_1_block_2_conv1_weight: LogicalTensor
    p_decoder_up_1_block_2_conv1_bias: LogicalTensor
    p_decoder_up_1_block_2_norm2_weight: LogicalTensor
    p_decoder_up_1_block_2_norm2_bias: LogicalTensor
    p_decoder_up_1_block_2_conv2_weight: LogicalTensor
    p_decoder_up_1_block_2_conv2_bias: LogicalTensor
    p_decoder_up_1_upsample_conv_weight: LogicalTensor
    p_decoder_up_1_upsample_conv_bias: LogicalTensor
    p_decoder_up_2_block_0_norm1_weight: LogicalTensor
    p_decoder_up_2_block_0_norm1_bias: LogicalTensor
    p_decoder_up_2_block_0_conv1_weight: LogicalTensor
    p_decoder_up_2_block_0_conv1_bias: LogicalTensor
    p_decoder_up_2_block_0_norm2_weight: LogicalTensor
    p_decoder_up_2_block_0_norm2_bias: LogicalTensor
    p_decoder_up_2_block_0_conv2_weight: LogicalTensor
    p_decoder_up_2_block_0_conv2_bias: LogicalTensor
    p_decoder_up_2_block_1_norm1_weight: LogicalTensor
    p_decoder_up_2_block_1_norm1_bias: LogicalTensor
    p_decoder_up_2_block_1_conv1_weight: LogicalTensor
    p_decoder_up_2_block_1_conv1_bias: LogicalTensor
    p_decoder_up_2_block_1_norm2_weight: LogicalTensor
    p_decoder_up_2_block_1_norm2_bias: LogicalTensor
    p_decoder_up_2_block_1_conv2_weight: LogicalTensor
    p_decoder_up_2_block_1_conv2_bias: LogicalTensor
    p_decoder_up_2_block_2_norm1_weight: LogicalTensor
    p_decoder_up_2_block_2_norm1_bias: LogicalTensor
    p_decoder_up_2_block_2_conv1_weight: LogicalTensor
    p_decoder_up_2_block_2_conv1_bias: LogicalTensor
    p_decoder_up_2_block_2_norm2_weight: LogicalTensor
    p_decoder_up_2_block_2_norm2_bias: LogicalTensor
    p_decoder_up_2_block_2_conv2_weight: LogicalTensor
    p_decoder_up_2_block_2_conv2_bias: LogicalTensor
    p_decoder_up_2_upsample_conv_weight: LogicalTensor
    p_decoder_up_2_upsample_conv_bias: LogicalTensor
    p_decoder_up_3_block_0_norm1_weight: LogicalTensor
    p_decoder_up_3_block_0_norm1_bias: LogicalTensor
    p_decoder_up_3_block_0_conv1_weight: LogicalTensor
    p_decoder_up_3_block_0_conv1_bias: LogicalTensor
    p_decoder_up_3_block_0_norm2_weight: LogicalTensor
    p_decoder_up_3_block_0_norm2_bias: LogicalTensor
    p_decoder_up_3_block_0_conv2_weight: LogicalTensor
    p_decoder_up_3_block_0_conv2_bias: LogicalTensor
    p_decoder_up_3_block_1_norm1_weight: LogicalTensor
    p_decoder_up_3_block_1_norm1_bias: LogicalTensor
    p_decoder_up_3_block_1_conv1_weight: LogicalTensor
    p_decoder_up_3_block_1_conv1_bias: LogicalTensor
    p_decoder_up_3_block_1_norm2_weight: LogicalTensor
    p_decoder_up_3_block_1_norm2_bias: LogicalTensor
    p_decoder_up_3_block_1_conv2_weight: LogicalTensor
    p_decoder_up_3_block_1_conv2_bias: LogicalTensor
    p_decoder_up_3_block_2_norm1_weight: LogicalTensor
    p_decoder_up_3_block_2_norm1_bias: LogicalTensor
    p_decoder_up_3_block_2_conv1_weight: LogicalTensor
    p_decoder_up_3_block_2_conv1_bias: LogicalTensor
    p_decoder_up_3_block_2_norm2_weight: LogicalTensor
    p_decoder_up_3_block_2_norm2_bias: LogicalTensor
    p_decoder_up_3_block_2_conv2_weight: LogicalTensor
    p_decoder_up_3_block_2_conv2_bias: LogicalTensor
    p_decoder_up_3_upsample_conv_weight: LogicalTensor
    p_decoder_up_3_upsample_conv_bias: LogicalTensor
    p_decoder_norm_out_weight: LogicalTensor
    p_decoder_norm_out_bias: LogicalTensor
    p_decoder_conv_out_weight: LogicalTensor
    p_decoder_conv_out_bias: LogicalTensor
    b_bn_running_mean: LogicalTensor
    b_bn_running_var: LogicalTensor
    tokens: LogicalTensor
    reshape: LogicalTensor
    permute: LogicalTensor
    contiguous: LogicalTensor
    view: LogicalTensor
    add: LogicalTensor
    sqrt: LogicalTensor
    view_1: LogicalTensor
    mul: LogicalTensor
    add_1: LogicalTensor
    reshape_1: LogicalTensor
    permute_1: LogicalTensor
    reshape_2: LogicalTensor
    conv2d: LogicalTensor
    conv2d_1: LogicalTensor
    group_norm: LogicalTensor
    sigmoid: LogicalTensor
    mul_1: LogicalTensor
    conv2d_2: LogicalTensor
    group_norm_1: LogicalTensor
    sigmoid_1: LogicalTensor
    mul_2: LogicalTensor
    conv2d_3: LogicalTensor
    add_2: LogicalTensor
    group_norm_2: LogicalTensor
    conv2d_4: LogicalTensor
    conv2d_5: LogicalTensor
    conv2d_6: LogicalTensor
    permute_2: LogicalTensor
    reshape_3: LogicalTensor
    contiguous_1: LogicalTensor
    permute_3: LogicalTensor
    reshape_4: LogicalTensor
    contiguous_2: LogicalTensor
    permute_4: LogicalTensor
    reshape_5: LogicalTensor
    contiguous_3: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    reshape_6: LogicalTensor
    permute_5: LogicalTensor
    conv2d_7: LogicalTensor
    add_3: LogicalTensor
    group_norm_3: LogicalTensor
    sigmoid_2: LogicalTensor
    mul_3: LogicalTensor
    conv2d_8: LogicalTensor
    group_norm_4: LogicalTensor
    sigmoid_3: LogicalTensor
    mul_4: LogicalTensor
    conv2d_9: LogicalTensor
    add_4: LogicalTensor
    to: LogicalTensor
    group_norm_5: LogicalTensor
    sigmoid_4: LogicalTensor
    mul_5: LogicalTensor
    conv2d_10: LogicalTensor
    group_norm_6: LogicalTensor
    sigmoid_5: LogicalTensor
    mul_6: LogicalTensor
    conv2d_11: LogicalTensor
    add_5: LogicalTensor
    group_norm_7: LogicalTensor
    sigmoid_6: LogicalTensor
    mul_7: LogicalTensor
    conv2d_12: LogicalTensor
    group_norm_8: LogicalTensor
    sigmoid_7: LogicalTensor
    mul_8: LogicalTensor
    conv2d_13: LogicalTensor
    add_6: LogicalTensor
    group_norm_9: LogicalTensor
    sigmoid_8: LogicalTensor
    mul_9: LogicalTensor
    conv2d_14: LogicalTensor
    group_norm_10: LogicalTensor
    sigmoid_9: LogicalTensor
    mul_10: LogicalTensor
    conv2d_15: LogicalTensor
    add_7: LogicalTensor
    upsample_nearest2d: LogicalTensor
    conv2d_16: LogicalTensor
    group_norm_11: LogicalTensor
    sigmoid_10: LogicalTensor
    mul_11: LogicalTensor
    conv2d_17: LogicalTensor
    group_norm_12: LogicalTensor
    sigmoid_11: LogicalTensor
    mul_12: LogicalTensor
    conv2d_18: LogicalTensor
    add_8: LogicalTensor
    group_norm_13: LogicalTensor
    sigmoid_12: LogicalTensor
    mul_13: LogicalTensor
    conv2d_19: LogicalTensor
    group_norm_14: LogicalTensor
    sigmoid_13: LogicalTensor
    mul_14: LogicalTensor
    conv2d_20: LogicalTensor
    add_9: LogicalTensor
    group_norm_15: LogicalTensor
    sigmoid_14: LogicalTensor
    mul_15: LogicalTensor
    conv2d_21: LogicalTensor
    group_norm_16: LogicalTensor
    sigmoid_15: LogicalTensor
    mul_16: LogicalTensor
    conv2d_22: LogicalTensor
    add_10: LogicalTensor
    upsample_nearest2d_1: LogicalTensor
    conv2d_23: LogicalTensor
    group_norm_17: LogicalTensor
    sigmoid_16: LogicalTensor
    mul_17: LogicalTensor
    conv2d_24: LogicalTensor
    group_norm_18: LogicalTensor
    sigmoid_17: LogicalTensor
    mul_18: LogicalTensor
    conv2d_25: LogicalTensor
    conv2d_26: LogicalTensor
    add_11: LogicalTensor
    group_norm_19: LogicalTensor
    sigmoid_18: LogicalTensor
    mul_19: LogicalTensor
    conv2d_27: LogicalTensor
    group_norm_20: LogicalTensor
    sigmoid_19: LogicalTensor
    mul_20: LogicalTensor
    conv2d_28: LogicalTensor
    add_12: LogicalTensor
    group_norm_21: LogicalTensor
    sigmoid_20: LogicalTensor
    mul_21: LogicalTensor
    conv2d_29: LogicalTensor
    group_norm_22: LogicalTensor
    sigmoid_21: LogicalTensor
    mul_22: LogicalTensor
    conv2d_30: LogicalTensor
    add_13: LogicalTensor
    upsample_nearest2d_2: LogicalTensor
    conv2d_31: LogicalTensor
    group_norm_23: LogicalTensor
    sigmoid_22: LogicalTensor
    mul_23: LogicalTensor
    conv2d_32: LogicalTensor
    group_norm_24: LogicalTensor
    sigmoid_23: LogicalTensor
    mul_24: LogicalTensor
    conv2d_33: LogicalTensor
    conv2d_34: LogicalTensor
    add_14: LogicalTensor
    group_norm_25: LogicalTensor
    sigmoid_24: LogicalTensor
    mul_25: LogicalTensor
    conv2d_35: LogicalTensor
    group_norm_26: LogicalTensor
    sigmoid_25: LogicalTensor
    mul_26: LogicalTensor
    conv2d_36: LogicalTensor
    add_15: LogicalTensor
    group_norm_27: LogicalTensor
    sigmoid_26: LogicalTensor
    mul_27: LogicalTensor
    conv2d_37: LogicalTensor
    group_norm_28: LogicalTensor
    sigmoid_27: LogicalTensor
    mul_28: LogicalTensor
    conv2d_38: LogicalTensor
    add_16: LogicalTensor
    group_norm_29: LogicalTensor
    sigmoid_28: LogicalTensor
    mul_29: LogicalTensor
    conv2d_39: LogicalTensor


AE_DECODE_OUTPUT: str = 'conv2d_39'


def create_ae_decode(
    prefix: str,
    *,
    p_decoder_post_quant_conv_weight: LogicalTensor | None = None,
    p_decoder_post_quant_conv_bias: LogicalTensor | None = None,
    p_decoder_conv_in_weight: LogicalTensor | None = None,
    p_decoder_conv_in_bias: LogicalTensor | None = None,
    p_decoder_mid_block_1_norm1_weight: LogicalTensor | None = None,
    p_decoder_mid_block_1_norm1_bias: LogicalTensor | None = None,
    p_decoder_mid_block_1_conv1_weight: LogicalTensor | None = None,
    p_decoder_mid_block_1_conv1_bias: LogicalTensor | None = None,
    p_decoder_mid_block_1_norm2_weight: LogicalTensor | None = None,
    p_decoder_mid_block_1_norm2_bias: LogicalTensor | None = None,
    p_decoder_mid_block_1_conv2_weight: LogicalTensor | None = None,
    p_decoder_mid_block_1_conv2_bias: LogicalTensor | None = None,
    p_decoder_mid_attn_1_norm_weight: LogicalTensor | None = None,
    p_decoder_mid_attn_1_norm_bias: LogicalTensor | None = None,
    p_decoder_mid_attn_1_q_weight: LogicalTensor | None = None,
    p_decoder_mid_attn_1_q_bias: LogicalTensor | None = None,
    p_decoder_mid_attn_1_k_weight: LogicalTensor | None = None,
    p_decoder_mid_attn_1_k_bias: LogicalTensor | None = None,
    p_decoder_mid_attn_1_v_weight: LogicalTensor | None = None,
    p_decoder_mid_attn_1_v_bias: LogicalTensor | None = None,
    p_decoder_mid_attn_1_proj_out_weight: LogicalTensor | None = None,
    p_decoder_mid_attn_1_proj_out_bias: LogicalTensor | None = None,
    p_decoder_mid_block_2_norm1_weight: LogicalTensor | None = None,
    p_decoder_mid_block_2_norm1_bias: LogicalTensor | None = None,
    p_decoder_mid_block_2_conv1_weight: LogicalTensor | None = None,
    p_decoder_mid_block_2_conv1_bias: LogicalTensor | None = None,
    p_decoder_mid_block_2_norm2_weight: LogicalTensor | None = None,
    p_decoder_mid_block_2_norm2_bias: LogicalTensor | None = None,
    p_decoder_mid_block_2_conv2_weight: LogicalTensor | None = None,
    p_decoder_mid_block_2_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_0_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_0_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_0_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_0_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_0_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_0_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_0_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_0_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_0_nin_shortcut_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_0_nin_shortcut_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_1_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_1_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_1_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_1_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_1_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_1_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_1_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_1_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_2_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_2_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_2_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_2_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_2_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_2_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_0_block_2_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_0_block_2_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_0_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_0_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_0_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_0_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_0_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_0_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_0_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_0_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_0_nin_shortcut_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_0_nin_shortcut_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_1_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_1_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_1_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_1_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_1_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_1_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_1_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_1_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_2_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_2_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_2_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_2_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_2_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_2_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_1_block_2_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_1_block_2_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_1_upsample_conv_weight: LogicalTensor | None = None,
    p_decoder_up_1_upsample_conv_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_0_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_0_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_0_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_0_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_0_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_0_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_0_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_0_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_1_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_1_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_1_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_1_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_1_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_1_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_1_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_1_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_2_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_2_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_2_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_2_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_2_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_2_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_2_block_2_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_2_block_2_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_2_upsample_conv_weight: LogicalTensor | None = None,
    p_decoder_up_2_upsample_conv_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_0_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_0_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_0_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_0_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_0_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_0_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_0_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_0_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_1_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_1_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_1_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_1_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_1_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_1_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_1_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_1_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_2_norm1_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_2_norm1_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_2_conv1_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_2_conv1_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_2_norm2_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_2_norm2_bias: LogicalTensor | None = None,
    p_decoder_up_3_block_2_conv2_weight: LogicalTensor | None = None,
    p_decoder_up_3_block_2_conv2_bias: LogicalTensor | None = None,
    p_decoder_up_3_upsample_conv_weight: LogicalTensor | None = None,
    p_decoder_up_3_upsample_conv_bias: LogicalTensor | None = None,
    p_decoder_norm_out_weight: LogicalTensor | None = None,
    p_decoder_norm_out_bias: LogicalTensor | None = None,
    p_decoder_conv_out_weight: LogicalTensor | None = None,
    p_decoder_conv_out_bias: LogicalTensor | None = None,
    b_bn_running_mean: LogicalTensor | None = None,
    b_bn_running_var: LogicalTensor | None = None,
    tokens: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    permute: LogicalTensor | None = None,
    contiguous: LogicalTensor | None = None,
    view: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    sqrt: LogicalTensor | None = None,
    view_1: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    reshape_1: LogicalTensor | None = None,
    permute_1: LogicalTensor | None = None,
    reshape_2: LogicalTensor | None = None,
    conv2d: LogicalTensor | None = None,
    conv2d_1: LogicalTensor | None = None,
    group_norm: LogicalTensor | None = None,
    sigmoid: LogicalTensor | None = None,
    mul_1: LogicalTensor | None = None,
    conv2d_2: LogicalTensor | None = None,
    group_norm_1: LogicalTensor | None = None,
    sigmoid_1: LogicalTensor | None = None,
    mul_2: LogicalTensor | None = None,
    conv2d_3: LogicalTensor | None = None,
    add_2: LogicalTensor | None = None,
    group_norm_2: LogicalTensor | None = None,
    conv2d_4: LogicalTensor | None = None,
    conv2d_5: LogicalTensor | None = None,
    conv2d_6: LogicalTensor | None = None,
    permute_2: LogicalTensor | None = None,
    reshape_3: LogicalTensor | None = None,
    contiguous_1: LogicalTensor | None = None,
    permute_3: LogicalTensor | None = None,
    reshape_4: LogicalTensor | None = None,
    contiguous_2: LogicalTensor | None = None,
    permute_4: LogicalTensor | None = None,
    reshape_5: LogicalTensor | None = None,
    contiguous_3: LogicalTensor | None = None,
    scaled_dot_product_attention: LogicalTensor | None = None,
    reshape_6: LogicalTensor | None = None,
    permute_5: LogicalTensor | None = None,
    conv2d_7: LogicalTensor | None = None,
    add_3: LogicalTensor | None = None,
    group_norm_3: LogicalTensor | None = None,
    sigmoid_2: LogicalTensor | None = None,
    mul_3: LogicalTensor | None = None,
    conv2d_8: LogicalTensor | None = None,
    group_norm_4: LogicalTensor | None = None,
    sigmoid_3: LogicalTensor | None = None,
    mul_4: LogicalTensor | None = None,
    conv2d_9: LogicalTensor | None = None,
    add_4: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    group_norm_5: LogicalTensor | None = None,
    sigmoid_4: LogicalTensor | None = None,
    mul_5: LogicalTensor | None = None,
    conv2d_10: LogicalTensor | None = None,
    group_norm_6: LogicalTensor | None = None,
    sigmoid_5: LogicalTensor | None = None,
    mul_6: LogicalTensor | None = None,
    conv2d_11: LogicalTensor | None = None,
    add_5: LogicalTensor | None = None,
    group_norm_7: LogicalTensor | None = None,
    sigmoid_6: LogicalTensor | None = None,
    mul_7: LogicalTensor | None = None,
    conv2d_12: LogicalTensor | None = None,
    group_norm_8: LogicalTensor | None = None,
    sigmoid_7: LogicalTensor | None = None,
    mul_8: LogicalTensor | None = None,
    conv2d_13: LogicalTensor | None = None,
    add_6: LogicalTensor | None = None,
    group_norm_9: LogicalTensor | None = None,
    sigmoid_8: LogicalTensor | None = None,
    mul_9: LogicalTensor | None = None,
    conv2d_14: LogicalTensor | None = None,
    group_norm_10: LogicalTensor | None = None,
    sigmoid_9: LogicalTensor | None = None,
    mul_10: LogicalTensor | None = None,
    conv2d_15: LogicalTensor | None = None,
    add_7: LogicalTensor | None = None,
    upsample_nearest2d: LogicalTensor | None = None,
    conv2d_16: LogicalTensor | None = None,
    group_norm_11: LogicalTensor | None = None,
    sigmoid_10: LogicalTensor | None = None,
    mul_11: LogicalTensor | None = None,
    conv2d_17: LogicalTensor | None = None,
    group_norm_12: LogicalTensor | None = None,
    sigmoid_11: LogicalTensor | None = None,
    mul_12: LogicalTensor | None = None,
    conv2d_18: LogicalTensor | None = None,
    add_8: LogicalTensor | None = None,
    group_norm_13: LogicalTensor | None = None,
    sigmoid_12: LogicalTensor | None = None,
    mul_13: LogicalTensor | None = None,
    conv2d_19: LogicalTensor | None = None,
    group_norm_14: LogicalTensor | None = None,
    sigmoid_13: LogicalTensor | None = None,
    mul_14: LogicalTensor | None = None,
    conv2d_20: LogicalTensor | None = None,
    add_9: LogicalTensor | None = None,
    group_norm_15: LogicalTensor | None = None,
    sigmoid_14: LogicalTensor | None = None,
    mul_15: LogicalTensor | None = None,
    conv2d_21: LogicalTensor | None = None,
    group_norm_16: LogicalTensor | None = None,
    sigmoid_15: LogicalTensor | None = None,
    mul_16: LogicalTensor | None = None,
    conv2d_22: LogicalTensor | None = None,
    add_10: LogicalTensor | None = None,
    upsample_nearest2d_1: LogicalTensor | None = None,
    conv2d_23: LogicalTensor | None = None,
    group_norm_17: LogicalTensor | None = None,
    sigmoid_16: LogicalTensor | None = None,
    mul_17: LogicalTensor | None = None,
    conv2d_24: LogicalTensor | None = None,
    group_norm_18: LogicalTensor | None = None,
    sigmoid_17: LogicalTensor | None = None,
    mul_18: LogicalTensor | None = None,
    conv2d_25: LogicalTensor | None = None,
    conv2d_26: LogicalTensor | None = None,
    add_11: LogicalTensor | None = None,
    group_norm_19: LogicalTensor | None = None,
    sigmoid_18: LogicalTensor | None = None,
    mul_19: LogicalTensor | None = None,
    conv2d_27: LogicalTensor | None = None,
    group_norm_20: LogicalTensor | None = None,
    sigmoid_19: LogicalTensor | None = None,
    mul_20: LogicalTensor | None = None,
    conv2d_28: LogicalTensor | None = None,
    add_12: LogicalTensor | None = None,
    group_norm_21: LogicalTensor | None = None,
    sigmoid_20: LogicalTensor | None = None,
    mul_21: LogicalTensor | None = None,
    conv2d_29: LogicalTensor | None = None,
    group_norm_22: LogicalTensor | None = None,
    sigmoid_21: LogicalTensor | None = None,
    mul_22: LogicalTensor | None = None,
    conv2d_30: LogicalTensor | None = None,
    add_13: LogicalTensor | None = None,
    upsample_nearest2d_2: LogicalTensor | None = None,
    conv2d_31: LogicalTensor | None = None,
    group_norm_23: LogicalTensor | None = None,
    sigmoid_22: LogicalTensor | None = None,
    mul_23: LogicalTensor | None = None,
    conv2d_32: LogicalTensor | None = None,
    group_norm_24: LogicalTensor | None = None,
    sigmoid_23: LogicalTensor | None = None,
    mul_24: LogicalTensor | None = None,
    conv2d_33: LogicalTensor | None = None,
    conv2d_34: LogicalTensor | None = None,
    add_14: LogicalTensor | None = None,
    group_norm_25: LogicalTensor | None = None,
    sigmoid_24: LogicalTensor | None = None,
    mul_25: LogicalTensor | None = None,
    conv2d_35: LogicalTensor | None = None,
    group_norm_26: LogicalTensor | None = None,
    sigmoid_25: LogicalTensor | None = None,
    mul_26: LogicalTensor | None = None,
    conv2d_36: LogicalTensor | None = None,
    add_15: LogicalTensor | None = None,
    group_norm_27: LogicalTensor | None = None,
    sigmoid_26: LogicalTensor | None = None,
    mul_27: LogicalTensor | None = None,
    conv2d_37: LogicalTensor | None = None,
    group_norm_28: LogicalTensor | None = None,
    sigmoid_27: LogicalTensor | None = None,
    mul_28: LogicalTensor | None = None,
    conv2d_38: LogicalTensor | None = None,
    add_16: LogicalTensor | None = None,
    group_norm_29: LogicalTensor | None = None,
    sigmoid_28: LogicalTensor | None = None,
    mul_29: LogicalTensor | None = None,
    conv2d_39: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> AeDecodeTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('conv2d_39',)))
    tensors = AeDecodeTensors(
        p_decoder_post_quant_conv_weight=_bind_tensor(
            p_decoder_post_quant_conv_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.post_quant_conv.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.post_quant_conv.weight", dtype='float32', shape=(32, 32, 1, 1)),
                layout=_quantized_weight_layout("decoder.post_quant_conv.weight", dtype='float32', shape=(32, 32, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_post_quant_conv_weight' in request_state_outputs,
            ),
        ),
        p_decoder_post_quant_conv_bias=_bind_tensor(
            p_decoder_post_quant_conv_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.post_quant_conv.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.post_quant_conv.bias", dtype='float32', shape=(32,)),
                layout=_quantized_weight_layout("decoder.post_quant_conv.bias", dtype='float32', shape=(32,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_post_quant_conv_bias' in request_state_outputs,
            ),
        ),
        p_decoder_conv_in_weight=_bind_tensor(
            p_decoder_conv_in_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.conv_in.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.conv_in.weight", dtype='float32', shape=(512, 32, 3, 3)),
                layout=_quantized_weight_layout("decoder.conv_in.weight", dtype='float32', shape=(512, 32, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_conv_in_weight' in request_state_outputs,
            ),
        ),
        p_decoder_conv_in_bias=_bind_tensor(
            p_decoder_conv_in_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.conv_in.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.conv_in.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.conv_in.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_conv_in_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_norm1_weight=_bind_tensor(
            p_decoder_mid_block_1_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_1.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_norm1_bias=_bind_tensor(
            p_decoder_mid_block_1_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_1.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_conv1_weight=_bind_tensor(
            p_decoder_mid_block_1_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.mid.block_1.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_conv1_bias=_bind_tensor(
            p_decoder_mid_block_1_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_1.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_norm2_weight=_bind_tensor(
            p_decoder_mid_block_1_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_1.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_norm2_bias=_bind_tensor(
            p_decoder_mid_block_1_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_1.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_conv2_weight=_bind_tensor(
            p_decoder_mid_block_1_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.mid.block_1.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_1_conv2_bias=_bind_tensor(
            p_decoder_mid_block_1_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_1.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_1.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_1.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_1_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_norm_weight=_bind_tensor(
            p_decoder_mid_attn_1_norm_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.norm.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.norm.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.norm.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_norm_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_norm_bias=_bind_tensor(
            p_decoder_mid_attn_1_norm_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.norm.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.norm.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.norm.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_norm_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_q_weight=_bind_tensor(
            p_decoder_mid_attn_1_q_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.q.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.q.weight", dtype='float32', shape=(512, 512, 1, 1)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.q.weight", dtype='float32', shape=(512, 512, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_q_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_q_bias=_bind_tensor(
            p_decoder_mid_attn_1_q_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.q.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.q.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.q.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_q_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_k_weight=_bind_tensor(
            p_decoder_mid_attn_1_k_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.k.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.k.weight", dtype='float32', shape=(512, 512, 1, 1)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.k.weight", dtype='float32', shape=(512, 512, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_k_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_k_bias=_bind_tensor(
            p_decoder_mid_attn_1_k_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.k.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.k.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.k.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_k_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_v_weight=_bind_tensor(
            p_decoder_mid_attn_1_v_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.v.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.v.weight", dtype='float32', shape=(512, 512, 1, 1)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.v.weight", dtype='float32', shape=(512, 512, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_v_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_v_bias=_bind_tensor(
            p_decoder_mid_attn_1_v_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.v.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.v.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.v.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_v_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_proj_out_weight=_bind_tensor(
            p_decoder_mid_attn_1_proj_out_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.proj_out.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.proj_out.weight", dtype='float32', shape=(512, 512, 1, 1)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.proj_out.weight", dtype='float32', shape=(512, 512, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_proj_out_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_attn_1_proj_out_bias=_bind_tensor(
            p_decoder_mid_attn_1_proj_out_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.attn_1.proj_out.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.attn_1.proj_out.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.attn_1.proj_out.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_attn_1_proj_out_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_norm1_weight=_bind_tensor(
            p_decoder_mid_block_2_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_2.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_norm1_bias=_bind_tensor(
            p_decoder_mid_block_2_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_2.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_conv1_weight=_bind_tensor(
            p_decoder_mid_block_2_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.mid.block_2.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_conv1_bias=_bind_tensor(
            p_decoder_mid_block_2_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_2.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_norm2_weight=_bind_tensor(
            p_decoder_mid_block_2_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_2.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_norm2_bias=_bind_tensor(
            p_decoder_mid_block_2_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_2.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_conv2_weight=_bind_tensor(
            p_decoder_mid_block_2_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.mid.block_2.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_mid_block_2_conv2_bias=_bind_tensor(
            p_decoder_mid_block_2_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.mid.block_2.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.mid.block_2.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.mid.block_2.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_mid_block_2_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_norm1_weight=_bind_tensor(
            p_decoder_up_0_block_0_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.norm1.weight", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.norm1.weight", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_norm1_bias=_bind_tensor(
            p_decoder_up_0_block_0_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.norm1.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.norm1.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_conv1_weight=_bind_tensor(
            p_decoder_up_0_block_0_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.conv1.weight", dtype='float32', shape=(128, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.conv1.weight", dtype='float32', shape=(128, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_conv1_bias=_bind_tensor(
            p_decoder_up_0_block_0_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.conv1.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.conv1.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_norm2_weight=_bind_tensor(
            p_decoder_up_0_block_0_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.norm2.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.norm2.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_norm2_bias=_bind_tensor(
            p_decoder_up_0_block_0_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.norm2.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.norm2.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_conv2_weight=_bind_tensor(
            p_decoder_up_0_block_0_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.conv2.weight", dtype='float32', shape=(128, 128, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.conv2.weight", dtype='float32', shape=(128, 128, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_conv2_bias=_bind_tensor(
            p_decoder_up_0_block_0_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.conv2.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.conv2.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_nin_shortcut_weight=_bind_tensor(
            p_decoder_up_0_block_0_nin_shortcut_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.nin_shortcut.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.nin_shortcut.weight", dtype='float32', shape=(128, 256, 1, 1)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.nin_shortcut.weight", dtype='float32', shape=(128, 256, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_nin_shortcut_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_0_nin_shortcut_bias=_bind_tensor(
            p_decoder_up_0_block_0_nin_shortcut_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.0.nin_shortcut.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.0.nin_shortcut.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.0.nin_shortcut.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_0_nin_shortcut_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_norm1_weight=_bind_tensor(
            p_decoder_up_0_block_1_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.norm1.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.norm1.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_norm1_bias=_bind_tensor(
            p_decoder_up_0_block_1_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.norm1.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.norm1.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_conv1_weight=_bind_tensor(
            p_decoder_up_0_block_1_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.conv1.weight", dtype='float32', shape=(128, 128, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.conv1.weight", dtype='float32', shape=(128, 128, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_conv1_bias=_bind_tensor(
            p_decoder_up_0_block_1_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.conv1.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.conv1.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_norm2_weight=_bind_tensor(
            p_decoder_up_0_block_1_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.norm2.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.norm2.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_norm2_bias=_bind_tensor(
            p_decoder_up_0_block_1_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.norm2.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.norm2.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_conv2_weight=_bind_tensor(
            p_decoder_up_0_block_1_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.conv2.weight", dtype='float32', shape=(128, 128, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.conv2.weight", dtype='float32', shape=(128, 128, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_1_conv2_bias=_bind_tensor(
            p_decoder_up_0_block_1_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.1.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.1.conv2.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.1.conv2.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_1_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_norm1_weight=_bind_tensor(
            p_decoder_up_0_block_2_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.norm1.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.norm1.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_norm1_bias=_bind_tensor(
            p_decoder_up_0_block_2_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.norm1.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.norm1.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_conv1_weight=_bind_tensor(
            p_decoder_up_0_block_2_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.conv1.weight", dtype='float32', shape=(128, 128, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.conv1.weight", dtype='float32', shape=(128, 128, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_conv1_bias=_bind_tensor(
            p_decoder_up_0_block_2_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.conv1.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.conv1.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_norm2_weight=_bind_tensor(
            p_decoder_up_0_block_2_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.norm2.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.norm2.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_norm2_bias=_bind_tensor(
            p_decoder_up_0_block_2_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.norm2.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.norm2.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_conv2_weight=_bind_tensor(
            p_decoder_up_0_block_2_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.conv2.weight", dtype='float32', shape=(128, 128, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.conv2.weight", dtype='float32', shape=(128, 128, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_0_block_2_conv2_bias=_bind_tensor(
            p_decoder_up_0_block_2_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.0.block.2.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.0.block.2.conv2.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.up.0.block.2.conv2.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_0_block_2_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_norm1_weight=_bind_tensor(
            p_decoder_up_1_block_0_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_norm1_bias=_bind_tensor(
            p_decoder_up_1_block_0_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_conv1_weight=_bind_tensor(
            p_decoder_up_1_block_0_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.conv1.weight", dtype='float32', shape=(256, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.conv1.weight", dtype='float32', shape=(256, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_conv1_bias=_bind_tensor(
            p_decoder_up_1_block_0_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.conv1.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.conv1.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_norm2_weight=_bind_tensor(
            p_decoder_up_1_block_0_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.norm2.weight", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.norm2.weight", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_norm2_bias=_bind_tensor(
            p_decoder_up_1_block_0_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.norm2.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.norm2.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_conv2_weight=_bind_tensor(
            p_decoder_up_1_block_0_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.conv2.weight", dtype='float32', shape=(256, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.conv2.weight", dtype='float32', shape=(256, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_conv2_bias=_bind_tensor(
            p_decoder_up_1_block_0_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.conv2.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.conv2.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_nin_shortcut_weight=_bind_tensor(
            p_decoder_up_1_block_0_nin_shortcut_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.nin_shortcut.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.nin_shortcut.weight", dtype='float32', shape=(256, 512, 1, 1)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.nin_shortcut.weight", dtype='float32', shape=(256, 512, 1, 1)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_nin_shortcut_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_0_nin_shortcut_bias=_bind_tensor(
            p_decoder_up_1_block_0_nin_shortcut_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.0.nin_shortcut.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.0.nin_shortcut.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.0.nin_shortcut.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_0_nin_shortcut_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_norm1_weight=_bind_tensor(
            p_decoder_up_1_block_1_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.norm1.weight", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.norm1.weight", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_norm1_bias=_bind_tensor(
            p_decoder_up_1_block_1_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.norm1.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.norm1.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_conv1_weight=_bind_tensor(
            p_decoder_up_1_block_1_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.conv1.weight", dtype='float32', shape=(256, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.conv1.weight", dtype='float32', shape=(256, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_conv1_bias=_bind_tensor(
            p_decoder_up_1_block_1_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.conv1.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.conv1.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_norm2_weight=_bind_tensor(
            p_decoder_up_1_block_1_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.norm2.weight", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.norm2.weight", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_norm2_bias=_bind_tensor(
            p_decoder_up_1_block_1_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.norm2.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.norm2.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_conv2_weight=_bind_tensor(
            p_decoder_up_1_block_1_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.conv2.weight", dtype='float32', shape=(256, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.conv2.weight", dtype='float32', shape=(256, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_1_conv2_bias=_bind_tensor(
            p_decoder_up_1_block_1_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.1.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.1.conv2.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.1.conv2.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_1_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_norm1_weight=_bind_tensor(
            p_decoder_up_1_block_2_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.norm1.weight", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.norm1.weight", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_norm1_bias=_bind_tensor(
            p_decoder_up_1_block_2_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.norm1.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.norm1.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_conv1_weight=_bind_tensor(
            p_decoder_up_1_block_2_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.conv1.weight", dtype='float32', shape=(256, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.conv1.weight", dtype='float32', shape=(256, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_conv1_bias=_bind_tensor(
            p_decoder_up_1_block_2_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.conv1.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.conv1.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_norm2_weight=_bind_tensor(
            p_decoder_up_1_block_2_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.norm2.weight", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.norm2.weight", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_norm2_bias=_bind_tensor(
            p_decoder_up_1_block_2_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.norm2.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.norm2.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_conv2_weight=_bind_tensor(
            p_decoder_up_1_block_2_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.conv2.weight", dtype='float32', shape=(256, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.conv2.weight", dtype='float32', shape=(256, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_block_2_conv2_bias=_bind_tensor(
            p_decoder_up_1_block_2_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.block.2.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.block.2.conv2.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.block.2.conv2.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_block_2_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_upsample_conv_weight=_bind_tensor(
            p_decoder_up_1_upsample_conv_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.upsample.conv.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.upsample.conv.weight", dtype='float32', shape=(256, 256, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.1.upsample.conv.weight", dtype='float32', shape=(256, 256, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_upsample_conv_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_1_upsample_conv_bias=_bind_tensor(
            p_decoder_up_1_upsample_conv_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.1.upsample.conv.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.1.upsample.conv.bias", dtype='float32', shape=(256,)),
                layout=_quantized_weight_layout("decoder.up.1.upsample.conv.bias", dtype='float32', shape=(256,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_1_upsample_conv_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_norm1_weight=_bind_tensor(
            p_decoder_up_2_block_0_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_norm1_bias=_bind_tensor(
            p_decoder_up_2_block_0_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_conv1_weight=_bind_tensor(
            p_decoder_up_2_block_0_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_conv1_bias=_bind_tensor(
            p_decoder_up_2_block_0_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_norm2_weight=_bind_tensor(
            p_decoder_up_2_block_0_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_norm2_bias=_bind_tensor(
            p_decoder_up_2_block_0_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_conv2_weight=_bind_tensor(
            p_decoder_up_2_block_0_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_0_conv2_bias=_bind_tensor(
            p_decoder_up_2_block_0_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.0.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.0.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.0.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_0_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_norm1_weight=_bind_tensor(
            p_decoder_up_2_block_1_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_norm1_bias=_bind_tensor(
            p_decoder_up_2_block_1_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_conv1_weight=_bind_tensor(
            p_decoder_up_2_block_1_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_conv1_bias=_bind_tensor(
            p_decoder_up_2_block_1_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_norm2_weight=_bind_tensor(
            p_decoder_up_2_block_1_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_norm2_bias=_bind_tensor(
            p_decoder_up_2_block_1_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_conv2_weight=_bind_tensor(
            p_decoder_up_2_block_1_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_1_conv2_bias=_bind_tensor(
            p_decoder_up_2_block_1_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.1.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.1.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.1.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_1_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_norm1_weight=_bind_tensor(
            p_decoder_up_2_block_2_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_norm1_bias=_bind_tensor(
            p_decoder_up_2_block_2_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_conv1_weight=_bind_tensor(
            p_decoder_up_2_block_2_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_conv1_bias=_bind_tensor(
            p_decoder_up_2_block_2_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_norm2_weight=_bind_tensor(
            p_decoder_up_2_block_2_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_norm2_bias=_bind_tensor(
            p_decoder_up_2_block_2_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_conv2_weight=_bind_tensor(
            p_decoder_up_2_block_2_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_block_2_conv2_bias=_bind_tensor(
            p_decoder_up_2_block_2_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.block.2.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.block.2.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.block.2.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_block_2_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_upsample_conv_weight=_bind_tensor(
            p_decoder_up_2_upsample_conv_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.upsample.conv.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.upsample.conv.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.2.upsample.conv.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_upsample_conv_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_2_upsample_conv_bias=_bind_tensor(
            p_decoder_up_2_upsample_conv_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.2.upsample.conv.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.2.upsample.conv.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.2.upsample.conv.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_2_upsample_conv_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_norm1_weight=_bind_tensor(
            p_decoder_up_3_block_0_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_norm1_bias=_bind_tensor(
            p_decoder_up_3_block_0_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_conv1_weight=_bind_tensor(
            p_decoder_up_3_block_0_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_conv1_bias=_bind_tensor(
            p_decoder_up_3_block_0_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_norm2_weight=_bind_tensor(
            p_decoder_up_3_block_0_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_norm2_bias=_bind_tensor(
            p_decoder_up_3_block_0_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_conv2_weight=_bind_tensor(
            p_decoder_up_3_block_0_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_0_conv2_bias=_bind_tensor(
            p_decoder_up_3_block_0_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.0.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.0.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.0.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_0_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_norm1_weight=_bind_tensor(
            p_decoder_up_3_block_1_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_norm1_bias=_bind_tensor(
            p_decoder_up_3_block_1_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_conv1_weight=_bind_tensor(
            p_decoder_up_3_block_1_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_conv1_bias=_bind_tensor(
            p_decoder_up_3_block_1_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_norm2_weight=_bind_tensor(
            p_decoder_up_3_block_1_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_norm2_bias=_bind_tensor(
            p_decoder_up_3_block_1_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_conv2_weight=_bind_tensor(
            p_decoder_up_3_block_1_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_1_conv2_bias=_bind_tensor(
            p_decoder_up_3_block_1_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.1.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.1.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.1.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_1_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_norm1_weight=_bind_tensor(
            p_decoder_up_3_block_2_norm1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.norm1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.norm1.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.norm1.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_norm1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_norm1_bias=_bind_tensor(
            p_decoder_up_3_block_2_norm1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.norm1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.norm1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.norm1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_norm1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_conv1_weight=_bind_tensor(
            p_decoder_up_3_block_2_conv1_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.conv1.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.conv1.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_conv1_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_conv1_bias=_bind_tensor(
            p_decoder_up_3_block_2_conv1_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.conv1.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.conv1.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.conv1.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_conv1_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_norm2_weight=_bind_tensor(
            p_decoder_up_3_block_2_norm2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.norm2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.norm2.weight", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.norm2.weight", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_norm2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_norm2_bias=_bind_tensor(
            p_decoder_up_3_block_2_norm2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.norm2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.norm2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.norm2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_norm2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_conv2_weight=_bind_tensor(
            p_decoder_up_3_block_2_conv2_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.conv2.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.conv2.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_conv2_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_block_2_conv2_bias=_bind_tensor(
            p_decoder_up_3_block_2_conv2_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.block.2.conv2.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.block.2.conv2.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.block.2.conv2.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_block_2_conv2_bias' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_upsample_conv_weight=_bind_tensor(
            p_decoder_up_3_upsample_conv_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.upsample.conv.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.upsample.conv.weight", dtype='float32', shape=(512, 512, 3, 3)),
                layout=_quantized_weight_layout("decoder.up.3.upsample.conv.weight", dtype='float32', shape=(512, 512, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_upsample_conv_weight' in request_state_outputs,
            ),
        ),
        p_decoder_up_3_upsample_conv_bias=_bind_tensor(
            p_decoder_up_3_upsample_conv_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.up.3.upsample.conv.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.up.3.upsample.conv.bias", dtype='float32', shape=(512,)),
                layout=_quantized_weight_layout("decoder.up.3.upsample.conv.bias", dtype='float32', shape=(512,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_up_3_upsample_conv_bias' in request_state_outputs,
            ),
        ),
        p_decoder_norm_out_weight=_bind_tensor(
            p_decoder_norm_out_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.norm_out.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.norm_out.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.norm_out.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_norm_out_weight' in request_state_outputs,
            ),
        ),
        p_decoder_norm_out_bias=_bind_tensor(
            p_decoder_norm_out_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.norm_out.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.norm_out.bias", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("decoder.norm_out.bias", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_norm_out_bias' in request_state_outputs,
            ),
        ),
        p_decoder_conv_out_weight=_bind_tensor(
            p_decoder_conv_out_weight,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.conv_out.weight",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.conv_out.weight", dtype='float32', shape=(3, 128, 3, 3)),
                layout=_quantized_weight_layout("decoder.conv_out.weight", dtype='float32', shape=(3, 128, 3, 3)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_conv_out_weight' in request_state_outputs,
            ),
        ),
        p_decoder_conv_out_bias=_bind_tensor(
            p_decoder_conv_out_bias,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="decoder.conv_out.bias",
                reference_key=None,
                spec=_quantized_weight_spec("decoder.conv_out.bias", dtype='float32', shape=(3,)),
                layout=_quantized_weight_layout("decoder.conv_out.bias", dtype='float32', shape=(3,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_decoder_conv_out_bias' in request_state_outputs,
            ),
        ),
        b_bn_running_mean=_bind_tensor(
            b_bn_running_mean,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="bn.running_mean",
                reference_key=None,
                spec=_quantized_weight_spec("bn.running_mean", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("bn.running_mean", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='b_bn_running_mean' in request_state_outputs,
            ),
        ),
        b_bn_running_var=_bind_tensor(
            b_bn_running_var,
            _declare_tensor(
                checkpoint='ae/model.gguf',
                checkpoint_key="bn.running_var",
                reference_key=None,
                spec=_quantized_weight_spec("bn.running_var", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout("bn.running_var", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='b_bn_running_var' in request_state_outputs,
            ),
        ),
        tokens=_bind_tensor(
            tokens,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(1, 1024, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='tokens' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape',
                spec=TensorSpec(dtype='float16', shape=(1, 32, 32, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape' in request_state_outputs,
            ),
        ),
        permute=_bind_tensor(
            permute,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 32, 32)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute' in request_state_outputs,
            ),
        ),
        contiguous=_bind_tensor(
            contiguous,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='contiguous',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 32, 32)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous' in request_state_outputs,
            ),
        ),
        view=_bind_tensor(
            view,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view',
                spec=TensorSpec(dtype='float32', shape=(1, 128, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        sqrt=_bind_tensor(
            sqrt,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sqrt',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sqrt' in request_state_outputs,
            ),
        ),
        view_1=_bind_tensor(
            view_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view_1',
                spec=TensorSpec(dtype='float32', shape=(1, 128, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view_1' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 32, 32)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_1',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 32, 32)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            reshape_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_1',
                spec=TensorSpec(dtype='float16', shape=(1, 32, 2, 2, 32, 32)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_1' in request_state_outputs,
            ),
        ),
        permute_1=_bind_tensor(
            permute_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_1',
                spec=TensorSpec(dtype='float16', shape=(1, 32, 32, 2, 32, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_1' in request_state_outputs,
            ),
        ),
        reshape_2=_bind_tensor(
            reshape_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_2',
                spec=TensorSpec(dtype='float16', shape=(1, 32, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_2' in request_state_outputs,
            ),
        ),
        conv2d=_bind_tensor(
            conv2d,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d',
                spec=TensorSpec(dtype='float16', shape=(1, 32, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d' in request_state_outputs,
            ),
        ),
        conv2d_1=_bind_tensor(
            conv2d_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_1',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_1' in request_state_outputs,
            ),
        ),
        group_norm=_bind_tensor(
            group_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm' in request_state_outputs,
            ),
        ),
        sigmoid=_bind_tensor(
            sigmoid,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_1',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_1' in request_state_outputs,
            ),
        ),
        conv2d_2=_bind_tensor(
            conv2d_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_2',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_2' in request_state_outputs,
            ),
        ),
        group_norm_1=_bind_tensor(
            group_norm_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_1',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_1' in request_state_outputs,
            ),
        ),
        sigmoid_1=_bind_tensor(
            sigmoid_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_1',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_1' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_2',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_2' in request_state_outputs,
            ),
        ),
        conv2d_3=_bind_tensor(
            conv2d_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_3',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_3' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_2',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_2' in request_state_outputs,
            ),
        ),
        group_norm_2=_bind_tensor(
            group_norm_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_2',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_2' in request_state_outputs,
            ),
        ),
        conv2d_4=_bind_tensor(
            conv2d_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_4',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_4' in request_state_outputs,
            ),
        ),
        conv2d_5=_bind_tensor(
            conv2d_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_5',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_5' in request_state_outputs,
            ),
        ),
        conv2d_6=_bind_tensor(
            conv2d_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_6',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_6' in request_state_outputs,
            ),
        ),
        permute_2=_bind_tensor(
            permute_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_2',
                spec=TensorSpec(dtype='float16', shape=(1, 64, 64, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_2' in request_state_outputs,
            ),
        ),
        reshape_3=_bind_tensor(
            reshape_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_3',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_3' in request_state_outputs,
            ),
        ),
        contiguous_1=_bind_tensor(
            contiguous_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='contiguous_1',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous_1' in request_state_outputs,
            ),
        ),
        permute_3=_bind_tensor(
            permute_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_3',
                spec=TensorSpec(dtype='float16', shape=(1, 64, 64, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_3' in request_state_outputs,
            ),
        ),
        reshape_4=_bind_tensor(
            reshape_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_4',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_4' in request_state_outputs,
            ),
        ),
        contiguous_2=_bind_tensor(
            contiguous_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='contiguous_2',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous_2' in request_state_outputs,
            ),
        ),
        permute_4=_bind_tensor(
            permute_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_4',
                spec=TensorSpec(dtype='float16', shape=(1, 64, 64, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_4' in request_state_outputs,
            ),
        ),
        reshape_5=_bind_tensor(
            reshape_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_5',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_5' in request_state_outputs,
            ),
        ),
        contiguous_3=_bind_tensor(
            contiguous_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='contiguous_3',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous_3' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                spec=TensorSpec(dtype='float16', shape=(1, 1, 4096, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='scaled_dot_product_attention' in request_state_outputs,
            ),
        ),
        reshape_6=_bind_tensor(
            reshape_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_6',
                spec=TensorSpec(dtype='float16', shape=(1, 64, 64, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_6' in request_state_outputs,
            ),
        ),
        permute_5=_bind_tensor(
            permute_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_5',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_5' in request_state_outputs,
            ),
        ),
        conv2d_7=_bind_tensor(
            conv2d_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_7',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_7' in request_state_outputs,
            ),
        ),
        add_3=_bind_tensor(
            add_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_3',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_3' in request_state_outputs,
            ),
        ),
        group_norm_3=_bind_tensor(
            group_norm_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_3',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_3' in request_state_outputs,
            ),
        ),
        sigmoid_2=_bind_tensor(
            sigmoid_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_2',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_2' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_3',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_3' in request_state_outputs,
            ),
        ),
        conv2d_8=_bind_tensor(
            conv2d_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_8',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_8' in request_state_outputs,
            ),
        ),
        group_norm_4=_bind_tensor(
            group_norm_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_4',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_4' in request_state_outputs,
            ),
        ),
        sigmoid_3=_bind_tensor(
            sigmoid_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_3',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_3' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_4',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_4' in request_state_outputs,
            ),
        ),
        conv2d_9=_bind_tensor(
            conv2d_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_9',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_9' in request_state_outputs,
            ),
        ),
        add_4=_bind_tensor(
            add_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_4',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_4' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to' in request_state_outputs,
            ),
        ),
        group_norm_5=_bind_tensor(
            group_norm_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_5',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_5' in request_state_outputs,
            ),
        ),
        sigmoid_4=_bind_tensor(
            sigmoid_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_4',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_4' in request_state_outputs,
            ),
        ),
        mul_5=_bind_tensor(
            mul_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_5',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_5' in request_state_outputs,
            ),
        ),
        conv2d_10=_bind_tensor(
            conv2d_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_10',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_10' in request_state_outputs,
            ),
        ),
        group_norm_6=_bind_tensor(
            group_norm_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_6',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_6' in request_state_outputs,
            ),
        ),
        sigmoid_5=_bind_tensor(
            sigmoid_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_5',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_5' in request_state_outputs,
            ),
        ),
        mul_6=_bind_tensor(
            mul_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_6',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_6' in request_state_outputs,
            ),
        ),
        conv2d_11=_bind_tensor(
            conv2d_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_11',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_11' in request_state_outputs,
            ),
        ),
        add_5=_bind_tensor(
            add_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_5',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_5' in request_state_outputs,
            ),
        ),
        group_norm_7=_bind_tensor(
            group_norm_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_7',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_7' in request_state_outputs,
            ),
        ),
        sigmoid_6=_bind_tensor(
            sigmoid_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_6',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_6' in request_state_outputs,
            ),
        ),
        mul_7=_bind_tensor(
            mul_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_7',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_7' in request_state_outputs,
            ),
        ),
        conv2d_12=_bind_tensor(
            conv2d_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_12',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_12' in request_state_outputs,
            ),
        ),
        group_norm_8=_bind_tensor(
            group_norm_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_8',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_8' in request_state_outputs,
            ),
        ),
        sigmoid_7=_bind_tensor(
            sigmoid_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_7',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_7' in request_state_outputs,
            ),
        ),
        mul_8=_bind_tensor(
            mul_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_8',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_8' in request_state_outputs,
            ),
        ),
        conv2d_13=_bind_tensor(
            conv2d_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_13',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_13' in request_state_outputs,
            ),
        ),
        add_6=_bind_tensor(
            add_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_6',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_6' in request_state_outputs,
            ),
        ),
        group_norm_9=_bind_tensor(
            group_norm_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_9',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_9' in request_state_outputs,
            ),
        ),
        sigmoid_8=_bind_tensor(
            sigmoid_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_8',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_8' in request_state_outputs,
            ),
        ),
        mul_9=_bind_tensor(
            mul_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_9',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_9' in request_state_outputs,
            ),
        ),
        conv2d_14=_bind_tensor(
            conv2d_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_14',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_14' in request_state_outputs,
            ),
        ),
        group_norm_10=_bind_tensor(
            group_norm_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_10',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_10' in request_state_outputs,
            ),
        ),
        sigmoid_9=_bind_tensor(
            sigmoid_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_9',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_9' in request_state_outputs,
            ),
        ),
        mul_10=_bind_tensor(
            mul_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_10',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_10' in request_state_outputs,
            ),
        ),
        conv2d_15=_bind_tensor(
            conv2d_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_15',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_15' in request_state_outputs,
            ),
        ),
        add_7=_bind_tensor(
            add_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_7',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 64, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_7' in request_state_outputs,
            ),
        ),
        upsample_nearest2d=_bind_tensor(
            upsample_nearest2d,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='upsample_nearest2d',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='upsample_nearest2d' in request_state_outputs,
            ),
        ),
        conv2d_16=_bind_tensor(
            conv2d_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_16',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_16' in request_state_outputs,
            ),
        ),
        group_norm_11=_bind_tensor(
            group_norm_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_11',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_11' in request_state_outputs,
            ),
        ),
        sigmoid_10=_bind_tensor(
            sigmoid_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_10',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_10' in request_state_outputs,
            ),
        ),
        mul_11=_bind_tensor(
            mul_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_11',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_11' in request_state_outputs,
            ),
        ),
        conv2d_17=_bind_tensor(
            conv2d_17,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_17',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_17' in request_state_outputs,
            ),
        ),
        group_norm_12=_bind_tensor(
            group_norm_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_12',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_12' in request_state_outputs,
            ),
        ),
        sigmoid_11=_bind_tensor(
            sigmoid_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_11',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_11' in request_state_outputs,
            ),
        ),
        mul_12=_bind_tensor(
            mul_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_12',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_12' in request_state_outputs,
            ),
        ),
        conv2d_18=_bind_tensor(
            conv2d_18,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_18',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_18' in request_state_outputs,
            ),
        ),
        add_8=_bind_tensor(
            add_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_8',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_8' in request_state_outputs,
            ),
        ),
        group_norm_13=_bind_tensor(
            group_norm_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_13',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_13' in request_state_outputs,
            ),
        ),
        sigmoid_12=_bind_tensor(
            sigmoid_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_12',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_12' in request_state_outputs,
            ),
        ),
        mul_13=_bind_tensor(
            mul_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_13',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_13' in request_state_outputs,
            ),
        ),
        conv2d_19=_bind_tensor(
            conv2d_19,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_19',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_19' in request_state_outputs,
            ),
        ),
        group_norm_14=_bind_tensor(
            group_norm_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_14',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_14' in request_state_outputs,
            ),
        ),
        sigmoid_13=_bind_tensor(
            sigmoid_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_13',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_13' in request_state_outputs,
            ),
        ),
        mul_14=_bind_tensor(
            mul_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_14',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_14' in request_state_outputs,
            ),
        ),
        conv2d_20=_bind_tensor(
            conv2d_20,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_20',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_20' in request_state_outputs,
            ),
        ),
        add_9=_bind_tensor(
            add_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_9',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_9' in request_state_outputs,
            ),
        ),
        group_norm_15=_bind_tensor(
            group_norm_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_15',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_15' in request_state_outputs,
            ),
        ),
        sigmoid_14=_bind_tensor(
            sigmoid_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_14',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_14' in request_state_outputs,
            ),
        ),
        mul_15=_bind_tensor(
            mul_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_15',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_15' in request_state_outputs,
            ),
        ),
        conv2d_21=_bind_tensor(
            conv2d_21,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_21',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_21' in request_state_outputs,
            ),
        ),
        group_norm_16=_bind_tensor(
            group_norm_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_16',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_16' in request_state_outputs,
            ),
        ),
        sigmoid_15=_bind_tensor(
            sigmoid_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_15',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_15' in request_state_outputs,
            ),
        ),
        mul_16=_bind_tensor(
            mul_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_16',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_16' in request_state_outputs,
            ),
        ),
        conv2d_22=_bind_tensor(
            conv2d_22,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_22',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_22' in request_state_outputs,
            ),
        ),
        add_10=_bind_tensor(
            add_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_10',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 128, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_10' in request_state_outputs,
            ),
        ),
        upsample_nearest2d_1=_bind_tensor(
            upsample_nearest2d_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='upsample_nearest2d_1',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='upsample_nearest2d_1' in request_state_outputs,
            ),
        ),
        conv2d_23=_bind_tensor(
            conv2d_23,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_23',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_23' in request_state_outputs,
            ),
        ),
        group_norm_17=_bind_tensor(
            group_norm_17,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_17',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_17' in request_state_outputs,
            ),
        ),
        sigmoid_16=_bind_tensor(
            sigmoid_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_16',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_16' in request_state_outputs,
            ),
        ),
        mul_17=_bind_tensor(
            mul_17,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_17',
                spec=TensorSpec(dtype='float16', shape=(1, 512, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_17' in request_state_outputs,
            ),
        ),
        conv2d_24=_bind_tensor(
            conv2d_24,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_24',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_24' in request_state_outputs,
            ),
        ),
        group_norm_18=_bind_tensor(
            group_norm_18,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_18',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_18' in request_state_outputs,
            ),
        ),
        sigmoid_17=_bind_tensor(
            sigmoid_17,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_17',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_17' in request_state_outputs,
            ),
        ),
        mul_18=_bind_tensor(
            mul_18,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_18',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_18' in request_state_outputs,
            ),
        ),
        conv2d_25=_bind_tensor(
            conv2d_25,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_25',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_25' in request_state_outputs,
            ),
        ),
        conv2d_26=_bind_tensor(
            conv2d_26,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_26',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_26' in request_state_outputs,
            ),
        ),
        add_11=_bind_tensor(
            add_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_11',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_11' in request_state_outputs,
            ),
        ),
        group_norm_19=_bind_tensor(
            group_norm_19,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_19',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_19' in request_state_outputs,
            ),
        ),
        sigmoid_18=_bind_tensor(
            sigmoid_18,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_18',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_18' in request_state_outputs,
            ),
        ),
        mul_19=_bind_tensor(
            mul_19,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_19',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_19' in request_state_outputs,
            ),
        ),
        conv2d_27=_bind_tensor(
            conv2d_27,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_27',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_27' in request_state_outputs,
            ),
        ),
        group_norm_20=_bind_tensor(
            group_norm_20,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_20',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_20' in request_state_outputs,
            ),
        ),
        sigmoid_19=_bind_tensor(
            sigmoid_19,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_19',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_19' in request_state_outputs,
            ),
        ),
        mul_20=_bind_tensor(
            mul_20,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_20',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_20' in request_state_outputs,
            ),
        ),
        conv2d_28=_bind_tensor(
            conv2d_28,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_28',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_28' in request_state_outputs,
            ),
        ),
        add_12=_bind_tensor(
            add_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_12',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_12' in request_state_outputs,
            ),
        ),
        group_norm_21=_bind_tensor(
            group_norm_21,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_21',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_21' in request_state_outputs,
            ),
        ),
        sigmoid_20=_bind_tensor(
            sigmoid_20,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_20',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_20' in request_state_outputs,
            ),
        ),
        mul_21=_bind_tensor(
            mul_21,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_21',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_21' in request_state_outputs,
            ),
        ),
        conv2d_29=_bind_tensor(
            conv2d_29,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_29',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_29' in request_state_outputs,
            ),
        ),
        group_norm_22=_bind_tensor(
            group_norm_22,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_22',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_22' in request_state_outputs,
            ),
        ),
        sigmoid_21=_bind_tensor(
            sigmoid_21,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_21',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_21' in request_state_outputs,
            ),
        ),
        mul_22=_bind_tensor(
            mul_22,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_22',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_22' in request_state_outputs,
            ),
        ),
        conv2d_30=_bind_tensor(
            conv2d_30,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_30',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_30' in request_state_outputs,
            ),
        ),
        add_13=_bind_tensor(
            add_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_13',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 256, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_13' in request_state_outputs,
            ),
        ),
        upsample_nearest2d_2=_bind_tensor(
            upsample_nearest2d_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='upsample_nearest2d_2',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='upsample_nearest2d_2' in request_state_outputs,
            ),
        ),
        conv2d_31=_bind_tensor(
            conv2d_31,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_31',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_31' in request_state_outputs,
            ),
        ),
        group_norm_23=_bind_tensor(
            group_norm_23,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_23',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_23' in request_state_outputs,
            ),
        ),
        sigmoid_22=_bind_tensor(
            sigmoid_22,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_22',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_22' in request_state_outputs,
            ),
        ),
        mul_23=_bind_tensor(
            mul_23,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_23',
                spec=TensorSpec(dtype='float16', shape=(1, 256, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_23' in request_state_outputs,
            ),
        ),
        conv2d_32=_bind_tensor(
            conv2d_32,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_32',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_32' in request_state_outputs,
            ),
        ),
        group_norm_24=_bind_tensor(
            group_norm_24,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_24',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_24' in request_state_outputs,
            ),
        ),
        sigmoid_23=_bind_tensor(
            sigmoid_23,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_23',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_23' in request_state_outputs,
            ),
        ),
        mul_24=_bind_tensor(
            mul_24,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_24',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_24' in request_state_outputs,
            ),
        ),
        conv2d_33=_bind_tensor(
            conv2d_33,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_33',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_33' in request_state_outputs,
            ),
        ),
        conv2d_34=_bind_tensor(
            conv2d_34,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_34',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_34' in request_state_outputs,
            ),
        ),
        add_14=_bind_tensor(
            add_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_14',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_14' in request_state_outputs,
            ),
        ),
        group_norm_25=_bind_tensor(
            group_norm_25,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_25',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_25' in request_state_outputs,
            ),
        ),
        sigmoid_24=_bind_tensor(
            sigmoid_24,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_24',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_24' in request_state_outputs,
            ),
        ),
        mul_25=_bind_tensor(
            mul_25,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_25',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_25' in request_state_outputs,
            ),
        ),
        conv2d_35=_bind_tensor(
            conv2d_35,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_35',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_35' in request_state_outputs,
            ),
        ),
        group_norm_26=_bind_tensor(
            group_norm_26,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_26',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_26' in request_state_outputs,
            ),
        ),
        sigmoid_25=_bind_tensor(
            sigmoid_25,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_25',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_25' in request_state_outputs,
            ),
        ),
        mul_26=_bind_tensor(
            mul_26,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_26',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_26' in request_state_outputs,
            ),
        ),
        conv2d_36=_bind_tensor(
            conv2d_36,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_36',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_36' in request_state_outputs,
            ),
        ),
        add_15=_bind_tensor(
            add_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_15',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_15' in request_state_outputs,
            ),
        ),
        group_norm_27=_bind_tensor(
            group_norm_27,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_27',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_27' in request_state_outputs,
            ),
        ),
        sigmoid_26=_bind_tensor(
            sigmoid_26,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_26',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_26' in request_state_outputs,
            ),
        ),
        mul_27=_bind_tensor(
            mul_27,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_27',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_27' in request_state_outputs,
            ),
        ),
        conv2d_37=_bind_tensor(
            conv2d_37,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_37',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_37' in request_state_outputs,
            ),
        ),
        group_norm_28=_bind_tensor(
            group_norm_28,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_28',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_28' in request_state_outputs,
            ),
        ),
        sigmoid_27=_bind_tensor(
            sigmoid_27,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_27',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_27' in request_state_outputs,
            ),
        ),
        mul_28=_bind_tensor(
            mul_28,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_28',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_28' in request_state_outputs,
            ),
        ),
        conv2d_38=_bind_tensor(
            conv2d_38,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_38',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_38' in request_state_outputs,
            ),
        ),
        add_16=_bind_tensor(
            add_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_16',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_16' in request_state_outputs,
            ),
        ),
        group_norm_29=_bind_tensor(
            group_norm_29,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='group_norm_29',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='group_norm_29' in request_state_outputs,
            ),
        ),
        sigmoid_28=_bind_tensor(
            sigmoid_28,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sigmoid_28',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sigmoid_28' in request_state_outputs,
            ),
        ),
        mul_29=_bind_tensor(
            mul_29,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_29',
                spec=TensorSpec(dtype='float16', shape=(1, 128, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_29' in request_state_outputs,
            ),
        ),
        conv2d_39=_bind_tensor(
            conv2d_39,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='conv2d_39',
                spec=TensorSpec(dtype='float16', shape=(1, 3, 512, 512)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_39' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.tokens, tensors.reshape)
    _bind_alias_source(tensors.permute, tensors.contiguous)
    _bind_alias_source(tensors.b_bn_running_var, tensors.view)
    _bind_alias_source(tensors.b_bn_running_mean, tensors.view_1)
    _bind_alias_source(tensors.add_1, tensors.reshape_1)
    _bind_alias_source(tensors.permute_1, tensors.reshape_2)
    _bind_alias_source(tensors.permute_2, tensors.reshape_3)
    _bind_alias_source(tensors.reshape_3, tensors.contiguous_1)
    _bind_alias_source(tensors.permute_3, tensors.reshape_4)
    _bind_alias_source(tensors.reshape_4, tensors.contiguous_2)
    _bind_alias_source(tensors.permute_4, tensors.reshape_5)
    _bind_alias_source(tensors.reshape_5, tensors.contiguous_3)
    _bind_alias_source(tensors.scaled_dot_product_attention, tensors.reshape_6)
    _bind_alias_source(tensors.add_4, tensors.to)
    return tensors


_Q6_TENSOR_NAMES = frozenset(())
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(())
_Q8_TENSOR_PREFIXES = ('',)


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
