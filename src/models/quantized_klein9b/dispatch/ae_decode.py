"""Generated dispatch function for run_ae_decode."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_f32 import ADD_F32
from models.quantized_klein9b.shaders.ae_decode_add_broadcast_inner import AE_DECODE_ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.ae_decode_add_scalar import AE_DECODE_ADD_SCALAR
from models.quantized_klein9b.shaders.ae_decode_mul_broadcast import AE_DECODE_MUL_BROADCAST
from models.quantized_klein9b.shaders.ae_decode_mul_f32 import AE_DECODE_MUL_F32
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16 import CONV2D_Q8_0W_F32B_F16
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_11 import CONV2D_Q8_0W_F32B_F16_11
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_12 import CONV2D_Q8_0W_F32B_F16_12
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_14 import CONV2D_Q8_0W_F32B_F16_14
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_15 import CONV2D_Q8_0W_F32B_F16_15
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_16 import CONV2D_Q8_0W_F32B_F16_16
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_20 import CONV2D_Q8_0W_F32B_F16_20
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_21 import CONV2D_Q8_0W_F32B_F16_21
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_22 import CONV2D_Q8_0W_F32B_F16_22
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_23 import CONV2D_Q8_0W_F32B_F16_23
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_24 import CONV2D_Q8_0W_F32B_F16_24
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_25 import CONV2D_Q8_0W_F32B_F16_25
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_26 import CONV2D_Q8_0W_F32B_F16_26
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_27 import CONV2D_Q8_0W_F32B_F16_27
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_28 import CONV2D_Q8_0W_F32B_F16_28
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_30 import CONV2D_Q8_0W_F32B_F16_30
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_31 import CONV2D_Q8_0W_F32B_F16_31
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_32 import CONV2D_Q8_0W_F32B_F16_32
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_33 import CONV2D_Q8_0W_F32B_F16_33
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_34 import CONV2D_Q8_0W_F32B_F16_34
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_35 import CONV2D_Q8_0W_F32B_F16_35
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_36 import CONV2D_Q8_0W_F32B_F16_36
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_37 import CONV2D_Q8_0W_F32B_F16_37
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_38 import CONV2D_Q8_0W_F32B_F16_38
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_39 import CONV2D_Q8_0W_F32B_F16_39
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_40 import CONV2D_Q8_0W_F32B_F16_40
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_41 import CONV2D_Q8_0W_F32B_F16_41
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_42 import CONV2D_Q8_0W_F32B_F16_42
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_43 import CONV2D_Q8_0W_F32B_F16_43
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_44 import CONV2D_Q8_0W_F32B_F16_44
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_45 import CONV2D_Q8_0W_F32B_F16_45
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_46 import CONV2D_Q8_0W_F32B_F16_46
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_47 import CONV2D_Q8_0W_F32B_F16_47
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_48 import CONV2D_Q8_0W_F32B_F16_48
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_49 import CONV2D_Q8_0W_F32B_F16_49
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_50 import CONV2D_Q8_0W_F32B_F16_50
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_51 import CONV2D_Q8_0W_F32B_F16_51
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_52 import CONV2D_Q8_0W_F32B_F16_52
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_53 import CONV2D_Q8_0W_F32B_F16_53
from models.quantized_klein9b.shaders.conv2d_q8_0w_f32b_f16_7 import CONV2D_Q8_0W_F32B_F16_7
from models.quantized_klein9b.shaders.group_norm_f32w_f32b_f32 import GROUP_NORM_F32W_F32B_F32
from models.quantized_klein9b.shaders.permute_f32_3ccd7bc8af import PERMUTE_F32_3CCD7BC8AF
from models.quantized_klein9b.shaders.permute_f32_43fee4ac2b import PERMUTE_F32_43FEE4AC2B
from models.quantized_klein9b.shaders.permute_f32_671cd9afc1 import PERMUTE_F32_671CD9AFC1
from models.quantized_klein9b.shaders.permute_f32_79954a5639 import PERMUTE_F32_79954A5639
from models.quantized_klein9b.shaders.sdpa_wide_f16 import SDPA_WIDE_F16
from models.quantized_klein9b.shaders.sigmoid_f32 import SIGMOID_F32
from models.quantized_klein9b.shaders.sqrt_f32 import SQRT_F32
from models.quantized_klein9b.shaders.upsample_nearest2d_f32 import UPSAMPLE_NEAREST2D_F32
from models.quantized_klein9b.tensors.ae_decode import AeDecodeTensors
from torch2vk.runtime.session import RuntimeSession


def _run_ae_decode_with_tensors(rt: RuntimeSession, tensors: AeDecodeTensors) -> None:
    PERMUTE_F32_43FEE4AC2B(rt, x=tensors.tokens, output=tensors.permute)
    AE_DECODE_ADD_SCALAR(rt, x=tensors.view, output=tensors.add)
    SQRT_F32(rt, x=tensors.add, output=tensors.sqrt)
    AE_DECODE_MUL_BROADCAST(rt, x=tensors.contiguous, y=tensors.sqrt, output=tensors.mul)
    AE_DECODE_ADD_BROADCAST_INNER(rt, x=tensors.mul, y=tensors.view_1, output=tensors.add_1)
    PERMUTE_F32_3CCD7BC8AF(rt, x=tensors.reshape, output=tensors.permute_1)
    CONV2D_Q8_0W_F32B_F16(rt, x=tensors.reshape_1, weight=tensors.p_decoder_post_quant_conv_weight, bias=tensors.p_decoder_post_quant_conv_bias, output=tensors.conv2d)
    CONV2D_Q8_0W_F32B_F16_7(rt, x=tensors.conv2d, weight=tensors.p_decoder_conv_in_weight, bias=tensors.p_decoder_conv_in_bias, output=tensors.conv2d_1)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_1, weight=tensors.p_decoder_mid_block_1_norm1_weight, bias=tensors.p_decoder_mid_block_1_norm1_bias, output=tensors.group_norm)
    SIGMOID_F32(rt, x=tensors.group_norm, output=tensors.sigmoid)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm, y=tensors.sigmoid, output=tensors.mul_5)
    CONV2D_Q8_0W_F32B_F16_11(rt, x=tensors.mul_5, weight=tensors.p_decoder_mid_block_1_conv1_weight, bias=tensors.p_decoder_mid_block_1_conv1_bias, output=tensors.conv2d_2)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_2, weight=tensors.p_decoder_mid_block_1_norm2_weight, bias=tensors.p_decoder_mid_block_1_norm2_bias, output=tensors.group_norm_1)
    SIGMOID_F32(rt, x=tensors.group_norm_1, output=tensors.sigmoid_1)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_1, y=tensors.sigmoid_1, output=tensors.mul_8)
    CONV2D_Q8_0W_F32B_F16_12(rt, x=tensors.mul_8, weight=tensors.p_decoder_mid_block_1_conv2_weight, bias=tensors.p_decoder_mid_block_1_conv2_bias, output=tensors.conv2d_3)
    ADD_F32(rt, x=tensors.conv2d_1, y=tensors.conv2d_3, output=tensors.add_2)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_2, weight=tensors.p_decoder_mid_attn_1_norm_weight, bias=tensors.p_decoder_mid_attn_1_norm_bias, output=tensors.group_norm_2)
    CONV2D_Q8_0W_F32B_F16_14(rt, x=tensors.group_norm_2, weight=tensors.p_decoder_mid_attn_1_q_weight, bias=tensors.p_decoder_mid_attn_1_q_bias, output=tensors.conv2d_4)
    CONV2D_Q8_0W_F32B_F16_15(rt, x=tensors.group_norm_2, weight=tensors.p_decoder_mid_attn_1_k_weight, bias=tensors.p_decoder_mid_attn_1_k_bias, output=tensors.conv2d_5)
    CONV2D_Q8_0W_F32B_F16_16(rt, x=tensors.group_norm_2, weight=tensors.p_decoder_mid_attn_1_v_weight, bias=tensors.p_decoder_mid_attn_1_v_bias, output=tensors.conv2d_6)
    PERMUTE_F32_671CD9AFC1(rt, x=tensors.conv2d_4, output=tensors.permute_2)
    PERMUTE_F32_671CD9AFC1(rt, x=tensors.conv2d_5, output=tensors.permute_3)
    PERMUTE_F32_671CD9AFC1(rt, x=tensors.conv2d_6, output=tensors.permute_4)
    SDPA_WIDE_F16(rt, q=tensors.contiguous_1, k=tensors.contiguous_2, v=tensors.contiguous_3, output=tensors.scaled_dot_product_attention)
    PERMUTE_F32_79954A5639(rt, x=tensors.reshape_5, output=tensors.permute_5)
    CONV2D_Q8_0W_F32B_F16_20(rt, x=tensors.permute_5, weight=tensors.p_decoder_mid_attn_1_proj_out_weight, bias=tensors.p_decoder_mid_attn_1_proj_out_bias, output=tensors.conv2d_7)
    ADD_F32(rt, x=tensors.add_2, y=tensors.conv2d_7, output=tensors.add_3)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_3, weight=tensors.p_decoder_mid_block_2_norm1_weight, bias=tensors.p_decoder_mid_block_2_norm1_bias, output=tensors.group_norm_3)
    SIGMOID_F32(rt, x=tensors.group_norm_3, output=tensors.sigmoid_2)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_3, y=tensors.sigmoid_2, output=tensors.mul_17)
    CONV2D_Q8_0W_F32B_F16_21(rt, x=tensors.mul_17, weight=tensors.p_decoder_mid_block_2_conv1_weight, bias=tensors.p_decoder_mid_block_2_conv1_bias, output=tensors.conv2d_8)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_8, weight=tensors.p_decoder_mid_block_2_norm2_weight, bias=tensors.p_decoder_mid_block_2_norm2_bias, output=tensors.group_norm_4)
    SIGMOID_F32(rt, x=tensors.group_norm_4, output=tensors.sigmoid_3)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_4, y=tensors.sigmoid_3, output=tensors.mul_20)
    CONV2D_Q8_0W_F32B_F16_22(rt, x=tensors.mul_20, weight=tensors.p_decoder_mid_block_2_conv2_weight, bias=tensors.p_decoder_mid_block_2_conv2_bias, output=tensors.conv2d_9)
    ADD_F32(rt, x=tensors.add_3, y=tensors.conv2d_9, output=tensors.add_4)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.to, weight=tensors.p_decoder_up_3_block_0_norm1_weight, bias=tensors.p_decoder_up_3_block_0_norm1_bias, output=tensors.group_norm_5)
    SIGMOID_F32(rt, x=tensors.group_norm_5, output=tensors.sigmoid_4)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_5, y=tensors.sigmoid_4, output=tensors.mul_23)
    CONV2D_Q8_0W_F32B_F16_23(rt, x=tensors.mul_23, weight=tensors.p_decoder_up_3_block_0_conv1_weight, bias=tensors.p_decoder_up_3_block_0_conv1_bias, output=tensors.conv2d_10)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_10, weight=tensors.p_decoder_up_3_block_0_norm2_weight, bias=tensors.p_decoder_up_3_block_0_norm2_bias, output=tensors.group_norm_6)
    SIGMOID_F32(rt, x=tensors.group_norm_6, output=tensors.sigmoid_5)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_6, y=tensors.sigmoid_5, output=tensors.mul_26)
    CONV2D_Q8_0W_F32B_F16_24(rt, x=tensors.mul_26, weight=tensors.p_decoder_up_3_block_0_conv2_weight, bias=tensors.p_decoder_up_3_block_0_conv2_bias, output=tensors.conv2d_11)
    ADD_F32(rt, x=tensors.to, y=tensors.conv2d_11, output=tensors.add_5)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_5, weight=tensors.p_decoder_up_3_block_1_norm1_weight, bias=tensors.p_decoder_up_3_block_1_norm1_bias, output=tensors.group_norm_7)
    SIGMOID_F32(rt, x=tensors.group_norm_7, output=tensors.sigmoid_6)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_7, y=tensors.sigmoid_6, output=tensors.mul_29)
    CONV2D_Q8_0W_F32B_F16_25(rt, x=tensors.mul_29, weight=tensors.p_decoder_up_3_block_1_conv1_weight, bias=tensors.p_decoder_up_3_block_1_conv1_bias, output=tensors.conv2d_12)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_12, weight=tensors.p_decoder_up_3_block_1_norm2_weight, bias=tensors.p_decoder_up_3_block_1_norm2_bias, output=tensors.group_norm_8)
    SIGMOID_F32(rt, x=tensors.group_norm_8, output=tensors.sigmoid_7)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_8, y=tensors.sigmoid_7, output=tensors.mul_32)
    CONV2D_Q8_0W_F32B_F16_26(rt, x=tensors.mul_32, weight=tensors.p_decoder_up_3_block_1_conv2_weight, bias=tensors.p_decoder_up_3_block_1_conv2_bias, output=tensors.conv2d_13)
    ADD_F32(rt, x=tensors.add_5, y=tensors.conv2d_13, output=tensors.add_6)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_6, weight=tensors.p_decoder_up_3_block_2_norm1_weight, bias=tensors.p_decoder_up_3_block_2_norm1_bias, output=tensors.group_norm_9)
    SIGMOID_F32(rt, x=tensors.group_norm_9, output=tensors.sigmoid_8)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_9, y=tensors.sigmoid_8, output=tensors.mul_35)
    CONV2D_Q8_0W_F32B_F16_27(rt, x=tensors.mul_35, weight=tensors.p_decoder_up_3_block_2_conv1_weight, bias=tensors.p_decoder_up_3_block_2_conv1_bias, output=tensors.conv2d_14)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_14, weight=tensors.p_decoder_up_3_block_2_norm2_weight, bias=tensors.p_decoder_up_3_block_2_norm2_bias, output=tensors.group_norm_10)
    SIGMOID_F32(rt, x=tensors.group_norm_10, output=tensors.sigmoid_9)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_10, y=tensors.sigmoid_9, output=tensors.mul_38)
    CONV2D_Q8_0W_F32B_F16_28(rt, x=tensors.mul_38, weight=tensors.p_decoder_up_3_block_2_conv2_weight, bias=tensors.p_decoder_up_3_block_2_conv2_bias, output=tensors.conv2d_15)
    ADD_F32(rt, x=tensors.add_6, y=tensors.conv2d_15, output=tensors.add_7)
    UPSAMPLE_NEAREST2D_F32(rt, x=tensors.add_7, output=tensors.upsample_nearest2d)
    CONV2D_Q8_0W_F32B_F16_30(rt, x=tensors.upsample_nearest2d, weight=tensors.p_decoder_up_3_upsample_conv_weight, bias=tensors.p_decoder_up_3_upsample_conv_bias, output=tensors.conv2d_16)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_16, weight=tensors.p_decoder_up_2_block_0_norm1_weight, bias=tensors.p_decoder_up_2_block_0_norm1_bias, output=tensors.group_norm_11)
    SIGMOID_F32(rt, x=tensors.group_norm_11, output=tensors.sigmoid_10)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_11, y=tensors.sigmoid_10, output=tensors.mul_41)
    CONV2D_Q8_0W_F32B_F16_31(rt, x=tensors.mul_41, weight=tensors.p_decoder_up_2_block_0_conv1_weight, bias=tensors.p_decoder_up_2_block_0_conv1_bias, output=tensors.conv2d_17)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_17, weight=tensors.p_decoder_up_2_block_0_norm2_weight, bias=tensors.p_decoder_up_2_block_0_norm2_bias, output=tensors.group_norm_12)
    SIGMOID_F32(rt, x=tensors.group_norm_12, output=tensors.sigmoid_11)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_12, y=tensors.sigmoid_11, output=tensors.mul_44)
    CONV2D_Q8_0W_F32B_F16_32(rt, x=tensors.mul_44, weight=tensors.p_decoder_up_2_block_0_conv2_weight, bias=tensors.p_decoder_up_2_block_0_conv2_bias, output=tensors.conv2d_18)
    ADD_F32(rt, x=tensors.conv2d_16, y=tensors.conv2d_18, output=tensors.add_8)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_8, weight=tensors.p_decoder_up_2_block_1_norm1_weight, bias=tensors.p_decoder_up_2_block_1_norm1_bias, output=tensors.group_norm_13)
    SIGMOID_F32(rt, x=tensors.group_norm_13, output=tensors.sigmoid_12)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_13, y=tensors.sigmoid_12, output=tensors.mul_47)
    CONV2D_Q8_0W_F32B_F16_33(rt, x=tensors.mul_47, weight=tensors.p_decoder_up_2_block_1_conv1_weight, bias=tensors.p_decoder_up_2_block_1_conv1_bias, output=tensors.conv2d_19)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_19, weight=tensors.p_decoder_up_2_block_1_norm2_weight, bias=tensors.p_decoder_up_2_block_1_norm2_bias, output=tensors.group_norm_14)
    SIGMOID_F32(rt, x=tensors.group_norm_14, output=tensors.sigmoid_13)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_14, y=tensors.sigmoid_13, output=tensors.mul_50)
    CONV2D_Q8_0W_F32B_F16_34(rt, x=tensors.mul_50, weight=tensors.p_decoder_up_2_block_1_conv2_weight, bias=tensors.p_decoder_up_2_block_1_conv2_bias, output=tensors.conv2d_20)
    ADD_F32(rt, x=tensors.add_8, y=tensors.conv2d_20, output=tensors.add_9)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_9, weight=tensors.p_decoder_up_2_block_2_norm1_weight, bias=tensors.p_decoder_up_2_block_2_norm1_bias, output=tensors.group_norm_15)
    SIGMOID_F32(rt, x=tensors.group_norm_15, output=tensors.sigmoid_14)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_15, y=tensors.sigmoid_14, output=tensors.mul_53)
    CONV2D_Q8_0W_F32B_F16_35(rt, x=tensors.mul_53, weight=tensors.p_decoder_up_2_block_2_conv1_weight, bias=tensors.p_decoder_up_2_block_2_conv1_bias, output=tensors.conv2d_21)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_21, weight=tensors.p_decoder_up_2_block_2_norm2_weight, bias=tensors.p_decoder_up_2_block_2_norm2_bias, output=tensors.group_norm_16)
    SIGMOID_F32(rt, x=tensors.group_norm_16, output=tensors.sigmoid_15)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_16, y=tensors.sigmoid_15, output=tensors.mul_56)
    CONV2D_Q8_0W_F32B_F16_36(rt, x=tensors.mul_56, weight=tensors.p_decoder_up_2_block_2_conv2_weight, bias=tensors.p_decoder_up_2_block_2_conv2_bias, output=tensors.conv2d_22)
    ADD_F32(rt, x=tensors.add_9, y=tensors.conv2d_22, output=tensors.add_10)
    UPSAMPLE_NEAREST2D_F32(rt, x=tensors.add_10, output=tensors.upsample_nearest2d_1)
    CONV2D_Q8_0W_F32B_F16_37(rt, x=tensors.upsample_nearest2d_1, weight=tensors.p_decoder_up_2_upsample_conv_weight, bias=tensors.p_decoder_up_2_upsample_conv_bias, output=tensors.conv2d_23)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_23, weight=tensors.p_decoder_up_1_block_0_norm1_weight, bias=tensors.p_decoder_up_1_block_0_norm1_bias, output=tensors.group_norm_17)
    SIGMOID_F32(rt, x=tensors.group_norm_17, output=tensors.sigmoid_16)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_17, y=tensors.sigmoid_16, output=tensors.mul_59)
    CONV2D_Q8_0W_F32B_F16_38(rt, x=tensors.mul_59, weight=tensors.p_decoder_up_1_block_0_conv1_weight, bias=tensors.p_decoder_up_1_block_0_conv1_bias, output=tensors.conv2d_24)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_24, weight=tensors.p_decoder_up_1_block_0_norm2_weight, bias=tensors.p_decoder_up_1_block_0_norm2_bias, output=tensors.group_norm_18)
    SIGMOID_F32(rt, x=tensors.group_norm_18, output=tensors.sigmoid_17)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_18, y=tensors.sigmoid_17, output=tensors.mul_62)
    CONV2D_Q8_0W_F32B_F16_39(rt, x=tensors.mul_62, weight=tensors.p_decoder_up_1_block_0_conv2_weight, bias=tensors.p_decoder_up_1_block_0_conv2_bias, output=tensors.conv2d_25)
    CONV2D_Q8_0W_F32B_F16_40(rt, x=tensors.conv2d_23, weight=tensors.p_decoder_up_1_block_0_nin_shortcut_weight, bias=tensors.p_decoder_up_1_block_0_nin_shortcut_bias, output=tensors.conv2d_26)
    ADD_F32(rt, x=tensors.conv2d_26, y=tensors.conv2d_25, output=tensors.add_11)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_11, weight=tensors.p_decoder_up_1_block_1_norm1_weight, bias=tensors.p_decoder_up_1_block_1_norm1_bias, output=tensors.group_norm_19)
    SIGMOID_F32(rt, x=tensors.group_norm_19, output=tensors.sigmoid_18)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_19, y=tensors.sigmoid_18, output=tensors.mul_65)
    CONV2D_Q8_0W_F32B_F16_41(rt, x=tensors.mul_65, weight=tensors.p_decoder_up_1_block_1_conv1_weight, bias=tensors.p_decoder_up_1_block_1_conv1_bias, output=tensors.conv2d_27)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_27, weight=tensors.p_decoder_up_1_block_1_norm2_weight, bias=tensors.p_decoder_up_1_block_1_norm2_bias, output=tensors.group_norm_20)
    SIGMOID_F32(rt, x=tensors.group_norm_20, output=tensors.sigmoid_19)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_20, y=tensors.sigmoid_19, output=tensors.mul_68)
    CONV2D_Q8_0W_F32B_F16_42(rt, x=tensors.mul_68, weight=tensors.p_decoder_up_1_block_1_conv2_weight, bias=tensors.p_decoder_up_1_block_1_conv2_bias, output=tensors.conv2d_28)
    ADD_F32(rt, x=tensors.add_11, y=tensors.conv2d_28, output=tensors.add_12)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_12, weight=tensors.p_decoder_up_1_block_2_norm1_weight, bias=tensors.p_decoder_up_1_block_2_norm1_bias, output=tensors.group_norm_21)
    SIGMOID_F32(rt, x=tensors.group_norm_21, output=tensors.sigmoid_20)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_21, y=tensors.sigmoid_20, output=tensors.mul_71)
    CONV2D_Q8_0W_F32B_F16_43(rt, x=tensors.mul_71, weight=tensors.p_decoder_up_1_block_2_conv1_weight, bias=tensors.p_decoder_up_1_block_2_conv1_bias, output=tensors.conv2d_29)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_29, weight=tensors.p_decoder_up_1_block_2_norm2_weight, bias=tensors.p_decoder_up_1_block_2_norm2_bias, output=tensors.group_norm_22)
    SIGMOID_F32(rt, x=tensors.group_norm_22, output=tensors.sigmoid_21)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_22, y=tensors.sigmoid_21, output=tensors.mul_74)
    CONV2D_Q8_0W_F32B_F16_44(rt, x=tensors.mul_74, weight=tensors.p_decoder_up_1_block_2_conv2_weight, bias=tensors.p_decoder_up_1_block_2_conv2_bias, output=tensors.conv2d_30)
    ADD_F32(rt, x=tensors.add_12, y=tensors.conv2d_30, output=tensors.add_13)
    UPSAMPLE_NEAREST2D_F32(rt, x=tensors.add_13, output=tensors.upsample_nearest2d_2)
    CONV2D_Q8_0W_F32B_F16_45(rt, x=tensors.upsample_nearest2d_2, weight=tensors.p_decoder_up_1_upsample_conv_weight, bias=tensors.p_decoder_up_1_upsample_conv_bias, output=tensors.conv2d_31)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_31, weight=tensors.p_decoder_up_0_block_0_norm1_weight, bias=tensors.p_decoder_up_0_block_0_norm1_bias, output=tensors.group_norm_23)
    SIGMOID_F32(rt, x=tensors.group_norm_23, output=tensors.sigmoid_22)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_23, y=tensors.sigmoid_22, output=tensors.mul_77)
    CONV2D_Q8_0W_F32B_F16_46(rt, x=tensors.mul_77, weight=tensors.p_decoder_up_0_block_0_conv1_weight, bias=tensors.p_decoder_up_0_block_0_conv1_bias, output=tensors.conv2d_32)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_32, weight=tensors.p_decoder_up_0_block_0_norm2_weight, bias=tensors.p_decoder_up_0_block_0_norm2_bias, output=tensors.group_norm_24)
    SIGMOID_F32(rt, x=tensors.group_norm_24, output=tensors.sigmoid_23)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_24, y=tensors.sigmoid_23, output=tensors.mul_80)
    CONV2D_Q8_0W_F32B_F16_47(rt, x=tensors.mul_80, weight=tensors.p_decoder_up_0_block_0_conv2_weight, bias=tensors.p_decoder_up_0_block_0_conv2_bias, output=tensors.conv2d_33)
    CONV2D_Q8_0W_F32B_F16_48(rt, x=tensors.conv2d_31, weight=tensors.p_decoder_up_0_block_0_nin_shortcut_weight, bias=tensors.p_decoder_up_0_block_0_nin_shortcut_bias, output=tensors.conv2d_34)
    ADD_F32(rt, x=tensors.conv2d_34, y=tensors.conv2d_33, output=tensors.add_14)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_14, weight=tensors.p_decoder_up_0_block_1_norm1_weight, bias=tensors.p_decoder_up_0_block_1_norm1_bias, output=tensors.group_norm_25)
    SIGMOID_F32(rt, x=tensors.group_norm_25, output=tensors.sigmoid_24)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_25, y=tensors.sigmoid_24, output=tensors.mul_83)
    CONV2D_Q8_0W_F32B_F16_49(rt, x=tensors.mul_83, weight=tensors.p_decoder_up_0_block_1_conv1_weight, bias=tensors.p_decoder_up_0_block_1_conv1_bias, output=tensors.conv2d_35)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_35, weight=tensors.p_decoder_up_0_block_1_norm2_weight, bias=tensors.p_decoder_up_0_block_1_norm2_bias, output=tensors.group_norm_26)
    SIGMOID_F32(rt, x=tensors.group_norm_26, output=tensors.sigmoid_25)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_26, y=tensors.sigmoid_25, output=tensors.mul_86)
    CONV2D_Q8_0W_F32B_F16_50(rt, x=tensors.mul_86, weight=tensors.p_decoder_up_0_block_1_conv2_weight, bias=tensors.p_decoder_up_0_block_1_conv2_bias, output=tensors.conv2d_36)
    ADD_F32(rt, x=tensors.add_14, y=tensors.conv2d_36, output=tensors.add_15)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_15, weight=tensors.p_decoder_up_0_block_2_norm1_weight, bias=tensors.p_decoder_up_0_block_2_norm1_bias, output=tensors.group_norm_27)
    SIGMOID_F32(rt, x=tensors.group_norm_27, output=tensors.sigmoid_26)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_27, y=tensors.sigmoid_26, output=tensors.mul_89)
    CONV2D_Q8_0W_F32B_F16_51(rt, x=tensors.mul_89, weight=tensors.p_decoder_up_0_block_2_conv1_weight, bias=tensors.p_decoder_up_0_block_2_conv1_bias, output=tensors.conv2d_37)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.conv2d_37, weight=tensors.p_decoder_up_0_block_2_norm2_weight, bias=tensors.p_decoder_up_0_block_2_norm2_bias, output=tensors.group_norm_28)
    SIGMOID_F32(rt, x=tensors.group_norm_28, output=tensors.sigmoid_27)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_28, y=tensors.sigmoid_27, output=tensors.mul_92)
    CONV2D_Q8_0W_F32B_F16_52(rt, x=tensors.mul_92, weight=tensors.p_decoder_up_0_block_2_conv2_weight, bias=tensors.p_decoder_up_0_block_2_conv2_bias, output=tensors.conv2d_38)
    ADD_F32(rt, x=tensors.add_15, y=tensors.conv2d_38, output=tensors.add_16)
    GROUP_NORM_F32W_F32B_F32(rt, x=tensors.add_16, weight=tensors.p_decoder_norm_out_weight, bias=tensors.p_decoder_norm_out_bias, output=tensors.group_norm_29)
    SIGMOID_F32(rt, x=tensors.group_norm_29, output=tensors.sigmoid_28)
    AE_DECODE_MUL_F32(rt, x=tensors.group_norm_29, y=tensors.sigmoid_28, output=tensors.mul_95)
    CONV2D_Q8_0W_F32B_F16_53(rt, x=tensors.mul_95, weight=tensors.p_decoder_conv_out_weight, bias=tensors.p_decoder_conv_out_bias, output=tensors.conv2d_39)


def _require_ae_decode_tensors() -> AeDecodeTensors:
    tensors = model_tensors().ae_decode
    if tensors is None:
        raise RuntimeError("AE decode tensors were not created")
    return tensors


def run_ae_decode(rt: RuntimeSession) -> None:
    tensors = _require_ae_decode_tensors()
    _run_ae_decode_with_tensors(rt, tensors)
