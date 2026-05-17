"""Generated dispatch function for run_flux_double_block."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_broadcast_inner import ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.add_f32_51 import ADD_F32_51
from models.quantized_klein9b.shaders.add_f32_57 import ADD_F32_57
from models.quantized_klein9b.shaders.add_f32_58 import ADD_F32_58
from models.quantized_klein9b.shaders.add_f32_60 import ADD_F32_60
from models.quantized_klein9b.shaders.add_scalar_11 import ADD_SCALAR_11
from models.quantized_klein9b.shaders.add_scalar_15 import ADD_SCALAR_15
from models.quantized_klein9b.shaders.add_scalar_21 import ADD_SCALAR_21
from models.quantized_klein9b.shaders.add_scalar_23 import ADD_SCALAR_23
from models.quantized_klein9b.shaders.cat_2_f32_26 import CAT_2_F32_26
from models.quantized_klein9b.shaders.cat_3_f32 import CAT_3_F32
from models.quantized_klein9b.shaders.flux_double_block_add_f32 import FLUX_DOUBLE_BLOCK_ADD_F32
from models.quantized_klein9b.shaders.flux_double_block_add_scalar import FLUX_DOUBLE_BLOCK_ADD_SCALAR
from models.quantized_klein9b.shaders.flux_double_block_cat_2_f32 import FLUX_DOUBLE_BLOCK_CAT_2_F32
from models.quantized_klein9b.shaders.flux_double_block_mean_dim_f32 import FLUX_DOUBLE_BLOCK_MEAN_DIM_F32
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast import FLUX_DOUBLE_BLOCK_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast_28 import FLUX_DOUBLE_BLOCK_MUL_BROADCAST_28
from models.quantized_klein9b.shaders.flux_double_block_pow_scalar_f32 import FLUX_DOUBLE_BLOCK_POW_SCALAR_F32
from models.quantized_klein9b.shaders.flux_double_block_rsqrt_f32 import FLUX_DOUBLE_BLOCK_RSQRT_F32
from models.quantized_klein9b.shaders.flux_double_block_slice_f32 import FLUX_DOUBLE_BLOCK_SLICE_F32
from models.quantized_klein9b.shaders.flux_double_block_tuple_getitem_slice_f32 import FLUX_DOUBLE_BLOCK_TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.layer_norm_nonew_noneb_f32 import LAYER_NORM_NONEW_NONEB_F32
from models.quantized_klein9b.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.quantized_klein9b.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.quantized_klein9b.shaders.mul_broadcast_13 import MUL_BROADCAST_13
from models.quantized_klein9b.shaders.mul_broadcast_16 import MUL_BROADCAST_16
from models.quantized_klein9b.shaders.mul_broadcast_22 import MUL_BROADCAST_22
from models.quantized_klein9b.shaders.mul_broadcast_24 import MUL_BROADCAST_24
from models.quantized_klein9b.shaders.mul_broadcast_31 import MUL_BROADCAST_31
from models.quantized_klein9b.shaders.mul_broadcast_33 import MUL_BROADCAST_33
from models.quantized_klein9b.shaders.mul_broadcast_36 import MUL_BROADCAST_36
from models.quantized_klein9b.shaders.mul_f32 import MUL_F32
from models.quantized_klein9b.shaders.mul_right_broadcast import MUL_RIGHT_BROADCAST
from models.quantized_klein9b.shaders.permute_f32_0d3ab17ae3 import PERMUTE_F32_0D3AB17AE3
from models.quantized_klein9b.shaders.permute_f32_7ebe673eb3 import PERMUTE_F32_7EBE673EB3
from models.quantized_klein9b.shaders.permute_f32_8475d3a978 import PERMUTE_F32_8475D3A978
from models.quantized_klein9b.shaders.sdpa_f16 import SDPA_F16
from models.quantized_klein9b.shaders.select_float16 import SELECT_FLOAT16
from models.quantized_klein9b.shaders.select_float16_29 import SELECT_FLOAT16_29
from models.quantized_klein9b.shaders.select_float16_30 import SELECT_FLOAT16_30
from models.quantized_klein9b.shaders.select_float16_34 import SELECT_FLOAT16_34
from models.quantized_klein9b.shaders.select_float16_35 import SELECT_FLOAT16_35
from models.quantized_klein9b.shaders.silu_f32 import SILU_F32
from models.quantized_klein9b.shaders.slice_f32_38 import SLICE_F32_38
from models.quantized_klein9b.shaders.slice_f32_39 import SLICE_F32_39
from models.quantized_klein9b.shaders.slice_f32_40 import SLICE_F32_40
from models.quantized_klein9b.shaders.slice_f32_41 import SLICE_F32_41
from models.quantized_klein9b.shaders.slice_f32_42 import SLICE_F32_42
from models.quantized_klein9b.shaders.slice_f32_43 import SLICE_F32_43
from models.quantized_klein9b.shaders.slice_f32_46 import SLICE_F32_46
from models.quantized_klein9b.shaders.slice_f32_47 import SLICE_F32_47
from models.quantized_klein9b.shaders.slice_f32_49 import SLICE_F32_49
from models.quantized_klein9b.shaders.slice_f32_50 import SLICE_F32_50
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_54 import TUPLE_GETITEM_SLICE_F32_54
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_59 import TUPLE_GETITEM_SLICE_F32_59
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32 import TUPLE_GETITEM_UNBIND_F32
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_18 import TUPLE_GETITEM_UNBIND_F32_18
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_19 import TUPLE_GETITEM_UNBIND_F32_19
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_20 import TUPLE_GETITEM_UNBIND_F32_20
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_7 import TUPLE_GETITEM_UNBIND_F32_7
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_8 import TUPLE_GETITEM_UNBIND_F32_8
from models.quantized_klein9b.tensors.flux_double_block import FluxDoubleBlockTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_double_block_with_tensors(rt: RuntimeSession, tensors: FluxDoubleBlockTensors) -> None:
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.img, output=tensors.layer_norm)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.img_mod1_scale, output=tensors.add)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add, y=tensors.layer_norm, output=tensors.mul)
    ADD_BROADCAST_INNER(rt, x=tensors.mul, y=tensors.img_mod1_shift, output=tensors.add_1)
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.add_1, weight=tensors.p_img_attn_qkv_weight, output=tensors.linear)
    PERMUTE_F32_0D3AB17AE3(rt, x=tensors.reshape, output=tensors.permute)
    TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute, output=tensors.getitem)
    TUPLE_GETITEM_UNBIND_F32_7(rt, x=tensors.permute, output=tensors.getitem_1)
    TUPLE_GETITEM_UNBIND_F32_8(rt, x=tensors.permute, output=tensors.getitem_2)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR_11(rt, x=tensors.mean, output=tensors.add_2)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_2, output=tensors.rsqrt)
    MUL_BROADCAST_13(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul_1)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_1, y=tensors.p_img_attn_norm_query_norm_scale, output=tensors.mul_2)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_2, output=tensors.pow_2)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_15(rt, x=tensors.mean_1, output=tensors.add_3)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_3, output=tensors.rsqrt_1)
    MUL_BROADCAST_16(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_3)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_3, y=tensors.p_img_attn_norm_key_norm_scale, output=tensors.mul_4)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.txt, output=tensors.layer_norm_1)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.txt_mod1_scale, output=tensors.add_4)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_4, y=tensors.layer_norm_1, output=tensors.mul_5)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_5, y=tensors.txt_mod1_shift, output=tensors.add_5)
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.add_5, weight=tensors.p_txt_attn_qkv_weight, output=tensors.linear_1)
    PERMUTE_F32_8475D3A978(rt, x=tensors.reshape_1, output=tensors.permute_1)
    TUPLE_GETITEM_UNBIND_F32_18(rt, x=tensors.permute_1, output=tensors.getitem_3)
    TUPLE_GETITEM_UNBIND_F32_19(rt, x=tensors.permute_1, output=tensors.getitem_4)
    TUPLE_GETITEM_UNBIND_F32_20(rt, x=tensors.permute_1, output=tensors.getitem_5)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_3)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_3, output=tensors.mean_2)
    ADD_SCALAR_21(rt, x=tensors.mean_2, output=tensors.add_6)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_2)
    MUL_BROADCAST_22(rt, x=tensors.to_6, y=tensors.rsqrt_2, output=tensors.mul_6)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_7, y=tensors.p_txt_attn_norm_query_norm_scale, output=tensors.mul_7)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_8, output=tensors.pow_4)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    ADD_SCALAR_23(rt, x=tensors.mean_3, output=tensors.add_7)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_7, output=tensors.rsqrt_3)
    MUL_BROADCAST_24(rt, x=tensors.to_8, y=tensors.rsqrt_3, output=tensors.mul_8)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_9, y=tensors.p_txt_attn_norm_key_norm_scale, output=tensors.mul_9)
    FLUX_DOUBLE_BLOCK_CAT_2_F32(rt, x0=tensors.to_10, x1=tensors.to_4, output=tensors.cat)
    FLUX_DOUBLE_BLOCK_CAT_2_F32(rt, x0=tensors.to_11, x1=tensors.to_5, output=tensors.cat_1)
    FLUX_DOUBLE_BLOCK_CAT_2_F32(rt, x0=tensors.getitem_5, x1=tensors.getitem_2, output=tensors.cat_2)
    CAT_2_F32_26(rt, x0=tensors.pe_ctx, x1=tensors.pe, output=tensors.cat_3)
    SELECT_FLOAT16(rt, x=tensors.cat_3, output=tensors.select)
    SELECT_FLOAT16(rt, x=tensors.reshape_2, output=tensors.select_1)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST_28(rt, x=tensors.select, y=tensors.select_1, output=tensors.mul_10)
    SELECT_FLOAT16_29(rt, x=tensors.cat_3, output=tensors.select_2)
    SELECT_FLOAT16_30(rt, x=tensors.reshape_2, output=tensors.select_3)
    MUL_BROADCAST_31(rt, x=tensors.select_2, y=tensors.select_3, output=tensors.mul_11)
    FLUX_DOUBLE_BLOCK_ADD_F32(rt, x=tensors.mul_10, y=tensors.mul_11, output=tensors.add_8)
    SELECT_FLOAT16(rt, x=tensors.cat_3, output=tensors.select_4)
    SELECT_FLOAT16(rt, x=tensors.reshape_3, output=tensors.select_5)
    MUL_BROADCAST_33(rt, x=tensors.select_4, y=tensors.select_5, output=tensors.mul_12)
    SELECT_FLOAT16_34(rt, x=tensors.cat_3, output=tensors.select_6)
    SELECT_FLOAT16_35(rt, x=tensors.reshape_3, output=tensors.select_7)
    MUL_BROADCAST_36(rt, x=tensors.select_6, y=tensors.select_7, output=tensors.mul_13)
    FLUX_DOUBLE_BLOCK_ADD_F32(rt, x=tensors.mul_12, y=tensors.mul_13, output=tensors.add_9)
    SLICE_F32_38(rt, x=tensors.type_as, output=tensors.slice_3)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.type_as, output=tensors.slice_4)
    SLICE_F32_39(rt, x=tensors.type_as, output=tensors.slice_5)
    SLICE_F32_40(rt, x=tensors.type_as_1, output=tensors.slice_6)
    SLICE_F32_41(rt, x=tensors.cat_2, output=tensors.slice_7)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.type_as_1, output=tensors.slice_8)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.cat_2, output=tensors.slice_9)
    SLICE_F32_42(rt, x=tensors.type_as_1, output=tensors.slice_10)
    SLICE_F32_43(rt, x=tensors.cat_2, output=tensors.slice_11)
    FLUX_DOUBLE_BLOCK_CAT_2_F32(rt, x0=tensors.slice_3, x1=tensors.slice_5, output=tensors.cat_4)
    CAT_3_F32(rt, x0=tensors.slice_6, x1=tensors.slice_8, x2=tensors.slice_10, output=tensors.cat_5)
    CAT_3_F32(rt, x0=tensors.slice_7, x1=tensors.slice_9, x2=tensors.slice_11, output=tensors.cat_6)
    SDPA_F16(rt, q=tensors.cat_4, k=tensors.cat_5, v=tensors.cat_6, output=tensors.scaled_dot_product_attention)
    SLICE_F32_46(rt, x=tensors.scaled_dot_product_attention, output=tensors.slice_12)
    SLICE_F32_47(rt, x=tensors.scaled_dot_product_attention, output=tensors.slice_13)
    SDPA_F16(rt, q=tensors.slice_4, k=tensors.slice_8, v=tensors.slice_9, output=tensors.scaled_dot_product_attention_1)
    CAT_3_F32(rt, x0=tensors.slice_12, x1=tensors.scaled_dot_product_attention_1, x2=tensors.slice_13, output=tensors.cat_7)
    PERMUTE_F32_7EBE673EB3(rt, x=tensors.cat_7, output=tensors.permute_2)
    SLICE_F32_49(rt, x=tensors.reshape_6, output=tensors.slice_14)
    SLICE_F32_50(rt, x=tensors.reshape_6, output=tensors.slice_15)
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.slice_15, weight=tensors.p_img_attn_proj_weight, output=tensors.linear_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.img_mod1_gate, y=tensors.linear_2, output=tensors.mul_14)
    ADD_F32_51(rt, x=tensors.img, y=tensors.mul_14, output=tensors.add_10)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.img_mod2_scale, output=tensors.add_11)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.add_10, output=tensors.layer_norm_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_11, y=tensors.layer_norm_2, output=tensors.mul_15)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_15, y=tensors.img_mod2_shift, output=tensors.add_12)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.add_12, weight=tensors.p_img_mlp_0_weight, output=tensors.linear_3)
    FLUX_DOUBLE_BLOCK_TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear_3, output=tensors.getitem_6)
    TUPLE_GETITEM_SLICE_F32_54(rt, x=tensors.linear_3, output=tensors.getitem_7)
    SILU_F32(rt, x=tensors.getitem_6, output=tensors.silu)
    MUL_F32(rt, x=tensors.silu, y=tensors.getitem_7, output=tensors.mul_16)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_16, weight=tensors.p_img_mlp_2_weight, output=tensors.linear_4)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.img_mod2_gate, y=tensors.linear_4, output=tensors.mul_17)
    ADD_F32_57(rt, x=tensors.add_10, y=tensors.mul_17, output=tensors.add_13)
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.slice_14, weight=tensors.p_txt_attn_proj_weight, output=tensors.linear_5)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.txt_mod1_gate, y=tensors.linear_5, output=tensors.mul_18)
    ADD_F32_58(rt, x=tensors.txt, y=tensors.mul_18, output=tensors.add_14)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.txt_mod2_scale, output=tensors.add_15)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.add_14, output=tensors.layer_norm_3)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_15, y=tensors.layer_norm_3, output=tensors.mul_19)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_19, y=tensors.txt_mod2_shift, output=tensors.add_16)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.add_16, weight=tensors.p_txt_mlp_0_weight, output=tensors.linear_6)
    FLUX_DOUBLE_BLOCK_TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear_6, output=tensors.getitem_8)
    TUPLE_GETITEM_SLICE_F32_59(rt, x=tensors.linear_6, output=tensors.getitem_9)
    SILU_F32(rt, x=tensors.getitem_8, output=tensors.silu_1)
    MUL_F32(rt, x=tensors.silu_1, y=tensors.getitem_9, output=tensors.mul_20)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_20, weight=tensors.p_txt_mlp_2_weight, output=tensors.linear_7)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.txt_mod2_gate, y=tensors.linear_7, output=tensors.mul_21)
    ADD_F32_60(rt, x=tensors.add_14, y=tensors.mul_21, output=tensors.add_17)


def run_flux_double_block(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors().flux_double_blocks[layer_idx]
    _run_flux_double_block_with_tensors(rt, tensors)
    rt.release_layer_workspace(
        tensors,
        layer=tensors.add_13.layer or "",
        keep=(
            tensors.add_13,
            tensors.add_17,
        ),
    )
