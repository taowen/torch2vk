"""Generated dispatch function for run_flux_double_block."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_broadcast_inner import ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.add_f32_54 import ADD_F32_54
from models.quantized_klein9b.shaders.add_f32_59 import ADD_F32_59
from models.quantized_klein9b.shaders.add_f32_60 import ADD_F32_60
from models.quantized_klein9b.shaders.add_f32_62 import ADD_F32_62
from models.quantized_klein9b.shaders.add_scalar_11 import ADD_SCALAR_11
from models.quantized_klein9b.shaders.add_scalar_14 import ADD_SCALAR_14
from models.quantized_klein9b.shaders.add_scalar_16 import ADD_SCALAR_16
from models.quantized_klein9b.shaders.add_scalar_21 import ADD_SCALAR_21
from models.quantized_klein9b.shaders.add_scalar_23 import ADD_SCALAR_23
from models.quantized_klein9b.shaders.add_scalar_25 import ADD_SCALAR_25
from models.quantized_klein9b.shaders.cat_2_f32_27 import CAT_2_F32_27
from models.quantized_klein9b.shaders.cat_2_f32_28 import CAT_2_F32_28
from models.quantized_klein9b.shaders.cat_2_f32_29 import CAT_2_F32_29
from models.quantized_klein9b.shaders.cat_2_f32_47 import CAT_2_F32_47
from models.quantized_klein9b.shaders.flux_double_block_add_f32 import FLUX_DOUBLE_BLOCK_ADD_F32
from models.quantized_klein9b.shaders.flux_double_block_add_scalar import FLUX_DOUBLE_BLOCK_ADD_SCALAR
from models.quantized_klein9b.shaders.flux_double_block_cat_2_f32 import FLUX_DOUBLE_BLOCK_CAT_2_F32
from models.quantized_klein9b.shaders.flux_double_block_cat_3_f32 import FLUX_DOUBLE_BLOCK_CAT_3_F32
from models.quantized_klein9b.shaders.flux_double_block_mean_dim_f32 import FLUX_DOUBLE_BLOCK_MEAN_DIM_F32
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast import FLUX_DOUBLE_BLOCK_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_double_block_mul_f32 import FLUX_DOUBLE_BLOCK_MUL_F32
from models.quantized_klein9b.shaders.flux_double_block_pow_scalar_f32 import FLUX_DOUBLE_BLOCK_POW_SCALAR_F32
from models.quantized_klein9b.shaders.flux_double_block_rsqrt_f32 import FLUX_DOUBLE_BLOCK_RSQRT_F32
from models.quantized_klein9b.shaders.flux_double_block_silu_f32 import FLUX_DOUBLE_BLOCK_SILU_F32
from models.quantized_klein9b.shaders.flux_double_block_slice_f32 import FLUX_DOUBLE_BLOCK_SLICE_F32
from models.quantized_klein9b.shaders.layer_norm_nonew_noneb_f32 import LAYER_NORM_NONEW_NONEB_F32
from models.quantized_klein9b.shaders.linear_nobias_q8_0_f32_act_f32 import LINEAR_NOBIAS_Q8_0_F32_ACT_F32
from models.quantized_klein9b.shaders.mul_broadcast_13 import MUL_BROADCAST_13
from models.quantized_klein9b.shaders.mul_broadcast_15 import MUL_BROADCAST_15
from models.quantized_klein9b.shaders.mul_broadcast_22 import MUL_BROADCAST_22
from models.quantized_klein9b.shaders.mul_broadcast_24 import MUL_BROADCAST_24
from models.quantized_klein9b.shaders.mul_broadcast_31 import MUL_BROADCAST_31
from models.quantized_klein9b.shaders.mul_broadcast_34 import MUL_BROADCAST_34
from models.quantized_klein9b.shaders.mul_broadcast_36 import MUL_BROADCAST_36
from models.quantized_klein9b.shaders.mul_broadcast_39 import MUL_BROADCAST_39
from models.quantized_klein9b.shaders.mul_right_broadcast import MUL_RIGHT_BROADCAST
from models.quantized_klein9b.shaders.permute_f32_0d3ab17ae3 import PERMUTE_F32_0D3AB17AE3
from models.quantized_klein9b.shaders.permute_f32_7ebe673eb3 import PERMUTE_F32_7EBE673EB3
from models.quantized_klein9b.shaders.permute_f32_8475d3a978 import PERMUTE_F32_8475D3A978
from models.quantized_klein9b.shaders.sdpa_f32 import SDPA_F32
from models.quantized_klein9b.shaders.select_float32 import SELECT_FLOAT32
from models.quantized_klein9b.shaders.select_float32_32 import SELECT_FLOAT32_32
from models.quantized_klein9b.shaders.select_float32_33 import SELECT_FLOAT32_33
from models.quantized_klein9b.shaders.select_float32_37 import SELECT_FLOAT32_37
from models.quantized_klein9b.shaders.select_float32_38 import SELECT_FLOAT32_38
from models.quantized_klein9b.shaders.slice_f32_41 import SLICE_F32_41
from models.quantized_klein9b.shaders.slice_f32_42 import SLICE_F32_42
from models.quantized_klein9b.shaders.slice_f32_43 import SLICE_F32_43
from models.quantized_klein9b.shaders.slice_f32_44 import SLICE_F32_44
from models.quantized_klein9b.shaders.slice_f32_45 import SLICE_F32_45
from models.quantized_klein9b.shaders.slice_f32_46 import SLICE_F32_46
from models.quantized_klein9b.shaders.slice_f32_50 import SLICE_F32_50
from models.quantized_klein9b.shaders.slice_f32_52 import SLICE_F32_52
from models.quantized_klein9b.shaders.slice_f32_53 import SLICE_F32_53
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32 import TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_56 import TUPLE_GETITEM_SLICE_F32_56
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_61 import TUPLE_GETITEM_SLICE_F32_61
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32 import TUPLE_GETITEM_UNBIND_F32
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
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_1, weight=tensors.p_img_attn_qkv_weight, output=tensors.linear)
    PERMUTE_F32_0D3AB17AE3(rt, x=tensors.reshape, output=tensors.permute)
    TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute, output=tensors.getitem)
    TUPLE_GETITEM_UNBIND_F32_7(rt, x=tensors.permute, output=tensors.getitem_1)
    TUPLE_GETITEM_UNBIND_F32_8(rt, x=tensors.permute, output=tensors.getitem_2)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_1, output=tensors.pow_1)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR_11(rt, x=tensors.mean, output=tensors.add_2)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_2, output=tensors.rsqrt)
    MUL_BROADCAST_13(rt, x=tensors.to_1, y=tensors.rsqrt, output=tensors.mul_1)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_4, output=tensors.pow_2)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_14(rt, x=tensors.mean_1, output=tensors.add_3)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_3, output=tensors.rsqrt_1)
    MUL_BROADCAST_15(rt, x=tensors.to_4, y=tensors.rsqrt_1, output=tensors.mul_2)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_3)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_3, output=tensors.mean_2)
    ADD_SCALAR_16(rt, x=tensors.mean_2, output=tensors.add_4)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_4, output=tensors.rsqrt_2)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_2, y=tensors.p_img_attn_norm_query_norm_scale, output=tensors.mul_3)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_5, y=tensors.p_img_attn_norm_key_norm_scale, output=tensors.mul_4)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.txt, output=tensors.layer_norm_1)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.txt_mod1_scale, output=tensors.add_5)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_5, y=tensors.layer_norm_1, output=tensors.mul_5)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_5, y=tensors.txt_mod1_shift, output=tensors.add_6)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_6, weight=tensors.p_txt_attn_qkv_weight, output=tensors.linear_1)
    PERMUTE_F32_8475D3A978(rt, x=tensors.reshape_1, output=tensors.permute_1)
    TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute_1, output=tensors.getitem_3)
    TUPLE_GETITEM_UNBIND_F32_19(rt, x=tensors.permute_1, output=tensors.getitem_4)
    TUPLE_GETITEM_UNBIND_F32_20(rt, x=tensors.permute_1, output=tensors.getitem_5)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_8, output=tensors.pow_4)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    ADD_SCALAR_21(rt, x=tensors.mean_3, output=tensors.add_7)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_7, output=tensors.rsqrt_3)
    MUL_BROADCAST_22(rt, x=tensors.to_8, y=tensors.rsqrt_3, output=tensors.mul_6)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_11, output=tensors.pow_5)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_5, output=tensors.mean_4)
    ADD_SCALAR_23(rt, x=tensors.mean_4, output=tensors.add_8)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_8, output=tensors.rsqrt_4)
    MUL_BROADCAST_24(rt, x=tensors.to_11, y=tensors.rsqrt_4, output=tensors.mul_7)
    FLUX_DOUBLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_13, output=tensors.pow_6)
    FLUX_DOUBLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_6, output=tensors.mean_5)
    ADD_SCALAR_25(rt, x=tensors.mean_5, output=tensors.add_9)
    FLUX_DOUBLE_BLOCK_RSQRT_F32(rt, x=tensors.add_9, output=tensors.rsqrt_5)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_9, y=tensors.p_txt_attn_norm_query_norm_scale, output=tensors.mul_8)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_12, y=tensors.p_txt_attn_norm_key_norm_scale, output=tensors.mul_9)
    FLUX_DOUBLE_BLOCK_CAT_2_F32(rt, x0=tensors.pe_ctx, x1=tensors.pe, output=tensors.cat)
    CAT_2_F32_27(rt, x0=tensors.mul_8, x1=tensors.mul_3, output=tensors.cat_1)
    CAT_2_F32_28(rt, x0=tensors.mul_9, x1=tensors.mul_4, output=tensors.cat_2)
    CAT_2_F32_29(rt, x0=tensors.getitem_5, x1=tensors.getitem_2, output=tensors.cat_3)
    SELECT_FLOAT32(rt, x=tensors.cat, output=tensors.select)
    SELECT_FLOAT32(rt, x=tensors.reshape_2, output=tensors.select_1)
    MUL_BROADCAST_31(rt, x=tensors.select, y=tensors.select_1, output=tensors.mul_10)
    SELECT_FLOAT32_32(rt, x=tensors.cat, output=tensors.select_2)
    SELECT_FLOAT32_33(rt, x=tensors.reshape_2, output=tensors.select_3)
    MUL_BROADCAST_34(rt, x=tensors.select_2, y=tensors.select_3, output=tensors.mul_11)
    FLUX_DOUBLE_BLOCK_ADD_F32(rt, x=tensors.mul_10, y=tensors.mul_11, output=tensors.add_10)
    SELECT_FLOAT32(rt, x=tensors.cat, output=tensors.select_4)
    SELECT_FLOAT32(rt, x=tensors.reshape_3, output=tensors.select_5)
    MUL_BROADCAST_36(rt, x=tensors.select_4, y=tensors.select_5, output=tensors.mul_12)
    SELECT_FLOAT32_37(rt, x=tensors.cat, output=tensors.select_6)
    SELECT_FLOAT32_38(rt, x=tensors.reshape_3, output=tensors.select_7)
    MUL_BROADCAST_39(rt, x=tensors.select_6, y=tensors.select_7, output=tensors.mul_13)
    FLUX_DOUBLE_BLOCK_ADD_F32(rt, x=tensors.mul_12, y=tensors.mul_13, output=tensors.add_11)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.type_as, output=tensors.slice_1)
    SLICE_F32_41(rt, x=tensors.type_as, output=tensors.slice_2)
    SLICE_F32_42(rt, x=tensors.type_as, output=tensors.slice_3)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.type_as_1, output=tensors.slice_4)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.cat_3, output=tensors.slice_5)
    SLICE_F32_43(rt, x=tensors.type_as_1, output=tensors.slice_6)
    SLICE_F32_44(rt, x=tensors.cat_3, output=tensors.slice_7)
    SLICE_F32_45(rt, x=tensors.type_as_1, output=tensors.slice_8)
    SLICE_F32_46(rt, x=tensors.cat_3, output=tensors.slice_9)
    CAT_2_F32_47(rt, x0=tensors.slice_1, x1=tensors.slice_3, output=tensors.cat_4)
    FLUX_DOUBLE_BLOCK_CAT_3_F32(rt, x0=tensors.slice_4, x1=tensors.slice_6, x2=tensors.slice_8, output=tensors.cat_5)
    FLUX_DOUBLE_BLOCK_CAT_3_F32(rt, x0=tensors.slice_5, x1=tensors.slice_7, x2=tensors.slice_9, output=tensors.cat_6)
    SDPA_F32(rt, q=tensors.cat_4, k=tensors.cat_5, v=tensors.cat_6, output=tensors.scaled_dot_product_attention)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.scaled_dot_product_attention, output=tensors.slice_10)
    SLICE_F32_50(rt, x=tensors.scaled_dot_product_attention, output=tensors.slice_11)
    SDPA_F32(rt, q=tensors.slice_2, k=tensors.slice_6, v=tensors.slice_7, output=tensors.scaled_dot_product_attention_1)
    FLUX_DOUBLE_BLOCK_CAT_3_F32(rt, x0=tensors.slice_10, x1=tensors.scaled_dot_product_attention_1, x2=tensors.slice_11, output=tensors.cat_7)
    PERMUTE_F32_7EBE673EB3(rt, x=tensors.cat_7, output=tensors.permute_2)
    SLICE_F32_52(rt, x=tensors.reshape_6, output=tensors.slice_12)
    SLICE_F32_53(rt, x=tensors.reshape_6, output=tensors.slice_13)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.slice_13, weight=tensors.p_img_attn_proj_weight, output=tensors.linear_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.img_mod1_gate, y=tensors.linear_2, output=tensors.mul_14)
    ADD_F32_54(rt, x=tensors.img, y=tensors.mul_14, output=tensors.add_12)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.img_mod2_scale, output=tensors.add_13)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.add_12, output=tensors.layer_norm_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_13, y=tensors.layer_norm_2, output=tensors.mul_15)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_15, y=tensors.img_mod2_shift, output=tensors.add_14)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_14, weight=tensors.p_img_mlp_0_weight, output=tensors.linear_3)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear_3, output=tensors.getitem_6)
    TUPLE_GETITEM_SLICE_F32_56(rt, x=tensors.linear_3, output=tensors.getitem_7)
    FLUX_DOUBLE_BLOCK_SILU_F32(rt, x=tensors.getitem_6, output=tensors.silu)
    FLUX_DOUBLE_BLOCK_MUL_F32(rt, x=tensors.silu, y=tensors.getitem_7, output=tensors.mul_16)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.mul_16, weight=tensors.p_img_mlp_2_weight, output=tensors.linear_4)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.img_mod2_gate, y=tensors.linear_4, output=tensors.mul_17)
    ADD_F32_59(rt, x=tensors.add_12, y=tensors.mul_17, output=tensors.add_15)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.slice_12, weight=tensors.p_txt_attn_proj_weight, output=tensors.linear_5)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.txt_mod1_gate, y=tensors.linear_5, output=tensors.mul_18)
    ADD_F32_60(rt, x=tensors.txt, y=tensors.mul_18, output=tensors.add_16)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.txt_mod2_scale, output=tensors.add_17)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.add_16, output=tensors.layer_norm_3)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_17, y=tensors.layer_norm_3, output=tensors.mul_19)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_19, y=tensors.txt_mod2_shift, output=tensors.add_18)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_18, weight=tensors.p_txt_mlp_0_weight, output=tensors.linear_6)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear_6, output=tensors.getitem_8)
    TUPLE_GETITEM_SLICE_F32_61(rt, x=tensors.linear_6, output=tensors.getitem_9)
    FLUX_DOUBLE_BLOCK_SILU_F32(rt, x=tensors.getitem_8, output=tensors.silu_1)
    FLUX_DOUBLE_BLOCK_MUL_F32(rt, x=tensors.silu_1, y=tensors.getitem_9, output=tensors.mul_20)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.mul_20, weight=tensors.p_txt_mlp_2_weight, output=tensors.linear_7)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.txt_mod2_gate, y=tensors.linear_7, output=tensors.mul_21)
    ADD_F32_62(rt, x=tensors.add_16, y=tensors.mul_21, output=tensors.add_19)


def run_flux_double_block(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors().flux_double_blocks[layer_idx]
    _run_flux_double_block_with_tensors(rt, tensors)
    rt.release_layer_workspace(
        tensors,
        layer=tensors.to_6.layer or "",
        keep=(
            tensors.to_6,
            tensors.to_13,
            tensors.rsqrt_2,
            tensors.rsqrt_5,
            tensors.to_9,
            tensors.to_2,
            tensors.to_12,
            tensors.to_5,
            tensors.mul_8,
            tensors.mul_3,
            tensors.mul_9,
            tensors.mul_4,
            tensors.getitem_5,
            tensors.getitem_2,
            tensors.to_14,
            tensors.to_15,
            tensors.cat_3,
            tensors.cat,
            tensors.type_as,
            tensors.type_as_1,
            tensors.reshape_6,
            tensors.slice_12,
            tensors.slice_13,
            tensors.linear_2,
            tensors.add_12,
            tensors.add_14,
            tensors.linear_3,
            tensors.mul_16,
            tensors.linear_4,
            tensors.add_15,
            tensors.linear_5,
            tensors.add_16,
            tensors.add_18,
            tensors.linear_6,
            tensors.mul_20,
            tensors.linear_7,
            tensors.mul_21,
            tensors.add_19,
        ),
    )
