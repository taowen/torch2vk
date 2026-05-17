"""Generated dispatch function for run_flux_single_block."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_broadcast_inner import ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.add_f32_47 import ADD_F32_47
from models.quantized_klein9b.shaders.add_scalar_13 import ADD_SCALAR_13
from models.quantized_klein9b.shaders.add_scalar_17 import ADD_SCALAR_17
from models.quantized_klein9b.shaders.cat_2_f32_46 import CAT_2_F32_46
from models.quantized_klein9b.shaders.cat_3_f32 import CAT_3_F32
from models.quantized_klein9b.shaders.flux_single_block_add_f32 import FLUX_SINGLE_BLOCK_ADD_F32
from models.quantized_klein9b.shaders.flux_single_block_add_scalar import FLUX_SINGLE_BLOCK_ADD_SCALAR
from models.quantized_klein9b.shaders.flux_single_block_cat_2_f32 import FLUX_SINGLE_BLOCK_CAT_2_F32
from models.quantized_klein9b.shaders.flux_single_block_mean_dim_f32 import FLUX_SINGLE_BLOCK_MEAN_DIM_F32
from models.quantized_klein9b.shaders.flux_single_block_mul_broadcast import FLUX_SINGLE_BLOCK_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_single_block_mul_broadcast_28 import FLUX_SINGLE_BLOCK_MUL_BROADCAST_28
from models.quantized_klein9b.shaders.flux_single_block_pow_scalar_f32 import FLUX_SINGLE_BLOCK_POW_SCALAR_F32
from models.quantized_klein9b.shaders.flux_single_block_rsqrt_f32 import FLUX_SINGLE_BLOCK_RSQRT_F32
from models.quantized_klein9b.shaders.flux_single_block_slice_f32 import FLUX_SINGLE_BLOCK_SLICE_F32
from models.quantized_klein9b.shaders.flux_single_block_slice_f32_39 import FLUX_SINGLE_BLOCK_SLICE_F32_39
from models.quantized_klein9b.shaders.flux_single_block_slice_f32_40 import FLUX_SINGLE_BLOCK_SLICE_F32_40
from models.quantized_klein9b.shaders.flux_single_block_tuple_getitem_slice_f32 import FLUX_SINGLE_BLOCK_TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.flux_single_block_tuple_getitem_unbind_f32 import FLUX_SINGLE_BLOCK_TUPLE_GETITEM_UNBIND_F32
from models.quantized_klein9b.shaders.layer_norm_nonew_noneb_f32 import LAYER_NORM_NONEW_NONEB_F32
from models.quantized_klein9b.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.quantized_klein9b.shaders.mul_broadcast_15 import MUL_BROADCAST_15
from models.quantized_klein9b.shaders.mul_broadcast_18 import MUL_BROADCAST_18
from models.quantized_klein9b.shaders.mul_broadcast_20 import MUL_BROADCAST_20
from models.quantized_klein9b.shaders.mul_broadcast_23 import MUL_BROADCAST_23
from models.quantized_klein9b.shaders.mul_broadcast_25 import MUL_BROADCAST_25
from models.quantized_klein9b.shaders.mul_f32 import MUL_F32
from models.quantized_klein9b.shaders.mul_right_broadcast import MUL_RIGHT_BROADCAST
from models.quantized_klein9b.shaders.permute_f32_2731f610b1 import PERMUTE_F32_2731F610B1
from models.quantized_klein9b.shaders.permute_f32_7ebe673eb3 import PERMUTE_F32_7EBE673EB3
from models.quantized_klein9b.shaders.sdpa_f16 import SDPA_F16
from models.quantized_klein9b.shaders.select_float16 import SELECT_FLOAT16
from models.quantized_klein9b.shaders.select_float16_21 import SELECT_FLOAT16_21
from models.quantized_klein9b.shaders.select_float16_22 import SELECT_FLOAT16_22
from models.quantized_klein9b.shaders.select_float16_26 import SELECT_FLOAT16_26
from models.quantized_klein9b.shaders.select_float16_27 import SELECT_FLOAT16_27
from models.quantized_klein9b.shaders.silu_f32 import SILU_F32
from models.quantized_klein9b.shaders.slice_f32_30 import SLICE_F32_30
from models.quantized_klein9b.shaders.slice_f32_31 import SLICE_F32_31
from models.quantized_klein9b.shaders.slice_f32_32 import SLICE_F32_32
from models.quantized_klein9b.shaders.slice_f32_33 import SLICE_F32_33
from models.quantized_klein9b.shaders.slice_f32_34 import SLICE_F32_34
from models.quantized_klein9b.shaders.slice_f32_35 import SLICE_F32_35
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_42 import TUPLE_GETITEM_SLICE_F32_42
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_43 import TUPLE_GETITEM_SLICE_F32_43
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_6 import TUPLE_GETITEM_SLICE_F32_6
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_10 import TUPLE_GETITEM_UNBIND_F32_10
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_9 import TUPLE_GETITEM_UNBIND_F32_9
from models.quantized_klein9b.tensors.flux_single_block import FluxSingleBlockTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_single_block_with_tensors(rt: RuntimeSession, tensors: FluxSingleBlockTensors) -> None:
    FLUX_SINGLE_BLOCK_ADD_SCALAR(rt, x=tensors.mod_scale, output=tensors.add)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.hidden_states, output=tensors.layer_norm)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add, y=tensors.layer_norm, output=tensors.mul)
    ADD_BROADCAST_INNER(rt, x=tensors.mul, y=tensors.mod_shift, output=tensors.add_1)
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.add_1, weight=tensors.p_linear1_weight, output=tensors.linear)
    FLUX_SINGLE_BLOCK_TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear, output=tensors.getitem)
    TUPLE_GETITEM_SLICE_F32_6(rt, x=tensors.linear, output=tensors.getitem_1)
    PERMUTE_F32_2731F610B1(rt, x=tensors.reshape, output=tensors.permute)
    FLUX_SINGLE_BLOCK_TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute, output=tensors.getitem_2)
    TUPLE_GETITEM_UNBIND_F32_9(rt, x=tensors.permute, output=tensors.getitem_3)
    TUPLE_GETITEM_UNBIND_F32_10(rt, x=tensors.permute, output=tensors.getitem_4)
    FLUX_SINGLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    FLUX_SINGLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR_13(rt, x=tensors.mean, output=tensors.add_2)
    FLUX_SINGLE_BLOCK_RSQRT_F32(rt, x=tensors.add_2, output=tensors.rsqrt)
    MUL_BROADCAST_15(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul_1)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_1, y=tensors.p_norm_query_norm_scale, output=tensors.mul_2)
    FLUX_SINGLE_BLOCK_POW_SCALAR_F32(rt, x=tensors.to_2, output=tensors.pow_2)
    FLUX_SINGLE_BLOCK_MEAN_DIM_F32(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_17(rt, x=tensors.mean_1, output=tensors.add_3)
    FLUX_SINGLE_BLOCK_RSQRT_F32(rt, x=tensors.add_3, output=tensors.rsqrt_1)
    MUL_BROADCAST_18(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_3)
    MUL_RIGHT_BROADCAST(rt, x=tensors.to_3, y=tensors.p_norm_key_norm_scale, output=tensors.mul_4)
    SELECT_FLOAT16(rt, x=tensors.pe, output=tensors.select)
    SELECT_FLOAT16(rt, x=tensors.reshape_1, output=tensors.select_1)
    MUL_BROADCAST_20(rt, x=tensors.select, y=tensors.select_1, output=tensors.mul_5)
    SELECT_FLOAT16_21(rt, x=tensors.pe, output=tensors.select_2)
    SELECT_FLOAT16_22(rt, x=tensors.reshape_1, output=tensors.select_3)
    MUL_BROADCAST_23(rt, x=tensors.select_2, y=tensors.select_3, output=tensors.mul_6)
    FLUX_SINGLE_BLOCK_ADD_F32(rt, x=tensors.mul_5, y=tensors.mul_6, output=tensors.add_4)
    SELECT_FLOAT16(rt, x=tensors.pe, output=tensors.select_4)
    SELECT_FLOAT16(rt, x=tensors.reshape_2, output=tensors.select_5)
    MUL_BROADCAST_25(rt, x=tensors.select_4, y=tensors.select_5, output=tensors.mul_7)
    SELECT_FLOAT16_26(rt, x=tensors.pe, output=tensors.select_6)
    SELECT_FLOAT16_27(rt, x=tensors.reshape_2, output=tensors.select_7)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST_28(rt, x=tensors.select_6, y=tensors.select_7, output=tensors.mul_8)
    FLUX_SINGLE_BLOCK_ADD_F32(rt, x=tensors.mul_7, y=tensors.mul_8, output=tensors.add_5)
    SLICE_F32_30(rt, x=tensors.type_as, output=tensors.slice_3)
    FLUX_SINGLE_BLOCK_SLICE_F32(rt, x=tensors.type_as, output=tensors.slice_4)
    SLICE_F32_31(rt, x=tensors.type_as, output=tensors.slice_5)
    SLICE_F32_32(rt, x=tensors.type_as_1, output=tensors.slice_6)
    SLICE_F32_33(rt, x=tensors.getitem_4, output=tensors.slice_7)
    FLUX_SINGLE_BLOCK_SLICE_F32(rt, x=tensors.type_as_1, output=tensors.slice_8)
    FLUX_SINGLE_BLOCK_SLICE_F32(rt, x=tensors.getitem_4, output=tensors.slice_9)
    SLICE_F32_34(rt, x=tensors.type_as_1, output=tensors.slice_10)
    SLICE_F32_35(rt, x=tensors.getitem_4, output=tensors.slice_11)
    FLUX_SINGLE_BLOCK_CAT_2_F32(rt, x0=tensors.slice_3, x1=tensors.slice_5, output=tensors.cat)
    CAT_3_F32(rt, x0=tensors.slice_6, x1=tensors.slice_8, x2=tensors.slice_10, output=tensors.cat_1)
    CAT_3_F32(rt, x0=tensors.slice_7, x1=tensors.slice_9, x2=tensors.slice_11, output=tensors.cat_2)
    SDPA_F16(rt, q=tensors.cat, k=tensors.cat_1, v=tensors.cat_2, output=tensors.scaled_dot_product_attention)
    FLUX_SINGLE_BLOCK_SLICE_F32_39(rt, x=tensors.scaled_dot_product_attention, output=tensors.slice_12)
    FLUX_SINGLE_BLOCK_SLICE_F32_40(rt, x=tensors.scaled_dot_product_attention, output=tensors.slice_13)
    SDPA_F16(rt, q=tensors.slice_4, k=tensors.slice_8, v=tensors.slice_9, output=tensors.scaled_dot_product_attention_1)
    CAT_3_F32(rt, x0=tensors.slice_12, x1=tensors.scaled_dot_product_attention_1, x2=tensors.slice_13, output=tensors.cat_3)
    PERMUTE_F32_7EBE673EB3(rt, x=tensors.cat_3, output=tensors.permute_1)
    TUPLE_GETITEM_SLICE_F32_42(rt, x=tensors.getitem_1, output=tensors.getitem_5)
    TUPLE_GETITEM_SLICE_F32_43(rt, x=tensors.getitem_1, output=tensors.getitem_6)
    SILU_F32(rt, x=tensors.getitem_5, output=tensors.silu)
    MUL_F32(rt, x=tensors.silu, y=tensors.getitem_6, output=tensors.mul_9)
    CAT_2_F32_46(rt, x0=tensors.reshape_5, x1=tensors.mul_9, output=tensors.cat_4)
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.cat_4, weight=tensors.p_linear2_weight, output=tensors.linear_1)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST(rt, x=tensors.mod_gate, y=tensors.linear_1, output=tensors.mul_10)
    ADD_F32_47(rt, x=tensors.hidden_states, y=tensors.mul_10, output=tensors.add_6)


def run_flux_single_block(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors().flux_single_blocks[layer_idx]
    _run_flux_single_block_with_tensors(rt, tensors)
    rt.release_layer_workspace(
        tensors,
        layer=tensors.add_6.layer or "",
        keep=(
            tensors.add_6,
        ),
    )
