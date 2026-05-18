"""Generated dispatch function for run_flux_double_block."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_broadcast_inner import ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.add_f32_31 import ADD_F32_31
from models.quantized_klein9b.shaders.add_f32_37 import ADD_F32_37
from models.quantized_klein9b.shaders.add_f32_39 import ADD_F32_39
from models.quantized_klein9b.shaders.cat_2_f32_14 import CAT_2_F32_14
from models.quantized_klein9b.shaders.cat_2_f32_15 import CAT_2_F32_15
from models.quantized_klein9b.shaders.cat_2_f32_16 import CAT_2_F32_16
from models.quantized_klein9b.shaders.flux_double_block_add_f32 import FLUX_DOUBLE_BLOCK_ADD_F32
from models.quantized_klein9b.shaders.flux_double_block_add_f32_36 import FLUX_DOUBLE_BLOCK_ADD_F32_36
from models.quantized_klein9b.shaders.flux_double_block_add_scalar import FLUX_DOUBLE_BLOCK_ADD_SCALAR
from models.quantized_klein9b.shaders.flux_double_block_cat_2_f32 import FLUX_DOUBLE_BLOCK_CAT_2_F32
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast import FLUX_DOUBLE_BLOCK_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast_18 import FLUX_DOUBLE_BLOCK_MUL_BROADCAST_18
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast_21 import FLUX_DOUBLE_BLOCK_MUL_BROADCAST_21
from models.quantized_klein9b.shaders.flux_double_block_mul_broadcast_26 import FLUX_DOUBLE_BLOCK_MUL_BROADCAST_26
from models.quantized_klein9b.shaders.flux_double_block_mul_f32 import FLUX_DOUBLE_BLOCK_MUL_F32
from models.quantized_klein9b.shaders.flux_double_block_silu_f32 import FLUX_DOUBLE_BLOCK_SILU_F32
from models.quantized_klein9b.shaders.flux_double_block_slice_f32 import FLUX_DOUBLE_BLOCK_SLICE_F32
from models.quantized_klein9b.shaders.layer_norm_nonew_noneb_f32 import LAYER_NORM_NONEW_NONEB_F32
from models.quantized_klein9b.shaders.linear_nobias_q8_0_f32_act_f32 import LINEAR_NOBIAS_Q8_0_F32_ACT_F32
from models.quantized_klein9b.shaders.mul_broadcast_23 import MUL_BROADCAST_23
from models.quantized_klein9b.shaders.permute_f32_0d3ab17ae3 import PERMUTE_F32_0D3AB17AE3
from models.quantized_klein9b.shaders.permute_f32_7ebe673eb3 import PERMUTE_F32_7EBE673EB3
from models.quantized_klein9b.shaders.permute_f32_8475d3a978 import PERMUTE_F32_8475D3A978
from models.quantized_klein9b.shaders.rms_norm_f32w_f32 import RMS_NORM_F32W_F32
from models.quantized_klein9b.shaders.sdpa_f32 import SDPA_F32
from models.quantized_klein9b.shaders.select_float32 import SELECT_FLOAT32
from models.quantized_klein9b.shaders.select_float32_19 import SELECT_FLOAT32_19
from models.quantized_klein9b.shaders.select_float32_20 import SELECT_FLOAT32_20
from models.quantized_klein9b.shaders.select_float32_24 import SELECT_FLOAT32_24
from models.quantized_klein9b.shaders.select_float32_25 import SELECT_FLOAT32_25
from models.quantized_klein9b.shaders.slice_f32_30 import SLICE_F32_30
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32 import TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_33 import TUPLE_GETITEM_SLICE_F32_33
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_38 import TUPLE_GETITEM_SLICE_F32_38
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32 import TUPLE_GETITEM_UNBIND_F32
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_11 import TUPLE_GETITEM_UNBIND_F32_11
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_12 import TUPLE_GETITEM_UNBIND_F32_12
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_7 import TUPLE_GETITEM_UNBIND_F32_7
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_8 import TUPLE_GETITEM_UNBIND_F32_8
from models.quantized_klein9b.tensors.flux_double_block import FluxDoubleBlockTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_double_block_with_tensors(rt: RuntimeSession, tensors: FluxDoubleBlockTensors) -> None:
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.img, output=tensors.layer_norm)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.img_mod1_scale, output=tensors.add)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add, y=tensors.layer_norm, output=tensors.mul)
    rt.release_frame_workspace(tensors.add)
    rt.release_frame_workspace(tensors.layer_norm)
    ADD_BROADCAST_INNER(rt, x=tensors.mul, y=tensors.img_mod1_shift, output=tensors.add_1)
    rt.release_frame_workspace(tensors.mul)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_1, weight=tensors.p_img_attn_qkv_weight, output=tensors.linear)
    rt.release_frame_workspace(tensors.add_1)
    PERMUTE_F32_0D3AB17AE3(rt, x=tensors.reshape, output=tensors.permute)
    rt.release_frame_workspace(tensors.linear)
    TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute, output=tensors.getitem)
    TUPLE_GETITEM_UNBIND_F32_7(rt, x=tensors.permute, output=tensors.getitem_1)
    TUPLE_GETITEM_UNBIND_F32_8(rt, x=tensors.permute, output=tensors.getitem_2)
    rt.release_frame_workspace(tensors.permute)
    RMS_NORM_F32W_F32(rt, x=tensors.getitem, weight=tensors.p_img_attn_norm_query_norm_scale, output=tensors.rms_norm)
    rt.release_frame_workspace(tensors.getitem)
    RMS_NORM_F32W_F32(rt, x=tensors.getitem_1, weight=tensors.p_img_attn_norm_key_norm_scale, output=tensors.rms_norm_1)
    rt.release_frame_workspace(tensors.getitem_1)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.txt, output=tensors.layer_norm_1)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.txt_mod1_scale, output=tensors.add_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_2, y=tensors.layer_norm_1, output=tensors.mul_1)
    rt.release_frame_workspace(tensors.add_2)
    rt.release_frame_workspace(tensors.layer_norm_1)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_1, y=tensors.txt_mod1_shift, output=tensors.add_3)
    rt.release_frame_workspace(tensors.mul_1)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_3, weight=tensors.p_txt_attn_qkv_weight, output=tensors.linear_1)
    rt.release_frame_workspace(tensors.add_3)
    PERMUTE_F32_8475D3A978(rt, x=tensors.reshape_1, output=tensors.permute_1)
    rt.release_frame_workspace(tensors.linear_1)
    TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute_1, output=tensors.getitem_3)
    TUPLE_GETITEM_UNBIND_F32_11(rt, x=tensors.permute_1, output=tensors.getitem_4)
    TUPLE_GETITEM_UNBIND_F32_12(rt, x=tensors.permute_1, output=tensors.getitem_5)
    rt.release_frame_workspace(tensors.permute_1)
    RMS_NORM_F32W_F32(rt, x=tensors.getitem_3, weight=tensors.p_txt_attn_norm_query_norm_scale, output=tensors.rms_norm_2)
    rt.release_frame_workspace(tensors.getitem_3)
    RMS_NORM_F32W_F32(rt, x=tensors.getitem_4, weight=tensors.p_txt_attn_norm_key_norm_scale, output=tensors.rms_norm_3)
    rt.release_frame_workspace(tensors.getitem_4)
    FLUX_DOUBLE_BLOCK_CAT_2_F32(rt, x0=tensors.pe_ctx, x1=tensors.pe, output=tensors.cat)
    CAT_2_F32_14(rt, x0=tensors.to_2, x1=tensors.to, output=tensors.cat_1)
    rt.release_frame_workspace(tensors.rms_norm)
    rt.release_frame_workspace(tensors.rms_norm_2)
    CAT_2_F32_15(rt, x0=tensors.to_3, x1=tensors.to_1, output=tensors.cat_2)
    rt.release_frame_workspace(tensors.rms_norm_1)
    rt.release_frame_workspace(tensors.rms_norm_3)
    CAT_2_F32_16(rt, x0=tensors.getitem_5, x1=tensors.getitem_2, output=tensors.cat_3)
    rt.release_frame_workspace(tensors.getitem_2)
    rt.release_frame_workspace(tensors.getitem_5)
    SELECT_FLOAT32(rt, x=tensors.cat, output=tensors.select)
    SELECT_FLOAT32(rt, x=tensors.reshape_2, output=tensors.select_1)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST_18(rt, x=tensors.select, y=tensors.select_1, output=tensors.mul_2)
    rt.release_frame_workspace(tensors.select)
    rt.release_frame_workspace(tensors.select_1)
    SELECT_FLOAT32_19(rt, x=tensors.cat, output=tensors.select_2)
    SELECT_FLOAT32_20(rt, x=tensors.reshape_2, output=tensors.select_3)
    rt.release_frame_workspace(tensors.cat_1)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST_21(rt, x=tensors.select_2, y=tensors.select_3, output=tensors.mul_3)
    rt.release_frame_workspace(tensors.select_2)
    rt.release_frame_workspace(tensors.select_3)
    FLUX_DOUBLE_BLOCK_ADD_F32(rt, x=tensors.mul_2, y=tensors.mul_3, output=tensors.add_4)
    rt.release_frame_workspace(tensors.mul_2)
    rt.release_frame_workspace(tensors.mul_3)
    SELECT_FLOAT32(rt, x=tensors.cat, output=tensors.select_4)
    SELECT_FLOAT32(rt, x=tensors.reshape_3, output=tensors.select_5)
    MUL_BROADCAST_23(rt, x=tensors.select_4, y=tensors.select_5, output=tensors.mul_4)
    rt.release_frame_workspace(tensors.select_4)
    rt.release_frame_workspace(tensors.select_5)
    SELECT_FLOAT32_24(rt, x=tensors.cat, output=tensors.select_6)
    rt.release_frame_workspace(tensors.cat)
    SELECT_FLOAT32_25(rt, x=tensors.reshape_3, output=tensors.select_7)
    rt.release_frame_workspace(tensors.cat_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST_26(rt, x=tensors.select_6, y=tensors.select_7, output=tensors.mul_5)
    rt.release_frame_workspace(tensors.select_6)
    rt.release_frame_workspace(tensors.select_7)
    FLUX_DOUBLE_BLOCK_ADD_F32(rt, x=tensors.mul_4, y=tensors.mul_5, output=tensors.add_5)
    rt.release_frame_workspace(tensors.mul_4)
    rt.release_frame_workspace(tensors.mul_5)
    SDPA_F32(rt, q=tensors.type_as, k=tensors.type_as_1, v=tensors.cat_3, output=tensors.scaled_dot_product_attention)
    rt.release_frame_workspace(tensors.add_4)
    rt.release_frame_workspace(tensors.add_5)
    rt.release_frame_workspace(tensors.cat_3)
    PERMUTE_F32_7EBE673EB3(rt, x=tensors.scaled_dot_product_attention, output=tensors.permute_2)
    rt.release_frame_workspace(tensors.scaled_dot_product_attention)
    FLUX_DOUBLE_BLOCK_SLICE_F32(rt, x=tensors.reshape_6, output=tensors.slice_1)
    SLICE_F32_30(rt, x=tensors.reshape_6, output=tensors.slice_2)
    rt.release_frame_workspace(tensors.permute_2)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.slice_2, weight=tensors.p_img_attn_proj_weight, output=tensors.linear_2)
    rt.release_frame_workspace(tensors.slice_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.img_mod1_gate, y=tensors.linear_2, output=tensors.mul_6)
    rt.release_frame_workspace(tensors.linear_2)
    ADD_F32_31(rt, x=tensors.img, y=tensors.mul_6, output=tensors.add_6)
    rt.release_frame_workspace(tensors.mul_6)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.img_mod2_scale, output=tensors.add_7)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.add_6, output=tensors.layer_norm_2)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_7, y=tensors.layer_norm_2, output=tensors.mul_7)
    rt.release_frame_workspace(tensors.add_7)
    rt.release_frame_workspace(tensors.layer_norm_2)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_7, y=tensors.img_mod2_shift, output=tensors.add_8)
    rt.release_frame_workspace(tensors.mul_7)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_8, weight=tensors.p_img_mlp_0_weight, output=tensors.linear_3)
    rt.release_frame_workspace(tensors.add_8)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear_3, output=tensors.getitem_6)
    TUPLE_GETITEM_SLICE_F32_33(rt, x=tensors.linear_3, output=tensors.getitem_7)
    rt.release_frame_workspace(tensors.linear_3)
    FLUX_DOUBLE_BLOCK_SILU_F32(rt, x=tensors.getitem_6, output=tensors.silu)
    rt.release_frame_workspace(tensors.getitem_6)
    FLUX_DOUBLE_BLOCK_MUL_F32(rt, x=tensors.silu, y=tensors.getitem_7, output=tensors.mul_8)
    rt.release_frame_workspace(tensors.getitem_7)
    rt.release_frame_workspace(tensors.silu)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.mul_8, weight=tensors.p_img_mlp_2_weight, output=tensors.linear_4)
    rt.release_frame_workspace(tensors.mul_8)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.img_mod2_gate, y=tensors.linear_4, output=tensors.mul_9)
    rt.release_frame_workspace(tensors.linear_4)
    FLUX_DOUBLE_BLOCK_ADD_F32_36(rt, x=tensors.add_6, y=tensors.mul_9, output=tensors.add_9)
    rt.release_frame_workspace(tensors.add_6)
    rt.release_frame_workspace(tensors.mul_9)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.slice_1, weight=tensors.p_txt_attn_proj_weight, output=tensors.linear_5)
    rt.release_frame_workspace(tensors.slice_1)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.txt_mod1_gate, y=tensors.linear_5, output=tensors.mul_10)
    rt.release_frame_workspace(tensors.linear_5)
    ADD_F32_37(rt, x=tensors.txt, y=tensors.mul_10, output=tensors.add_10)
    rt.release_frame_workspace(tensors.mul_10)
    FLUX_DOUBLE_BLOCK_ADD_SCALAR(rt, x=tensors.txt_mod2_scale, output=tensors.add_11)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.add_10, output=tensors.layer_norm_3)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add_11, y=tensors.layer_norm_3, output=tensors.mul_11)
    rt.release_frame_workspace(tensors.add_11)
    rt.release_frame_workspace(tensors.layer_norm_3)
    ADD_BROADCAST_INNER(rt, x=tensors.mul_11, y=tensors.txt_mod2_shift, output=tensors.add_12)
    rt.release_frame_workspace(tensors.mul_11)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_12, weight=tensors.p_txt_mlp_0_weight, output=tensors.linear_6)
    rt.release_frame_workspace(tensors.add_12)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear_6, output=tensors.getitem_8)
    TUPLE_GETITEM_SLICE_F32_38(rt, x=tensors.linear_6, output=tensors.getitem_9)
    rt.release_frame_workspace(tensors.linear_6)
    FLUX_DOUBLE_BLOCK_SILU_F32(rt, x=tensors.getitem_8, output=tensors.silu_1)
    rt.release_frame_workspace(tensors.getitem_8)
    FLUX_DOUBLE_BLOCK_MUL_F32(rt, x=tensors.silu_1, y=tensors.getitem_9, output=tensors.mul_12)
    rt.release_frame_workspace(tensors.getitem_9)
    rt.release_frame_workspace(tensors.silu_1)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.mul_12, weight=tensors.p_txt_mlp_2_weight, output=tensors.linear_7)
    rt.release_frame_workspace(tensors.mul_12)
    FLUX_DOUBLE_BLOCK_MUL_BROADCAST(rt, x=tensors.txt_mod2_gate, y=tensors.linear_7, output=tensors.mul_13)
    rt.release_frame_workspace(tensors.linear_7)
    ADD_F32_39(rt, x=tensors.add_10, y=tensors.mul_13, output=tensors.add_13)
    rt.release_frame_workspace(tensors.add_10)
    rt.release_frame_workspace(tensors.mul_13)


def run_flux_double_block(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors().flux_double_blocks[layer_idx]
    _run_flux_double_block_with_tensors(rt, tensors)
