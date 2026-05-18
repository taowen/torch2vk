"""Generated dispatch function for run_flux_single_block."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_broadcast_inner import ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.add_f32_28 import ADD_F32_28
from models.quantized_klein9b.shaders.flux_single_block_add_f32 import FLUX_SINGLE_BLOCK_ADD_F32
from models.quantized_klein9b.shaders.flux_single_block_add_scalar import FLUX_SINGLE_BLOCK_ADD_SCALAR
from models.quantized_klein9b.shaders.flux_single_block_cat_2_f32 import FLUX_SINGLE_BLOCK_CAT_2_F32
from models.quantized_klein9b.shaders.flux_single_block_mul_broadcast import FLUX_SINGLE_BLOCK_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_single_block_mul_broadcast_18 import FLUX_SINGLE_BLOCK_MUL_BROADCAST_18
from models.quantized_klein9b.shaders.flux_single_block_mul_broadcast_21 import FLUX_SINGLE_BLOCK_MUL_BROADCAST_21
from models.quantized_klein9b.shaders.flux_single_block_mul_f32 import FLUX_SINGLE_BLOCK_MUL_F32
from models.quantized_klein9b.shaders.flux_single_block_silu_f32 import FLUX_SINGLE_BLOCK_SILU_F32
from models.quantized_klein9b.shaders.layer_norm_nonew_noneb_f32 import LAYER_NORM_NONEW_NONEB_F32
from models.quantized_klein9b.shaders.linear_nobias_q8_0_f32_act_f32 import LINEAR_NOBIAS_Q8_0_F32_ACT_F32
from models.quantized_klein9b.shaders.mul_broadcast_13 import MUL_BROADCAST_13
from models.quantized_klein9b.shaders.mul_broadcast_16 import MUL_BROADCAST_16
from models.quantized_klein9b.shaders.permute_f32_2731f610b1 import PERMUTE_F32_2731F610B1
from models.quantized_klein9b.shaders.permute_f32_7ebe673eb3 import PERMUTE_F32_7EBE673EB3
from models.quantized_klein9b.shaders.rms_norm_f32w_f32 import RMS_NORM_F32W_F32
from models.quantized_klein9b.shaders.sdpa_f32 import SDPA_F32
from models.quantized_klein9b.shaders.select_float32 import SELECT_FLOAT32
from models.quantized_klein9b.shaders.select_float32_14 import SELECT_FLOAT32_14
from models.quantized_klein9b.shaders.select_float32_15 import SELECT_FLOAT32_15
from models.quantized_klein9b.shaders.select_float32_19 import SELECT_FLOAT32_19
from models.quantized_klein9b.shaders.select_float32_20 import SELECT_FLOAT32_20
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32 import TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_24 import TUPLE_GETITEM_SLICE_F32_24
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_6 import TUPLE_GETITEM_SLICE_F32_6
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32 import TUPLE_GETITEM_UNBIND_F32
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_10 import TUPLE_GETITEM_UNBIND_F32_10
from models.quantized_klein9b.shaders.tuple_getitem_unbind_f32_9 import TUPLE_GETITEM_UNBIND_F32_9
from models.quantized_klein9b.tensors.flux_single_block import FluxSingleBlockTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_single_block_with_tensors(rt: RuntimeSession, tensors: FluxSingleBlockTensors) -> None:
    FLUX_SINGLE_BLOCK_ADD_SCALAR(rt, x=tensors.mod_scale, output=tensors.add)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.hidden_states, output=tensors.layer_norm)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST(rt, x=tensors.add, y=tensors.layer_norm, output=tensors.mul)
    rt.release_frame_workspace(tensors.add)
    rt.release_frame_workspace(tensors.layer_norm)
    ADD_BROADCAST_INNER(rt, x=tensors.mul, y=tensors.mod_shift, output=tensors.add_1)
    rt.release_frame_workspace(tensors.mul)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.add_1, weight=tensors.p_linear1_weight, output=tensors.linear)
    rt.release_frame_workspace(tensors.add_1)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear, output=tensors.getitem)
    TUPLE_GETITEM_SLICE_F32_6(rt, x=tensors.linear, output=tensors.getitem_1)
    rt.release_frame_workspace(tensors.linear)
    PERMUTE_F32_2731F610B1(rt, x=tensors.reshape, output=tensors.permute)
    rt.release_frame_workspace(tensors.getitem)
    TUPLE_GETITEM_UNBIND_F32(rt, x=tensors.permute, output=tensors.getitem_2)
    TUPLE_GETITEM_UNBIND_F32_9(rt, x=tensors.permute, output=tensors.getitem_3)
    TUPLE_GETITEM_UNBIND_F32_10(rt, x=tensors.permute, output=tensors.getitem_4)
    rt.release_frame_workspace(tensors.permute)
    RMS_NORM_F32W_F32(rt, x=tensors.getitem_2, weight=tensors.p_norm_query_norm_scale, output=tensors.rms_norm)
    rt.release_frame_workspace(tensors.getitem_2)
    RMS_NORM_F32W_F32(rt, x=tensors.getitem_3, weight=tensors.p_norm_key_norm_scale, output=tensors.rms_norm_1)
    rt.release_frame_workspace(tensors.getitem_3)
    SELECT_FLOAT32(rt, x=tensors.pe, output=tensors.select)
    SELECT_FLOAT32(rt, x=tensors.reshape_1, output=tensors.select_1)
    MUL_BROADCAST_13(rt, x=tensors.select, y=tensors.select_1, output=tensors.mul_1)
    rt.release_frame_workspace(tensors.select)
    rt.release_frame_workspace(tensors.select_1)
    SELECT_FLOAT32_14(rt, x=tensors.pe, output=tensors.select_2)
    SELECT_FLOAT32_15(rt, x=tensors.reshape_1, output=tensors.select_3)
    rt.release_frame_workspace(tensors.rms_norm)
    MUL_BROADCAST_16(rt, x=tensors.select_2, y=tensors.select_3, output=tensors.mul_2)
    rt.release_frame_workspace(tensors.select_2)
    rt.release_frame_workspace(tensors.select_3)
    FLUX_SINGLE_BLOCK_ADD_F32(rt, x=tensors.mul_1, y=tensors.mul_2, output=tensors.add_2)
    rt.release_frame_workspace(tensors.mul_1)
    rt.release_frame_workspace(tensors.mul_2)
    SELECT_FLOAT32(rt, x=tensors.pe, output=tensors.select_4)
    SELECT_FLOAT32(rt, x=tensors.reshape_2, output=tensors.select_5)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST_18(rt, x=tensors.select_4, y=tensors.select_5, output=tensors.mul_3)
    rt.release_frame_workspace(tensors.select_4)
    rt.release_frame_workspace(tensors.select_5)
    SELECT_FLOAT32_19(rt, x=tensors.pe, output=tensors.select_6)
    SELECT_FLOAT32_20(rt, x=tensors.reshape_2, output=tensors.select_7)
    rt.release_frame_workspace(tensors.rms_norm_1)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST_21(rt, x=tensors.select_6, y=tensors.select_7, output=tensors.mul_4)
    rt.release_frame_workspace(tensors.select_6)
    rt.release_frame_workspace(tensors.select_7)
    FLUX_SINGLE_BLOCK_ADD_F32(rt, x=tensors.mul_3, y=tensors.mul_4, output=tensors.add_3)
    rt.release_frame_workspace(tensors.mul_3)
    rt.release_frame_workspace(tensors.mul_4)
    SDPA_F32(rt, q=tensors.type_as, k=tensors.type_as_1, v=tensors.getitem_4, output=tensors.scaled_dot_product_attention)
    rt.release_frame_workspace(tensors.add_2)
    rt.release_frame_workspace(tensors.add_3)
    rt.release_frame_workspace(tensors.getitem_4)
    PERMUTE_F32_7EBE673EB3(rt, x=tensors.scaled_dot_product_attention, output=tensors.permute_1)
    rt.release_frame_workspace(tensors.scaled_dot_product_attention)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.getitem_1, output=tensors.getitem_5)
    TUPLE_GETITEM_SLICE_F32_24(rt, x=tensors.getitem_1, output=tensors.getitem_6)
    rt.release_frame_workspace(tensors.getitem_1)
    FLUX_SINGLE_BLOCK_SILU_F32(rt, x=tensors.getitem_5, output=tensors.silu)
    rt.release_frame_workspace(tensors.getitem_5)
    FLUX_SINGLE_BLOCK_MUL_F32(rt, x=tensors.silu, y=tensors.getitem_6, output=tensors.mul_5)
    rt.release_frame_workspace(tensors.getitem_6)
    rt.release_frame_workspace(tensors.silu)
    FLUX_SINGLE_BLOCK_CAT_2_F32(rt, x0=tensors.reshape_5, x1=tensors.mul_5, output=tensors.cat)
    rt.release_frame_workspace(tensors.mul_5)
    rt.release_frame_workspace(tensors.permute_1)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.cat, weight=tensors.p_linear2_weight, output=tensors.linear_1)
    rt.release_frame_workspace(tensors.cat)
    FLUX_SINGLE_BLOCK_MUL_BROADCAST(rt, x=tensors.mod_gate, y=tensors.linear_1, output=tensors.mul_6)
    rt.release_frame_workspace(tensors.linear_1)
    ADD_F32_28(rt, x=tensors.hidden_states, y=tensors.mul_6, output=tensors.add_4)
    rt.release_frame_workspace(tensors.mul_6)


def run_flux_single_block(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors().flux_single_blocks[layer_idx]
    _run_flux_single_block_with_tensors(rt, tensors)
