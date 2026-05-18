"""Generated dispatch function for run_decode_layer."""

from __future__ import annotations

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.exported_qwen3_asr.shaders.add_f32_16 import ADD_F32_16
from models.exported_qwen3_asr.shaders.add_f32_19 import ADD_F32_19
from models.exported_qwen3_asr.shaders.cat_2_f32 import CAT_2_F32
from models.exported_qwen3_asr.shaders.decode_layer_add_f32 import DECODE_LAYER_ADD_F32
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast import DECODE_LAYER_MUL_BROADCAST
from models.exported_qwen3_asr.shaders.kv_cache_write_decode_f16 import KV_CACHE_WRITE_DECODE_F16
from models.exported_qwen3_asr.shaders.linear_nobias_bf16w_f32 import LINEAR_NOBIAS_BF16W_F32
from models.exported_qwen3_asr.shaders.mul_f32 import MUL_F32
from models.exported_qwen3_asr.shaders.neg_f32 import NEG_F32
from models.exported_qwen3_asr.shaders.rms_norm_bf16w_f32 import RMS_NORM_BF16W_F32
from models.exported_qwen3_asr.shaders.rms_norm_bf16w_f32_2 import RMS_NORM_BF16W_F32_2
from models.exported_qwen3_asr.shaders.rms_norm_bf16w_f32_4 import RMS_NORM_BF16W_F32_4
from models.exported_qwen3_asr.shaders.sdpa_decode_cache_f16 import SDPA_DECODE_CACHE_F16
from models.exported_qwen3_asr.shaders.silu_f32 import SILU_F32
from models.exported_qwen3_asr.shaders.slice_f32 import SLICE_F32
from models.exported_qwen3_asr.shaders.slice_f32_12 import SLICE_F32_12
from models.exported_qwen3_asr.shaders.slice_f32_8 import SLICE_F32_8
from models.exported_qwen3_asr.shaders.transpose_f32_9884b7b82d import TRANSPOSE_F32_9884B7B82D
from models.exported_qwen3_asr.shaders.transpose_f32_9e77b1cee2 import TRANSPOSE_F32_9E77B1CEE2
from models.exported_qwen3_asr.shaders.transpose_f32_d509518b4f import TRANSPOSE_F32_D509518B4F
from models.exported_qwen3_asr.tensors.decode_layer import DecodeLayerTensors
from torch2vk.runtime.session import RuntimeSession


def _run_decode_layer_with_tensors(rt: RuntimeSession, tensors: DecodeLayerTensors, *, cache_position: int) -> None:
    RMS_NORM_BF16W_F32(rt, x=tensors.hidden_states, weight=tensors.p_input_layernorm_weight, output=tensors.rms_norm)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.rms_norm, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    RMS_NORM_BF16W_F32_2(rt, x=tensors.view, weight=tensors.p_attn_q_norm_weight, output=tensors.rms_norm_1)
    rt.release_frame_workspace(tensors.linear)
    TRANSPOSE_F32_9884B7B82D(rt, x=tensors.rms_norm_1, output=tensors.transpose)
    rt.release_frame_workspace(tensors.rms_norm_1)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.rms_norm, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    RMS_NORM_BF16W_F32_4(rt, x=tensors.view_1, weight=tensors.p_attn_k_norm_weight, output=tensors.rms_norm_2)
    rt.release_frame_workspace(tensors.linear_1)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.rms_norm_2, output=tensors.transpose_1)
    rt.release_frame_workspace(tensors.rms_norm_2)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.rms_norm, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    rt.release_frame_workspace(tensors.rms_norm)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.view_2, output=tensors.transpose_2)
    rt.release_frame_workspace(tensors.linear_2)
    DECODE_LAYER_MUL_BROADCAST(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul)
    SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    SLICE_F32_8(rt, x=tensors.transpose, output=tensors.slice_2)
    rt.release_frame_workspace(tensors.transpose)
    NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    rt.release_frame_workspace(tensors.slice_2)
    CAT_2_F32(rt, x0=tensors.neg, x1=tensors.slice_1, output=tensors.cat)
    rt.release_frame_workspace(tensors.neg)
    rt.release_frame_workspace(tensors.slice_1)
    DECODE_LAYER_MUL_BROADCAST(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_1)
    rt.release_frame_workspace(tensors.cat)
    DECODE_LAYER_ADD_F32(rt, x=tensors.mul, y=tensors.mul_1, output=tensors.add)
    rt.release_frame_workspace(tensors.mul)
    rt.release_frame_workspace(tensors.mul_1)
    DECODE_LAYER_MUL_BROADCAST(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_2)
    SLICE_F32(rt, x=tensors.transpose_1, output=tensors.slice_3)
    SLICE_F32_12(rt, x=tensors.transpose_1, output=tensors.slice_4)
    rt.release_frame_workspace(tensors.transpose_1)
    NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    rt.release_frame_workspace(tensors.slice_4)
    CAT_2_F32(rt, x0=tensors.neg_1, x1=tensors.slice_3, output=tensors.cat_1)
    rt.release_frame_workspace(tensors.neg_1)
    rt.release_frame_workspace(tensors.slice_3)
    DECODE_LAYER_MUL_BROADCAST(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_3)
    rt.release_frame_workspace(tensors.cat_1)
    DECODE_LAYER_ADD_F32(rt, x=tensors.mul_2, y=tensors.mul_3, output=tensors.add_1)
    rt.release_frame_workspace(tensors.mul_2)
    rt.release_frame_workspace(tensors.mul_3)
    KV_CACHE_WRITE_DECODE_F16(rt, cache=tensors.index_copy, src=tensors.add_1, cache_position=cache_position)
    rt.release_frame_workspace(tensors.add_1)
    KV_CACHE_WRITE_DECODE_F16(rt, cache=tensors.index_copy_1, src=tensors.transpose_2, cache_position=cache_position)
    rt.release_frame_workspace(tensors.transpose_2)
    SDPA_DECODE_CACHE_F16(rt, q=tensors.add, k=tensors.index_copy, v=tensors.index_copy_1, output=tensors.scaled_dot_product_attention, cache_position=cache_position)
    rt.release_frame_workspace(tensors.add)
    TRANSPOSE_F32_9E77B1CEE2(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    rt.release_frame_workspace(tensors.scaled_dot_product_attention)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    rt.release_frame_workspace(tensors.transpose_3)
    ADD_F32_16(rt, x=tensors.hidden_states, y=tensors.linear_3, output=tensors.add_2)
    rt.release_frame_workspace(tensors.linear_3)
    RMS_NORM_BF16W_F32(rt, x=tensors.add_2, weight=tensors.p_post_attention_layernorm_weight, output=tensors.rms_norm_3)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.rms_norm_3, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    rt.release_frame_workspace(tensors.linear_4)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.rms_norm_3, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    rt.release_frame_workspace(tensors.rms_norm_3)
    MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_4)
    rt.release_frame_workspace(tensors.linear_5)
    rt.release_frame_workspace(tensors.silu)
    LINEAR_NOBIAS_BF16W_F32(rt, x=tensors.mul_4, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    rt.release_frame_workspace(tensors.mul_4)
    ADD_F32_19(rt, x=tensors.add_2, y=tensors.linear_6, output=tensors.add_3)
    rt.release_frame_workspace(tensors.add_2)
    rt.release_frame_workspace(tensors.linear_6)


def run_decode_layer(rt: RuntimeSession, layer_idx: int, *, cache_position: int) -> None:
    tensors = model_tensors().decode_layers[layer_idx]
    _run_decode_layer_with_tensors(rt, tensors, cache_position=cache_position)
