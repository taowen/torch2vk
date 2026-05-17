"""Generated dispatch function for run_audio_encoder."""

from __future__ import annotations

from models.quantized_qwen3_asr.tensors.model import model_tensors
from models.quantized_qwen3_asr.shaders.add_f32 import ADD_F32
from models.quantized_qwen3_asr.shaders.add_f32_17 import ADD_F32_17
from models.quantized_qwen3_asr.shaders.add_f32_19 import ADD_F32_19
from models.quantized_qwen3_asr.shaders.conv2d_q8_0w_f32b_f16 import CONV2D_Q8_0W_F32B_F16
from models.quantized_qwen3_asr.shaders.conv2d_q8_0w_f32b_f16_2 import CONV2D_Q8_0W_F32B_F16_2
from models.quantized_qwen3_asr.shaders.conv2d_q8_0w_f32b_f16_3 import CONV2D_Q8_0W_F32B_F16_3
from models.quantized_qwen3_asr.shaders.gelu_f32 import GELU_F32
from models.quantized_qwen3_asr.shaders.gelu_f32_18 import GELU_F32_18
from models.quantized_qwen3_asr.shaders.gelu_f32_88 import GELU_F32_88
from models.quantized_qwen3_asr.shaders.index_select_f32_c6680f8d95 import INDEX_SELECT_F32_C6680F8D95
from models.quantized_qwen3_asr.shaders.layer_norm_f32w_f32b_f32 import LAYER_NORM_F32W_F32B_F32
from models.quantized_qwen3_asr.shaders.linear_bias_q8_0w_f32b_f32 import LINEAR_BIAS_Q8_0W_F32B_F32
from models.quantized_qwen3_asr.shaders.linear_nobias_q8_0_f32 import LINEAR_NOBIAS_Q8_0_F32
from models.quantized_qwen3_asr.shaders.sdpa_masked_f32 import SDPA_MASKED_F32
from models.quantized_qwen3_asr.shaders.transpose_f32_48d16b9b88 import TRANSPOSE_F32_48D16B9B88
from models.quantized_qwen3_asr.shaders.transpose_f32_6a3397f037 import TRANSPOSE_F32_6A3397F037
from models.quantized_qwen3_asr.shaders.transpose_f32_d95ce920ac import TRANSPOSE_F32_D95CE920AC
from models.quantized_qwen3_asr.tensors.audio_encoder import AudioEncoderTensors
from torch2vk.runtime.session import RuntimeSession


def _run_audio_encoder_with_tensors(rt: RuntimeSession, tensors: AudioEncoderTensors) -> None:
    CONV2D_Q8_0W_F32B_F16(rt, x=tensors.x, weight=tensors.p_audio_tower_conv2d1_weight, bias=tensors.p_audio_tower_conv2d1_bias, output=tensors.conv2d)
    GELU_F32(rt, x=tensors.conv2d, output=tensors.gelu)
    CONV2D_Q8_0W_F32B_F16_2(rt, x=tensors.gelu, weight=tensors.p_audio_tower_conv2d2_weight, bias=tensors.p_audio_tower_conv2d2_bias, output=tensors.conv2d_1)
    GELU_F32(rt, x=tensors.conv2d_1, output=tensors.gelu_1)
    CONV2D_Q8_0W_F32B_F16_3(rt, x=tensors.gelu_1, weight=tensors.p_audio_tower_conv2d3_weight, bias=tensors.p_audio_tower_conv2d3_bias, output=tensors.conv2d_2)
    GELU_F32(rt, x=tensors.conv2d_2, output=tensors.gelu_2)
    TRANSPOSE_F32_D95CE920AC(rt, x=tensors.reshape, output=tensors.transpose)
    LINEAR_NOBIAS_Q8_0_F32(rt, x=tensors.transpose, weight=tensors.p_audio_tower_conv_out_weight, output=tensors.linear)
    ADD_F32(rt, x=tensors.linear, y=tensors.position_embedding, output=tensors.add)
    INDEX_SELECT_F32_C6680F8D95(rt, x=tensors.reshape_1, index=tensors.compact_index, output=tensors.index_select)
    carry = tensors.index_select
    for layer_t in tensors.layers:
        LAYER_NORM_F32W_F32B_F32(rt, x=carry, weight=layer_t.p_audio_tower_layers_0_self_attn_layer_norm_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_layer_norm_bias, output=layer_t.layer_norm)
        LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=layer_t.layer_norm, weight=layer_t.p_audio_tower_layers_0_self_attn_q_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_q_proj_bias, output=layer_t.linear_1)
        LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=layer_t.layer_norm, weight=layer_t.p_audio_tower_layers_0_self_attn_k_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_k_proj_bias, output=layer_t.linear_2)
        LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=layer_t.layer_norm, weight=layer_t.p_audio_tower_layers_0_self_attn_v_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_v_proj_bias, output=layer_t.linear_3)
        TRANSPOSE_F32_6A3397F037(rt, x=layer_t.reshape_2, output=layer_t.transpose_1)
        TRANSPOSE_F32_6A3397F037(rt, x=layer_t.reshape_3, output=layer_t.transpose_2)
        TRANSPOSE_F32_6A3397F037(rt, x=layer_t.reshape_4, output=layer_t.transpose_3)
        SDPA_MASKED_F32(rt, q=layer_t.unsqueeze, k=layer_t.unsqueeze_1, v=layer_t.unsqueeze_2, mask=tensors.attention_mask, output=layer_t.scaled_dot_product_attention)
        TRANSPOSE_F32_48D16B9B88(rt, x=layer_t.scaled_dot_product_attention, output=layer_t.transpose_4)
        LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=layer_t.reshape_5, weight=layer_t.p_audio_tower_layers_0_self_attn_out_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_out_proj_bias, output=layer_t.linear_4)
        ADD_F32_17(rt, x=carry, y=layer_t.linear_4, output=layer_t.add_1)
        LAYER_NORM_F32W_F32B_F32(rt, x=layer_t.add_1, weight=layer_t.p_audio_tower_layers_0_final_layer_norm_weight, bias=layer_t.p_audio_tower_layers_0_final_layer_norm_bias, output=layer_t.layer_norm_1)
        LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=layer_t.layer_norm_1, weight=layer_t.p_audio_tower_layers_0_fc1_weight, bias=layer_t.p_audio_tower_layers_0_fc1_bias, output=layer_t.linear_5)
        GELU_F32_18(rt, x=layer_t.linear_5, output=layer_t.gelu_3)
        LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=layer_t.gelu_3, weight=layer_t.p_audio_tower_layers_0_fc2_weight, bias=layer_t.p_audio_tower_layers_0_fc2_bias, output=layer_t.linear_6)
        ADD_F32_19(rt, x=layer_t.add_1, y=layer_t.linear_6, output=layer_t.add_2)
        carry = layer_t.add_2
        rt.release_layer_workspace(layer_t, layer=layer_t.add_2.layer or "", keep=(layer_t.add_2,))
    LAYER_NORM_F32W_F32B_F32(rt, x=carry, weight=tensors.p_audio_tower_ln_post_weight, bias=tensors.p_audio_tower_ln_post_bias, output=tensors.layer_norm_36)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.layer_norm_36, weight=tensors.p_audio_tower_proj1_weight, bias=tensors.p_audio_tower_proj1_bias, output=tensors.linear_109)
    GELU_F32_88(rt, x=tensors.linear_109, output=tensors.gelu_21)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.gelu_21, weight=tensors.p_audio_tower_proj2_weight, bias=tensors.p_audio_tower_proj2_bias, output=tensors.linear_110)


def run_audio_encoder(rt: RuntimeSession) -> None:
    tensors = model_tensors().audio_encoder
    _run_audio_encoder_with_tensors(rt, tensors)
