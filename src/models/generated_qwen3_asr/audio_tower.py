"""Generated generated_qwen3_asr.audio_tower frame scaffold."""

from __future__ import annotations

from torch2vk.runtime.session import RuntimeSession
from torch2vk.export.shaders import (
    ADD_F32,
    ADD_POSITION_F32,
    COMPACT_AFTER_CNN_F32,
    CONV2D_GELU_F32,
    CONV_OUT_F32,
    CU_SEQLENS_U32,
    ENCODER_ATTENTION_F32,
    LAYER_NORM_F32,
    LINEAR_F32,
    LINEAR_GELU_F32,
    PAD_FEATURE_F32,
)
from models.generated_qwen3_asr.tensors.audio_tower import GeneratedQwen3AsrAudioTowerTensors


def run_generated_qwen3_asr_audio_tower(
    rt: RuntimeSession,
    tensors: GeneratedQwen3AsrAudioTowerTensors,
    **_kw,
) -> None:
    with rt.frame("generated_qwen3_asr.audio_tower"):
        PAD_FEATURE_F32(
            rt,
            input_features=tensors.input_features,
            feature_lens=tensors.feature_lens,
            output=tensors.padded_feature,
        )
        CONV2D_GELU_F32(
            rt,
            x=tensors.padded_feature,
            weight=tensors.conv2d1_weight,
            bias=tensors.conv2d1_bias,
            output=tensors.conv2d1_gelu,
        )
        CONV2D_GELU_F32(
            rt,
            x=tensors.conv2d1_gelu,
            weight=tensors.conv2d2_weight,
            bias=tensors.conv2d2_bias,
            output=tensors.conv2d2_gelu,
        )
        CONV2D_GELU_F32(
            rt,
            x=tensors.conv2d2_gelu,
            weight=tensors.conv2d3_weight,
            bias=tensors.conv2d3_bias,
            output=tensors.conv2d3_gelu,
        )
        CONV_OUT_F32(
            rt, x=tensors.conv2d3_gelu, weight=tensors.conv_out_weight, output=tensors.conv_out
        )
        ADD_POSITION_F32(rt, x=tensors.conv_out, output=tensors.conv_out_add_position)
        COMPACT_AFTER_CNN_F32(
            rt,
            x=tensors.conv_out_add_position,
            feature_lens=tensors.feature_lens,
            output=tensors.hidden_states,
        )
        CU_SEQLENS_U32(rt, feature_lens=tensors.feature_lens, output=tensors.cu_seqlens)
        hidden_states = tensors.hidden_states
        for layer in tensors.layers:
            LAYER_NORM_F32(
                rt,
                x=hidden_states,
                weight=layer.self_attn_layer_norm_weight,
                bias=layer.self_attn_layer_norm_bias,
                output=layer.self_attn_layer_norm,
            )
            LINEAR_F32(
                rt,
                x=layer.self_attn_layer_norm,
                weight=layer.q_proj_weight,
                bias=layer.q_proj_bias,
                output=layer.q_proj,
            )
            LINEAR_F32(
                rt,
                x=layer.self_attn_layer_norm,
                weight=layer.k_proj_weight,
                bias=layer.k_proj_bias,
                output=layer.k_proj,
            )
            LINEAR_F32(
                rt,
                x=layer.self_attn_layer_norm,
                weight=layer.v_proj_weight,
                bias=layer.v_proj_bias,
                output=layer.v_proj,
            )
            ENCODER_ATTENTION_F32(
                rt,
                q=layer.q_proj,
                k=layer.k_proj,
                v=layer.v_proj,
                cu_seqlens=tensors.cu_seqlens,
                output=layer.self_attn,
            )
            LINEAR_F32(
                rt,
                x=layer.self_attn,
                weight=layer.out_proj_weight,
                bias=layer.out_proj_bias,
                output=layer.out_proj,
            )
            ADD_F32(rt, x=hidden_states, y=layer.out_proj, output=layer.self_attn_residual)
            LAYER_NORM_F32(
                rt,
                x=layer.self_attn_residual,
                weight=layer.final_layer_norm_weight,
                bias=layer.final_layer_norm_bias,
                output=layer.final_layer_norm,
            )
            LINEAR_GELU_F32(
                rt,
                x=layer.final_layer_norm,
                weight=layer.fc1_weight,
                bias=layer.fc1_bias,
                output=layer.fc1_gelu,
            )
            LINEAR_F32(
                rt, x=layer.fc1_gelu, weight=layer.fc2_weight, bias=layer.fc2_bias, output=layer.fc2
            )
            ADD_F32(rt, x=layer.self_attn_residual, y=layer.fc2, output=layer.output)
            hidden_states = layer.output
        LAYER_NORM_F32(
            rt,
            x=hidden_states,
            weight=tensors.ln_post_weight,
            bias=tensors.ln_post_bias,
            output=tensors.ln_post,
        )
        LINEAR_GELU_F32(
            rt,
            x=tensors.ln_post,
            weight=tensors.proj1_weight,
            bias=tensors.proj1_bias,
            output=tensors.proj1_gelu,
        )
        LINEAR_F32(
            rt,
            x=tensors.proj1_gelu,
            weight=tensors.proj2_weight,
            bias=tensors.proj2_bias,
            output=tensors.last_hidden_state,
        )
    return tensors.last_hidden_state
