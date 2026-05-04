"""Qwen3-ASR audio tower frame orchestration."""

from __future__ import annotations

from models.qwen3_asr.shaders.add_f32 import QWEN3_ASR_ADD_F32
from models.qwen3_asr.shaders.attention_f32 import QWEN3_ASR_ENCODER_ATTENTION_F32
from models.qwen3_asr.shaders.compact_after_cnn_f32 import QWEN3_ASR_COMPACT_AFTER_CNN_F32
from models.qwen3_asr.shaders.conv2d_gelu_f32 import QWEN3_ASR_CONV2D_GELU_F32
from models.qwen3_asr.shaders.conv_out_add_position_f32 import QWEN3_ASR_CONV_OUT_ADD_POSITION_F32
from models.qwen3_asr.shaders.cu_seqlens_u32 import QWEN3_ASR_CU_SEQLENS_U32
from models.qwen3_asr.shaders.layer_norm_f32 import QWEN3_ASR_LAYER_NORM_F32
from models.qwen3_asr.shaders.linear_f32 import QWEN3_ASR_LINEAR_F32, QWEN3_ASR_LINEAR_GELU_F32
from models.qwen3_asr.shaders.pad_feature_f32 import QWEN3_ASR_PAD_FEATURE_F32
from models.qwen3_asr.tensors.audio_tower import Qwen3AsrAudioTowerTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_audio_tower(
    rt: RuntimeSession,
    tensors: Qwen3AsrAudioTowerTensors,
) -> LogicalTensor:
    """Run the audio encoder as one PyTorch-comparable frame."""
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    with rt.frame(
        "qwen3_asr.audio_tower",
        pytorch_model_class=Qwen3ASRForConditionalGeneration,
        pytorch_model_submodule="thinker.audio_tower",
    ):
        QWEN3_ASR_PAD_FEATURE_F32(
            rt,
            input_features=tensors.input_features,
            feature_lens=tensors.feature_lens,
            output=tensors.padded_feature,
        )
        QWEN3_ASR_CONV2D_GELU_F32(
            rt,
            x=tensors.padded_feature,
            weight=tensors.conv2d1_weight,
            bias=tensors.conv2d1_bias,
            output=tensors.conv2d1_gelu,
        )
        QWEN3_ASR_CONV2D_GELU_F32(
            rt,
            x=tensors.conv2d1_gelu,
            weight=tensors.conv2d2_weight,
            bias=tensors.conv2d2_bias,
            output=tensors.conv2d2_gelu,
        )
        QWEN3_ASR_CONV2D_GELU_F32(
            rt,
            x=tensors.conv2d2_gelu,
            weight=tensors.conv2d3_weight,
            bias=tensors.conv2d3_bias,
            output=tensors.conv2d3_gelu,
        )
        QWEN3_ASR_CONV_OUT_ADD_POSITION_F32(
            rt,
            x=tensors.conv2d3_gelu,
            weight=tensors.conv_out_weight,
            output=tensors.conv_out_add_position,
        )
        QWEN3_ASR_COMPACT_AFTER_CNN_F32(
            rt,
            x=tensors.conv_out_add_position,
            feature_lens=tensors.feature_lens,
            output=tensors.hidden_states,
        )
        QWEN3_ASR_CU_SEQLENS_U32(
            rt,
            feature_lens=tensors.feature_lens,
            output=tensors.cu_seqlens,
        )
        hidden_states = tensors.hidden_states
        for layer_tensors in tensors.layers:
            QWEN3_ASR_LAYER_NORM_F32(
                rt,
                x=hidden_states,
                weight=layer_tensors.self_attn_layer_norm_weight,
                bias=layer_tensors.self_attn_layer_norm_bias,
                output=layer_tensors.self_attn_layer_norm,
            )
            QWEN3_ASR_LINEAR_F32(
                rt,
                x=layer_tensors.self_attn_layer_norm,
                weight=layer_tensors.q_proj_weight,
                bias=layer_tensors.q_proj_bias,
                output=layer_tensors.q_proj,
            )
            QWEN3_ASR_LINEAR_F32(
                rt,
                x=layer_tensors.self_attn_layer_norm,
                weight=layer_tensors.k_proj_weight,
                bias=layer_tensors.k_proj_bias,
                output=layer_tensors.k_proj,
            )
            QWEN3_ASR_LINEAR_F32(
                rt,
                x=layer_tensors.self_attn_layer_norm,
                weight=layer_tensors.v_proj_weight,
                bias=layer_tensors.v_proj_bias,
                output=layer_tensors.v_proj,
            )
            QWEN3_ASR_ENCODER_ATTENTION_F32(
                rt,
                q=layer_tensors.q_proj,
                k=layer_tensors.k_proj,
                v=layer_tensors.v_proj,
                cu_seqlens=tensors.cu_seqlens,
                output=layer_tensors.self_attn,
            )
            QWEN3_ASR_LINEAR_F32(
                rt,
                x=layer_tensors.self_attn,
                weight=layer_tensors.out_proj_weight,
                bias=layer_tensors.out_proj_bias,
                output=layer_tensors.out_proj,
            )
            QWEN3_ASR_ADD_F32(
                rt,
                x=hidden_states,
                y=layer_tensors.out_proj,
                output=layer_tensors.self_attn_residual,
            )
            QWEN3_ASR_LAYER_NORM_F32(
                rt,
                x=layer_tensors.self_attn_residual,
                weight=layer_tensors.final_layer_norm_weight,
                bias=layer_tensors.final_layer_norm_bias,
                output=layer_tensors.final_layer_norm,
            )
            QWEN3_ASR_LINEAR_GELU_F32(
                rt,
                x=layer_tensors.final_layer_norm,
                weight=layer_tensors.fc1_weight,
                bias=layer_tensors.fc1_bias,
                output=layer_tensors.fc1_gelu,
            )
            QWEN3_ASR_LINEAR_F32(
                rt,
                x=layer_tensors.fc1_gelu,
                weight=layer_tensors.fc2_weight,
                bias=layer_tensors.fc2_bias,
                output=layer_tensors.fc2,
            )
            QWEN3_ASR_ADD_F32(
                rt,
                x=layer_tensors.self_attn_residual,
                y=layer_tensors.fc2,
                output=layer_tensors.output,
            )
            hidden_states = layer_tensors.output

        QWEN3_ASR_LAYER_NORM_F32(
            rt,
            x=hidden_states,
            weight=tensors.ln_post_weight,
            bias=tensors.ln_post_bias,
            output=tensors.ln_post,
        )
        QWEN3_ASR_LINEAR_GELU_F32(
            rt,
            x=tensors.ln_post,
            weight=tensors.proj1_weight,
            bias=tensors.proj1_bias,
            output=tensors.proj1_gelu,
        )
        QWEN3_ASR_LINEAR_F32(
            rt,
            x=tensors.proj1_gelu,
            weight=tensors.proj2_weight,
            bias=tensors.proj2_bias,
            output=tensors.last_hidden_state,
        )
    return tensors.last_hidden_state
