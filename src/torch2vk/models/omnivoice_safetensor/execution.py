"""OmniVoice safetensor eager shader execution."""

from __future__ import annotations

from torch2vk.logical import LogicalTensor
from torch2vk.shader import ShaderVariant

from .shaders.omnivoice_audio_embedding_sum_f32 import OMNIVOICE_AUDIO_EMBEDDING_SUM_F32
from .shaders.omnivoice_audio_head_mat_vec_f32_f32 import OMNIVOICE_AUDIO_HEAD_MAT_VEC_F32_F32
from .shaders.omnivoice_codebook_argmax_f32 import OMNIVOICE_CODEBOOK_ARGMAX_F32
from .shaders.omnivoice_stage1_conv1d_k1_f32 import OMNIVOICE_STAGE1_CONV1D_K1_F32
from .shaders.omnivoice_stage1_conv1d_k7_f32 import OMNIVOICE_STAGE1_CONV1D_K7_F32
from .shaders.omnivoice_stage1_quantizer_embed_project_out_sum_f32 import (
    OMNIVOICE_STAGE1_QUANTIZER_EMBED_PROJECT_OUT_SUM_F32,
)
from .shaders.omnivoice_stage1_quantizer_embed_sum_f32 import (
    OMNIVOICE_STAGE1_QUANTIZER_EMBED_SUM_F32,
)
from .shaders.omnivoice_stage1_snake_conv1d_k1_residual_add_f32 import (
    OMNIVOICE_STAGE1_SNAKE_CONV1D_K1_RESIDUAL_ADD_F32,
)
from .shaders.omnivoice_stage1_snake_conv1d_k7_d1_f32 import (
    OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32,
)
from .shaders.omnivoice_stage1_snake_conv1d_k7_d3_f32 import (
    OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D3_F32,
)
from .shaders.omnivoice_stage1_snake_conv1d_k7_d9_f32 import (
    OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D9_F32,
)
from .shaders.omnivoice_stage1_snake_deconv1d_block0_f32 import (
    OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK0_F32,
)
from .shaders.omnivoice_stage1_snake_deconv1d_block1_f32 import (
    OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK1_F32,
)
from .shaders.omnivoice_stage1_snake_deconv1d_block2_f32 import (
    OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK2_F32,
)
from .shaders.omnivoice_stage1_snake_deconv1d_block3_f32 import (
    OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK3_F32,
)
from .shaders.omnivoice_stage1_snake_deconv1d_block4_f32 import (
    OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK4_F32,
)
from .tensors.stage0 import OmniVoiceStage0Tensors
from .tensors.stage1 import OmniVoiceStage1Tensors
from .tensors.weights import OmniVoiceWeights


def run_omnivoice_stage0_embedding_sum(
    ctx: object,
    *,
    tensors: OmniVoiceStage0Tensors,
    weights: OmniVoiceWeights,
) -> None:
    OMNIVOICE_AUDIO_EMBEDDING_SUM_F32(
        ctx,
        audio_ids=tensors.audio_ids,
        codebook_offsets=weights.stage0.codebook_layer_offsets,
        audio_embeddings=weights.stage0.audio_embeddings,
        output=tensors.audio_embedding_sum,
    )


def run_omnivoice_debug_audio_token_step(
    ctx: object,
    *,
    tensors: OmniVoiceStage0Tensors,
    stage1_tensors: OmniVoiceStage1Tensors | None = None,
    weights: OmniVoiceWeights,
) -> None:
    run_omnivoice_stage0_embedding_sum(
        ctx,
        tensors=tensors,
        weights=weights,
    )
    OMNIVOICE_AUDIO_HEAD_MAT_VEC_F32_F32(
        ctx,
        weight=weights.stage0.audio_heads,
        x=tensors.audio_head_hidden,
        output=tensors.audio_head_logits,
    )
    OMNIVOICE_CODEBOOK_ARGMAX_F32(
        ctx,
        logits=tensors.audio_head_logits,
        codebook_offsets=weights.stage0.codebook_layer_offsets,
        output_ids=tensors.argmax_ids,
    )
    if stage1_tensors is not None:
        embed0, embed1, embed2, embed3, embed4, embed5, embed6, embed7 = (
            weights.stage1.codebook_embeds
        )
        OMNIVOICE_STAGE1_QUANTIZER_EMBED_SUM_F32(
            ctx,
            audio_ids=tensors.argmax_ids,
            embed0=embed0,
            embed1=embed1,
            embed2=embed2,
            embed3=embed3,
            embed4=embed4,
            embed5=embed5,
            embed6=embed6,
            embed7=embed7,
            output=stage1_tensors.quantizer_embed_sum,
        )
        weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7 = (
            weights.stage1.project_out_weights
        )
        bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7 = (
            weights.stage1.project_out_biases
        )
        OMNIVOICE_STAGE1_QUANTIZER_EMBED_PROJECT_OUT_SUM_F32(
            ctx,
            audio_ids=tensors.argmax_ids,
            embed0=embed0,
            embed1=embed1,
            embed2=embed2,
            embed3=embed3,
            embed4=embed4,
            embed5=embed5,
            embed6=embed6,
            embed7=embed7,
            weight0=weight0,
            bias0=bias0,
            weight1=weight1,
            bias1=bias1,
            weight2=weight2,
            bias2=bias2,
            weight3=weight3,
            bias3=bias3,
            weight4=weight4,
            bias4=bias4,
            weight5=weight5,
            bias5=bias5,
            weight6=weight6,
            bias6=bias6,
            weight7=weight7,
            bias7=bias7,
            output=stage1_tensors.project_out_sum_hidden1024,
        )
        OMNIVOICE_STAGE1_CONV1D_K1_F32(
            ctx,
            x=stage1_tensors.project_out_sum_hidden1024,
            weight=weights.stage1.fc2_weight,
            bias=weights.stage1.fc2_bias,
            output=stage1_tensors.project_out_sum_hidden256,
        )
        OMNIVOICE_STAGE1_CONV1D_K7_F32(
            ctx,
            x=stage1_tensors.project_out_sum_hidden256,
            weight=weights.stage1.decoder_conv1_weight,
            bias=weights.stage1.decoder_conv1_bias,
            output=stage1_tensors.decoder_conv1,
        )
        OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK0_F32(
            ctx,
            x=stage1_tensors.decoder_conv1,
            alpha=weights.stage1.decoder_block0_snake_alpha,
            weight=weights.stage1.decoder_block0_deconv_weight,
            bias=weights.stage1.decoder_block0_deconv_bias,
            output=stage1_tensors.decoder_block0_deconv,
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block0_deconv,
            conv1_out=stage1_tensors.decoder_block0_res1_conv1,
            output=stage1_tensors.decoder_block0_res1_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32,
            snake1_alpha=weights.stage1.decoder_block0_resunit_snake1_alpha[0],
            conv1_weight=weights.stage1.decoder_block0_resunit_conv1_weight[0],
            conv1_bias=weights.stage1.decoder_block0_resunit_conv1_bias[0],
            snake2_alpha=weights.stage1.decoder_block0_resunit_snake2_alpha[0],
            conv2_weight=weights.stage1.decoder_block0_resunit_conv2_weight[0],
            conv2_bias=weights.stage1.decoder_block0_resunit_conv2_bias[0],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block0_res1_output,
            conv1_out=stage1_tensors.decoder_block0_res2_conv1,
            output=stage1_tensors.decoder_block0_res2_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D3_F32,
            snake1_alpha=weights.stage1.decoder_block0_resunit_snake1_alpha[1],
            conv1_weight=weights.stage1.decoder_block0_resunit_conv1_weight[1],
            conv1_bias=weights.stage1.decoder_block0_resunit_conv1_bias[1],
            snake2_alpha=weights.stage1.decoder_block0_resunit_snake2_alpha[1],
            conv2_weight=weights.stage1.decoder_block0_resunit_conv2_weight[1],
            conv2_bias=weights.stage1.decoder_block0_resunit_conv2_bias[1],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block0_res2_output,
            conv1_out=stage1_tensors.decoder_block0_res3_conv1,
            output=stage1_tensors.decoder_block0_res3_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D9_F32,
            snake1_alpha=weights.stage1.decoder_block0_resunit_snake1_alpha[2],
            conv1_weight=weights.stage1.decoder_block0_resunit_conv1_weight[2],
            conv1_bias=weights.stage1.decoder_block0_resunit_conv1_bias[2],
            snake2_alpha=weights.stage1.decoder_block0_resunit_snake2_alpha[2],
            conv2_weight=weights.stage1.decoder_block0_resunit_conv2_weight[2],
            conv2_bias=weights.stage1.decoder_block0_resunit_conv2_bias[2],
        )
        OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK1_F32(
            ctx,
            x=stage1_tensors.decoder_block0_res3_output,
            alpha=weights.stage1.decoder_block1_snake_alpha,
            weight=weights.stage1.decoder_block1_deconv_weight,
            bias=weights.stage1.decoder_block1_deconv_bias,
            output=stage1_tensors.decoder_block1_deconv,
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block1_deconv,
            conv1_out=stage1_tensors.decoder_block1_res1_conv1,
            output=stage1_tensors.decoder_block1_res1_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32,
            snake1_alpha=weights.stage1.decoder_block1_resunit_snake1_alpha[0],
            conv1_weight=weights.stage1.decoder_block1_resunit_conv1_weight[0],
            conv1_bias=weights.stage1.decoder_block1_resunit_conv1_bias[0],
            snake2_alpha=weights.stage1.decoder_block1_resunit_snake2_alpha[0],
            conv2_weight=weights.stage1.decoder_block1_resunit_conv2_weight[0],
            conv2_bias=weights.stage1.decoder_block1_resunit_conv2_bias[0],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block1_res1_output,
            conv1_out=stage1_tensors.decoder_block1_res2_conv1,
            output=stage1_tensors.decoder_block1_res2_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D3_F32,
            snake1_alpha=weights.stage1.decoder_block1_resunit_snake1_alpha[1],
            conv1_weight=weights.stage1.decoder_block1_resunit_conv1_weight[1],
            conv1_bias=weights.stage1.decoder_block1_resunit_conv1_bias[1],
            snake2_alpha=weights.stage1.decoder_block1_resunit_snake2_alpha[1],
            conv2_weight=weights.stage1.decoder_block1_resunit_conv2_weight[1],
            conv2_bias=weights.stage1.decoder_block1_resunit_conv2_bias[1],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block1_res2_output,
            conv1_out=stage1_tensors.decoder_block1_res3_conv1,
            output=stage1_tensors.decoder_block1_res3_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D9_F32,
            snake1_alpha=weights.stage1.decoder_block1_resunit_snake1_alpha[2],
            conv1_weight=weights.stage1.decoder_block1_resunit_conv1_weight[2],
            conv1_bias=weights.stage1.decoder_block1_resunit_conv1_bias[2],
            snake2_alpha=weights.stage1.decoder_block1_resunit_snake2_alpha[2],
            conv2_weight=weights.stage1.decoder_block1_resunit_conv2_weight[2],
            conv2_bias=weights.stage1.decoder_block1_resunit_conv2_bias[2],
        )
        OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK2_F32(
            ctx,
            x=stage1_tensors.decoder_block1_res3_output,
            alpha=weights.stage1.decoder_block2_snake_alpha,
            weight=weights.stage1.decoder_block2_deconv_weight,
            bias=weights.stage1.decoder_block2_deconv_bias,
            output=stage1_tensors.decoder_block2_deconv,
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block2_deconv,
            conv1_out=stage1_tensors.decoder_block2_res1_conv1,
            output=stage1_tensors.decoder_block2_res1_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32,
            snake1_alpha=weights.stage1.decoder_block2_resunit_snake1_alpha[0],
            conv1_weight=weights.stage1.decoder_block2_resunit_conv1_weight[0],
            conv1_bias=weights.stage1.decoder_block2_resunit_conv1_bias[0],
            snake2_alpha=weights.stage1.decoder_block2_resunit_snake2_alpha[0],
            conv2_weight=weights.stage1.decoder_block2_resunit_conv2_weight[0],
            conv2_bias=weights.stage1.decoder_block2_resunit_conv2_bias[0],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block2_res1_output,
            conv1_out=stage1_tensors.decoder_block2_res2_conv1,
            output=stage1_tensors.decoder_block2_res2_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D3_F32,
            snake1_alpha=weights.stage1.decoder_block2_resunit_snake1_alpha[1],
            conv1_weight=weights.stage1.decoder_block2_resunit_conv1_weight[1],
            conv1_bias=weights.stage1.decoder_block2_resunit_conv1_bias[1],
            snake2_alpha=weights.stage1.decoder_block2_resunit_snake2_alpha[1],
            conv2_weight=weights.stage1.decoder_block2_resunit_conv2_weight[1],
            conv2_bias=weights.stage1.decoder_block2_resunit_conv2_bias[1],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block2_res2_output,
            conv1_out=stage1_tensors.decoder_block2_res3_conv1,
            output=stage1_tensors.decoder_block2_res3_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D9_F32,
            snake1_alpha=weights.stage1.decoder_block2_resunit_snake1_alpha[2],
            conv1_weight=weights.stage1.decoder_block2_resunit_conv1_weight[2],
            conv1_bias=weights.stage1.decoder_block2_resunit_conv1_bias[2],
            snake2_alpha=weights.stage1.decoder_block2_resunit_snake2_alpha[2],
            conv2_weight=weights.stage1.decoder_block2_resunit_conv2_weight[2],
            conv2_bias=weights.stage1.decoder_block2_resunit_conv2_bias[2],
        )
        OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK3_F32(
            ctx,
            x=stage1_tensors.decoder_block2_res3_output,
            alpha=weights.stage1.decoder_block3_snake_alpha,
            weight=weights.stage1.decoder_block3_deconv_weight,
            bias=weights.stage1.decoder_block3_deconv_bias,
            output=stage1_tensors.decoder_block3_deconv,
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block3_deconv,
            conv1_out=stage1_tensors.decoder_block3_res1_conv1,
            output=stage1_tensors.decoder_block3_res1_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32,
            snake1_alpha=weights.stage1.decoder_block3_resunit_snake1_alpha[0],
            conv1_weight=weights.stage1.decoder_block3_resunit_conv1_weight[0],
            conv1_bias=weights.stage1.decoder_block3_resunit_conv1_bias[0],
            snake2_alpha=weights.stage1.decoder_block3_resunit_snake2_alpha[0],
            conv2_weight=weights.stage1.decoder_block3_resunit_conv2_weight[0],
            conv2_bias=weights.stage1.decoder_block3_resunit_conv2_bias[0],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block3_res1_output,
            conv1_out=stage1_tensors.decoder_block3_res2_conv1,
            output=stage1_tensors.decoder_block3_res2_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D3_F32,
            snake1_alpha=weights.stage1.decoder_block3_resunit_snake1_alpha[1],
            conv1_weight=weights.stage1.decoder_block3_resunit_conv1_weight[1],
            conv1_bias=weights.stage1.decoder_block3_resunit_conv1_bias[1],
            snake2_alpha=weights.stage1.decoder_block3_resunit_snake2_alpha[1],
            conv2_weight=weights.stage1.decoder_block3_resunit_conv2_weight[1],
            conv2_bias=weights.stage1.decoder_block3_resunit_conv2_bias[1],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block3_res2_output,
            conv1_out=stage1_tensors.decoder_block3_res3_conv1,
            output=stage1_tensors.decoder_block3_res3_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D9_F32,
            snake1_alpha=weights.stage1.decoder_block3_resunit_snake1_alpha[2],
            conv1_weight=weights.stage1.decoder_block3_resunit_conv1_weight[2],
            conv1_bias=weights.stage1.decoder_block3_resunit_conv1_bias[2],
            snake2_alpha=weights.stage1.decoder_block3_resunit_snake2_alpha[2],
            conv2_weight=weights.stage1.decoder_block3_resunit_conv2_weight[2],
            conv2_bias=weights.stage1.decoder_block3_resunit_conv2_bias[2],
        )
        OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK4_F32(
            ctx,
            x=stage1_tensors.decoder_block3_res3_output,
            alpha=weights.stage1.decoder_block4_snake_alpha,
            weight=weights.stage1.decoder_block4_deconv_weight,
            bias=weights.stage1.decoder_block4_deconv_bias,
            output=stage1_tensors.decoder_block4_deconv,
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block4_deconv,
            conv1_out=stage1_tensors.decoder_block4_res1_conv1,
            output=stage1_tensors.decoder_block4_res1_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32,
            snake1_alpha=weights.stage1.decoder_block4_resunit_snake1_alpha[0],
            conv1_weight=weights.stage1.decoder_block4_resunit_conv1_weight[0],
            conv1_bias=weights.stage1.decoder_block4_resunit_conv1_bias[0],
            snake2_alpha=weights.stage1.decoder_block4_resunit_snake2_alpha[0],
            conv2_weight=weights.stage1.decoder_block4_resunit_conv2_weight[0],
            conv2_bias=weights.stage1.decoder_block4_resunit_conv2_bias[0],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block4_res1_output,
            conv1_out=stage1_tensors.decoder_block4_res2_conv1,
            output=stage1_tensors.decoder_block4_res2_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D3_F32,
            snake1_alpha=weights.stage1.decoder_block4_resunit_snake1_alpha[1],
            conv1_weight=weights.stage1.decoder_block4_resunit_conv1_weight[1],
            conv1_bias=weights.stage1.decoder_block4_resunit_conv1_bias[1],
            snake2_alpha=weights.stage1.decoder_block4_resunit_snake2_alpha[1],
            conv2_weight=weights.stage1.decoder_block4_resunit_conv2_weight[1],
            conv2_bias=weights.stage1.decoder_block4_resunit_conv2_bias[1],
        )
        _run_block0_resunit(
            ctx,
            x=stage1_tensors.decoder_block4_res2_output,
            conv1_out=stage1_tensors.decoder_block4_res3_conv1,
            output=stage1_tensors.decoder_block4_res3_output,
            conv1_shader=OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D9_F32,
            snake1_alpha=weights.stage1.decoder_block4_resunit_snake1_alpha[2],
            conv1_weight=weights.stage1.decoder_block4_resunit_conv1_weight[2],
            conv1_bias=weights.stage1.decoder_block4_resunit_conv1_bias[2],
            snake2_alpha=weights.stage1.decoder_block4_resunit_snake2_alpha[2],
            conv2_weight=weights.stage1.decoder_block4_resunit_conv2_weight[2],
            conv2_bias=weights.stage1.decoder_block4_resunit_conv2_bias[2],
        )
        OMNIVOICE_STAGE1_SNAKE_CONV1D_K7_D1_F32(
            ctx,
            x=stage1_tensors.decoder_block4_res3_output,
            alpha=weights.stage1.decoder_final_snake_alpha,
            weight=weights.stage1.decoder_final_conv_weight,
            bias=weights.stage1.decoder_final_conv_bias,
            output=stage1_tensors.waveform,
        )


def _run_block0_resunit(
    ctx: object,
    *,
    x: LogicalTensor,
    conv1_out: LogicalTensor,
    output: LogicalTensor,
    conv1_shader: ShaderVariant,
    snake1_alpha: LogicalTensor,
    conv1_weight: LogicalTensor,
    conv1_bias: LogicalTensor,
    snake2_alpha: LogicalTensor,
    conv2_weight: LogicalTensor,
    conv2_bias: LogicalTensor,
) -> None:
    conv1_shader(
        ctx,
        x=x,
        alpha=snake1_alpha,
        weight=conv1_weight,
        bias=conv1_bias,
        output=conv1_out,
    )
    OMNIVOICE_STAGE1_SNAKE_CONV1D_K1_RESIDUAL_ADD_F32(
        ctx,
        x=conv1_out,
        alpha=snake2_alpha,
        weight=conv2_weight,
        bias=conv2_bias,
        residual=x,
        output=output,
    )
