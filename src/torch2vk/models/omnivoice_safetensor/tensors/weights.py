"""OmniVoice LogicalTensor weight tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import LogicalTensor
from torch2vk.models.omnivoice_safetensor.spec import OmniVoiceSpec
from torch2vk.schema import W


@dataclass(frozen=True, slots=True)
class OmniVoiceStage0Weights:
    audio_embeddings: LogicalTensor
    audio_heads: LogicalTensor
    codebook_layer_offsets: LogicalTensor


@dataclass(frozen=True, slots=True)
class OmniVoiceStage1Weights:
    codebook_embeds: tuple[LogicalTensor, ...]
    project_out_weights: tuple[LogicalTensor, ...]
    project_out_biases: tuple[LogicalTensor, ...]
    fc2_weight: LogicalTensor
    fc2_bias: LogicalTensor
    decoder_conv1_weight: LogicalTensor
    decoder_conv1_bias: LogicalTensor
    decoder_block0_snake_alpha: LogicalTensor
    decoder_block0_deconv_weight: LogicalTensor
    decoder_block0_deconv_bias: LogicalTensor
    decoder_block0_resunit_snake1_alpha: tuple[LogicalTensor, ...]
    decoder_block0_resunit_conv1_weight: tuple[LogicalTensor, ...]
    decoder_block0_resunit_conv1_bias: tuple[LogicalTensor, ...]
    decoder_block0_resunit_snake2_alpha: tuple[LogicalTensor, ...]
    decoder_block0_resunit_conv2_weight: tuple[LogicalTensor, ...]
    decoder_block0_resunit_conv2_bias: tuple[LogicalTensor, ...]
    decoder_block1_snake_alpha: LogicalTensor
    decoder_block1_deconv_weight: LogicalTensor
    decoder_block1_deconv_bias: LogicalTensor
    decoder_block1_resunit_snake1_alpha: tuple[LogicalTensor, ...]
    decoder_block1_resunit_conv1_weight: tuple[LogicalTensor, ...]
    decoder_block1_resunit_conv1_bias: tuple[LogicalTensor, ...]
    decoder_block1_resunit_snake2_alpha: tuple[LogicalTensor, ...]
    decoder_block1_resunit_conv2_weight: tuple[LogicalTensor, ...]
    decoder_block1_resunit_conv2_bias: tuple[LogicalTensor, ...]
    decoder_block2_snake_alpha: LogicalTensor
    decoder_block2_deconv_weight: LogicalTensor
    decoder_block2_deconv_bias: LogicalTensor
    decoder_block2_resunit_snake1_alpha: tuple[LogicalTensor, ...]
    decoder_block2_resunit_conv1_weight: tuple[LogicalTensor, ...]
    decoder_block2_resunit_conv1_bias: tuple[LogicalTensor, ...]
    decoder_block2_resunit_snake2_alpha: tuple[LogicalTensor, ...]
    decoder_block2_resunit_conv2_weight: tuple[LogicalTensor, ...]
    decoder_block2_resunit_conv2_bias: tuple[LogicalTensor, ...]
    decoder_block3_snake_alpha: LogicalTensor
    decoder_block3_deconv_weight: LogicalTensor
    decoder_block3_deconv_bias: LogicalTensor
    decoder_block3_resunit_snake1_alpha: tuple[LogicalTensor, ...]
    decoder_block3_resunit_conv1_weight: tuple[LogicalTensor, ...]
    decoder_block3_resunit_conv1_bias: tuple[LogicalTensor, ...]
    decoder_block3_resunit_snake2_alpha: tuple[LogicalTensor, ...]
    decoder_block3_resunit_conv2_weight: tuple[LogicalTensor, ...]
    decoder_block3_resunit_conv2_bias: tuple[LogicalTensor, ...]
    decoder_block4_snake_alpha: LogicalTensor
    decoder_block4_deconv_weight: LogicalTensor
    decoder_block4_deconv_bias: LogicalTensor
    decoder_block4_resunit_snake1_alpha: tuple[LogicalTensor, ...]
    decoder_block4_resunit_conv1_weight: tuple[LogicalTensor, ...]
    decoder_block4_resunit_conv1_bias: tuple[LogicalTensor, ...]
    decoder_block4_resunit_snake2_alpha: tuple[LogicalTensor, ...]
    decoder_block4_resunit_conv2_weight: tuple[LogicalTensor, ...]
    decoder_block4_resunit_conv2_bias: tuple[LogicalTensor, ...]
    decoder_final_snake_alpha: LogicalTensor
    decoder_final_conv_weight: LogicalTensor
    decoder_final_conv_bias: LogicalTensor


@dataclass(frozen=True, slots=True)
class OmniVoiceWeights:
    stage0: OmniVoiceStage0Weights
    stage1: OmniVoiceStage1Weights


def omnivoice_weights(spec: OmniVoiceSpec) -> OmniVoiceWeights:
    return OmniVoiceWeights(
        stage0=OmniVoiceStage0Weights(
            audio_embeddings=W(
                "weights.stage0.audio_embeddings",
                safetensor_key="audio_embeddings.weight",
                dtype="float32",
                shape=(spec.audio_vocab_size * spec.num_audio_codebook, spec.qwen3.hidden_size),
            ),
            audio_heads=W(
                "weights.stage0.audio_heads",
                safetensor_key="audio_heads.weight",
                dtype="float32",
                shape=(spec.audio_vocab_size * spec.num_audio_codebook, spec.qwen3.hidden_size),
            ),
            codebook_layer_offsets=W(
                "weights.stage0.codebook_layer_offsets",
                safetensor_key="codebook_layer_offsets",
                dtype="int64",
                shape=(spec.num_audio_codebook,),
            ),
        ),
        stage1=OmniVoiceStage1Weights(
            codebook_embeds=tuple(
                W(
                    f"weights.stage1.quantizer.{index}.codebook_embed",
                    safetensor_key=f"audio_tokenizer:quantizer.quantizers.{index}.codebook.embed",
                    dtype="float32",
                    shape=(1024, 64),
                )
                for index in range(spec.num_audio_codebook)
            ),
            project_out_weights=tuple(
                W(
                    f"weights.stage1.quantizer.{index}.project_out_weight",
                    safetensor_key=f"audio_tokenizer:quantizer.quantizers.{index}.project_out.weight",
                    dtype="float32",
                    shape=(1024, 64),
                )
                for index in range(spec.num_audio_codebook)
            ),
            project_out_biases=tuple(
                W(
                    f"weights.stage1.quantizer.{index}.project_out_bias",
                    safetensor_key=f"audio_tokenizer:quantizer.quantizers.{index}.project_out.bias",
                    dtype="float32",
                    shape=(1024,),
                )
                for index in range(spec.num_audio_codebook)
            ),
            fc2_weight=W(
                "weights.stage1.fc2.weight",
                safetensor_key="audio_tokenizer:fc2.weight",
                dtype="float32",
                shape=(256, 1024),
            ),
            fc2_bias=W(
                "weights.stage1.fc2.bias",
                safetensor_key="audio_tokenizer:fc2.bias",
                dtype="float32",
                shape=(256,),
            ),
            decoder_conv1_weight=W(
                "weights.stage1.acoustic_decoder.conv1.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.conv1.weight",
                dtype="float32",
                shape=(1024, 256, 7),
            ),
            decoder_conv1_bias=W(
                "weights.stage1.acoustic_decoder.conv1.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.conv1.bias",
                dtype="float32",
                shape=(1024,),
            ),
            decoder_block0_snake_alpha=W(
                "weights.stage1.acoustic_decoder.block0.snake1.alpha",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.0.snake1.alpha",
                dtype="float32",
                shape=(1, 1024, 1),
            ),
            decoder_block0_deconv_weight=W(
                "weights.stage1.acoustic_decoder.block0.conv_t1.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.0.conv_t1.weight",
                dtype="float32",
                shape=(1024, 512, 16),
            ),
            decoder_block0_deconv_bias=W(
                "weights.stage1.acoustic_decoder.block0.conv_t1.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.0.conv_t1.bias",
                dtype="float32",
                shape=(512,),
            ),
            decoder_block0_resunit_snake1_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block0.res_unit{index}.snake1.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.0.res_unit{index}.snake1.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 512, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block0_resunit_conv1_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block0.res_unit{index}.conv1.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.0.res_unit{index}.conv1.weight"
                    ),
                    dtype="float32",
                    shape=(512, 512, 7),
                )
                for index in range(1, 4)
            ),
            decoder_block0_resunit_conv1_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block0.res_unit{index}.conv1.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.0.res_unit{index}.conv1.bias"
                    ),
                    dtype="float32",
                    shape=(512,),
                )
                for index in range(1, 4)
            ),
            decoder_block0_resunit_snake2_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block0.res_unit{index}.snake2.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.0.res_unit{index}.snake2.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 512, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block0_resunit_conv2_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block0.res_unit{index}.conv2.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.0.res_unit{index}.conv2.weight"
                    ),
                    dtype="float32",
                    shape=(512, 512, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block0_resunit_conv2_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block0.res_unit{index}.conv2.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.0.res_unit{index}.conv2.bias"
                    ),
                    dtype="float32",
                    shape=(512,),
                )
                for index in range(1, 4)
            ),
            decoder_block1_snake_alpha=W(
                "weights.stage1.acoustic_decoder.block1.snake1.alpha",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.1.snake1.alpha",
                dtype="float32",
                shape=(1, 512, 1),
            ),
            decoder_block1_deconv_weight=W(
                "weights.stage1.acoustic_decoder.block1.conv_t1.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.1.conv_t1.weight",
                dtype="float32",
                shape=(512, 256, 10),
            ),
            decoder_block1_deconv_bias=W(
                "weights.stage1.acoustic_decoder.block1.conv_t1.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.1.conv_t1.bias",
                dtype="float32",
                shape=(256,),
            ),
            decoder_block1_resunit_snake1_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block1.res_unit{index}.snake1.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.1.res_unit{index}.snake1.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 256, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block1_resunit_conv1_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block1.res_unit{index}.conv1.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.1.res_unit{index}.conv1.weight"
                    ),
                    dtype="float32",
                    shape=(256, 256, 7),
                )
                for index in range(1, 4)
            ),
            decoder_block1_resunit_conv1_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block1.res_unit{index}.conv1.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.1.res_unit{index}.conv1.bias"
                    ),
                    dtype="float32",
                    shape=(256,),
                )
                for index in range(1, 4)
            ),
            decoder_block1_resunit_snake2_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block1.res_unit{index}.snake2.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.1.res_unit{index}.snake2.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 256, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block1_resunit_conv2_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block1.res_unit{index}.conv2.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.1.res_unit{index}.conv2.weight"
                    ),
                    dtype="float32",
                    shape=(256, 256, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block1_resunit_conv2_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block1.res_unit{index}.conv2.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.1.res_unit{index}.conv2.bias"
                    ),
                    dtype="float32",
                    shape=(256,),
                )
                for index in range(1, 4)
            ),
            decoder_block2_snake_alpha=W(
                "weights.stage1.acoustic_decoder.block2.snake1.alpha",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.2.snake1.alpha",
                dtype="float32",
                shape=(1, 256, 1),
            ),
            decoder_block2_deconv_weight=W(
                "weights.stage1.acoustic_decoder.block2.conv_t1.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.2.conv_t1.weight",
                dtype="float32",
                shape=(256, 128, 8),
            ),
            decoder_block2_deconv_bias=W(
                "weights.stage1.acoustic_decoder.block2.conv_t1.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.2.conv_t1.bias",
                dtype="float32",
                shape=(128,),
            ),
            decoder_block2_resunit_snake1_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block2.res_unit{index}.snake1.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.2.res_unit{index}.snake1.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 128, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block2_resunit_conv1_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block2.res_unit{index}.conv1.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.2.res_unit{index}.conv1.weight"
                    ),
                    dtype="float32",
                    shape=(128, 128, 7),
                )
                for index in range(1, 4)
            ),
            decoder_block2_resunit_conv1_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block2.res_unit{index}.conv1.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.2.res_unit{index}.conv1.bias"
                    ),
                    dtype="float32",
                    shape=(128,),
                )
                for index in range(1, 4)
            ),
            decoder_block2_resunit_snake2_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block2.res_unit{index}.snake2.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.2.res_unit{index}.snake2.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 128, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block2_resunit_conv2_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block2.res_unit{index}.conv2.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.2.res_unit{index}.conv2.weight"
                    ),
                    dtype="float32",
                    shape=(128, 128, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block2_resunit_conv2_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block2.res_unit{index}.conv2.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.2.res_unit{index}.conv2.bias"
                    ),
                    dtype="float32",
                    shape=(128,),
                )
                for index in range(1, 4)
            ),
            decoder_block3_snake_alpha=W(
                "weights.stage1.acoustic_decoder.block3.snake1.alpha",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.3.snake1.alpha",
                dtype="float32",
                shape=(1, 128, 1),
            ),
            decoder_block3_deconv_weight=W(
                "weights.stage1.acoustic_decoder.block3.conv_t1.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.3.conv_t1.weight",
                dtype="float32",
                shape=(128, 64, 4),
            ),
            decoder_block3_deconv_bias=W(
                "weights.stage1.acoustic_decoder.block3.conv_t1.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.3.conv_t1.bias",
                dtype="float32",
                shape=(64,),
            ),
            decoder_block3_resunit_snake1_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block3.res_unit{index}.snake1.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.3.res_unit{index}.snake1.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 64, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block3_resunit_conv1_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block3.res_unit{index}.conv1.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.3.res_unit{index}.conv1.weight"
                    ),
                    dtype="float32",
                    shape=(64, 64, 7),
                )
                for index in range(1, 4)
            ),
            decoder_block3_resunit_conv1_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block3.res_unit{index}.conv1.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.3.res_unit{index}.conv1.bias"
                    ),
                    dtype="float32",
                    shape=(64,),
                )
                for index in range(1, 4)
            ),
            decoder_block3_resunit_snake2_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block3.res_unit{index}.snake2.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.3.res_unit{index}.snake2.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 64, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block3_resunit_conv2_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block3.res_unit{index}.conv2.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.3.res_unit{index}.conv2.weight"
                    ),
                    dtype="float32",
                    shape=(64, 64, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block3_resunit_conv2_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block3.res_unit{index}.conv2.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.3.res_unit{index}.conv2.bias"
                    ),
                    dtype="float32",
                    shape=(64,),
                )
                for index in range(1, 4)
            ),
            decoder_block4_snake_alpha=W(
                "weights.stage1.acoustic_decoder.block4.snake1.alpha",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.4.snake1.alpha",
                dtype="float32",
                shape=(1, 64, 1),
            ),
            decoder_block4_deconv_weight=W(
                "weights.stage1.acoustic_decoder.block4.conv_t1.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.4.conv_t1.weight",
                dtype="float32",
                shape=(64, 32, 6),
            ),
            decoder_block4_deconv_bias=W(
                "weights.stage1.acoustic_decoder.block4.conv_t1.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.block.4.conv_t1.bias",
                dtype="float32",
                shape=(32,),
            ),
            decoder_block4_resunit_snake1_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block4.res_unit{index}.snake1.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.4.res_unit{index}.snake1.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 32, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block4_resunit_conv1_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block4.res_unit{index}.conv1.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.4.res_unit{index}.conv1.weight"
                    ),
                    dtype="float32",
                    shape=(32, 32, 7),
                )
                for index in range(1, 4)
            ),
            decoder_block4_resunit_conv1_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block4.res_unit{index}.conv1.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.4.res_unit{index}.conv1.bias"
                    ),
                    dtype="float32",
                    shape=(32,),
                )
                for index in range(1, 4)
            ),
            decoder_block4_resunit_snake2_alpha=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block4.res_unit{index}.snake2.alpha",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.4.res_unit{index}.snake2.alpha"
                    ),
                    dtype="float32",
                    shape=(1, 32, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block4_resunit_conv2_weight=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block4.res_unit{index}.conv2.weight",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.4.res_unit{index}.conv2.weight"
                    ),
                    dtype="float32",
                    shape=(32, 32, 1),
                )
                for index in range(1, 4)
            ),
            decoder_block4_resunit_conv2_bias=tuple(
                W(
                    f"weights.stage1.acoustic_decoder.block4.res_unit{index}.conv2.bias",
                    safetensor_key=(
                        "audio_tokenizer:"
                        f"acoustic_decoder.block.4.res_unit{index}.conv2.bias"
                    ),
                    dtype="float32",
                    shape=(32,),
                )
                for index in range(1, 4)
            ),
            decoder_final_snake_alpha=W(
                "weights.stage1.acoustic_decoder.snake1.alpha",
                safetensor_key="audio_tokenizer:acoustic_decoder.snake1.alpha",
                dtype="float32",
                shape=(1, 32, 1),
            ),
            decoder_final_conv_weight=W(
                "weights.stage1.acoustic_decoder.conv2.weight",
                safetensor_key="audio_tokenizer:acoustic_decoder.conv2.weight",
                dtype="float32",
                shape=(1, 32, 7),
            ),
            decoder_final_conv_bias=W(
                "weights.stage1.acoustic_decoder.conv2.bias",
                safetensor_key="audio_tokenizer:acoustic_decoder.conv2.bias",
                dtype="float32",
                shape=(1,),
            ),
        ),
    )


def omnivoice_weight_tensors(weights: OmniVoiceWeights) -> tuple[LogicalTensor, ...]:
    return (
        weights.stage0.audio_embeddings,
        weights.stage0.audio_heads,
        weights.stage0.codebook_layer_offsets,
        *weights.stage1.codebook_embeds,
        *weights.stage1.project_out_weights,
        *weights.stage1.project_out_biases,
        weights.stage1.fc2_weight,
        weights.stage1.fc2_bias,
        weights.stage1.decoder_conv1_weight,
        weights.stage1.decoder_conv1_bias,
        weights.stage1.decoder_block0_snake_alpha,
        weights.stage1.decoder_block0_deconv_weight,
        weights.stage1.decoder_block0_deconv_bias,
        *weights.stage1.decoder_block0_resunit_snake1_alpha,
        *weights.stage1.decoder_block0_resunit_conv1_weight,
        *weights.stage1.decoder_block0_resunit_conv1_bias,
        *weights.stage1.decoder_block0_resunit_snake2_alpha,
        *weights.stage1.decoder_block0_resunit_conv2_weight,
        *weights.stage1.decoder_block0_resunit_conv2_bias,
        weights.stage1.decoder_block1_snake_alpha,
        weights.stage1.decoder_block1_deconv_weight,
        weights.stage1.decoder_block1_deconv_bias,
        *weights.stage1.decoder_block1_resunit_snake1_alpha,
        *weights.stage1.decoder_block1_resunit_conv1_weight,
        *weights.stage1.decoder_block1_resunit_conv1_bias,
        *weights.stage1.decoder_block1_resunit_snake2_alpha,
        *weights.stage1.decoder_block1_resunit_conv2_weight,
        *weights.stage1.decoder_block1_resunit_conv2_bias,
        weights.stage1.decoder_block2_snake_alpha,
        weights.stage1.decoder_block2_deconv_weight,
        weights.stage1.decoder_block2_deconv_bias,
        *weights.stage1.decoder_block2_resunit_snake1_alpha,
        *weights.stage1.decoder_block2_resunit_conv1_weight,
        *weights.stage1.decoder_block2_resunit_conv1_bias,
        *weights.stage1.decoder_block2_resunit_snake2_alpha,
        *weights.stage1.decoder_block2_resunit_conv2_weight,
        *weights.stage1.decoder_block2_resunit_conv2_bias,
        weights.stage1.decoder_block3_snake_alpha,
        weights.stage1.decoder_block3_deconv_weight,
        weights.stage1.decoder_block3_deconv_bias,
        *weights.stage1.decoder_block3_resunit_snake1_alpha,
        *weights.stage1.decoder_block3_resunit_conv1_weight,
        *weights.stage1.decoder_block3_resunit_conv1_bias,
        *weights.stage1.decoder_block3_resunit_snake2_alpha,
        *weights.stage1.decoder_block3_resunit_conv2_weight,
        *weights.stage1.decoder_block3_resunit_conv2_bias,
        weights.stage1.decoder_block4_snake_alpha,
        weights.stage1.decoder_block4_deconv_weight,
        weights.stage1.decoder_block4_deconv_bias,
        *weights.stage1.decoder_block4_resunit_snake1_alpha,
        *weights.stage1.decoder_block4_resunit_conv1_weight,
        *weights.stage1.decoder_block4_resunit_conv1_bias,
        *weights.stage1.decoder_block4_resunit_snake2_alpha,
        *weights.stage1.decoder_block4_resunit_conv2_weight,
        *weights.stage1.decoder_block4_resunit_conv2_bias,
        weights.stage1.decoder_final_snake_alpha,
        weights.stage1.decoder_final_conv_weight,
        weights.stage1.decoder_final_conv_bias,
    )
