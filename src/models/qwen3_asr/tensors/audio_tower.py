"""Logical tensors for Qwen3-ASR audio tower frames."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
    TensorSpec,
)
from models.qwen3_asr.tensors.audio_tower_layer import (
    Qwen3AsrAudioEncoderLayerTensors,
    declare_qwen3_asr_audio_encoder_layer_tensors,
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerTensors:
    input_features: LogicalTensor
    feature_lens: LogicalTensor
    padded_feature: LogicalTensor
    conv2d1_weight: LogicalTensor
    conv2d1_bias: LogicalTensor
    conv2d1_gelu: LogicalTensor
    conv2d2_weight: LogicalTensor
    conv2d2_bias: LogicalTensor
    conv2d2_gelu: LogicalTensor
    conv2d3_weight: LogicalTensor
    conv2d3_bias: LogicalTensor
    conv2d3_gelu: LogicalTensor
    conv_out_weight: LogicalTensor
    conv_out_add_position: LogicalTensor
    hidden_states: LogicalTensor
    cu_seqlens: LogicalTensor
    layers: tuple[Qwen3AsrAudioEncoderLayerTensors, ...]
    ln_post_weight: LogicalTensor
    ln_post_bias: LogicalTensor
    ln_post: LogicalTensor
    proj1_weight: LogicalTensor
    proj1_bias: LogicalTensor
    proj1_gelu: LogicalTensor
    proj2_weight: LogicalTensor
    proj2_bias: LogicalTensor
    last_hidden_state: LogicalTensor


def declare_qwen3_asr_audio_tower_tensors(
    *,
    input_features_shape: tuple[int, int],
    hidden_size: int = 896,
    output_size: int = 1024,
    downsample_hidden_size: int = 480,
    encoder_layers: int = 18,
    encoder_ffn_dim: int = 3584,
) -> Qwen3AsrAudioTowerTensors:
    input_height, input_width = input_features_shape
    chunk_size = 100
    batch = (input_width + chunk_size - 1) // chunk_size
    chunk_width = min(chunk_size, input_width)
    chunk_lengths = [min(chunk_size, input_width - chunk * chunk_size) for chunk in range(batch)]
    chunk_aftercnn_lengths = [_conv_downsample_len(length) for length in chunk_lengths]
    conv1_height = (input_height + 1) // 2
    conv2_height = (conv1_height + 1) // 2
    conv3_height = (conv2_height + 1) // 2
    conv1_chunk_width = (chunk_width + 1) // 2
    conv2_chunk_width = (conv1_chunk_width + 1) // 2
    conv3_width = (conv2_chunk_width + 1) // 2
    hidden_rows = sum(chunk_aftercnn_lengths)
    window_aftercnn = conv3_width * 8
    cu_seqlens_count = (hidden_rows + window_aftercnn - 1) // window_aftercnn + 1
    conv_out_input_features = downsample_hidden_size * conv3_height
    weight_common = {
        "role": TensorRole.WEIGHT,
        "memory": MemoryClass.MODEL_WEIGHT,
        "lifetime": TensorLifetime.MODEL,
    }
    output_common = {
        "role": TensorRole.OUTPUT,
        "memory": MemoryClass.HOST_OUTPUT,
        "lifetime": TensorLifetime.FRAME,
    }
    activation_common = {
        "role": TensorRole.ACTIVATION,
        "memory": MemoryClass.FRAME_WORKSPACE,
        "lifetime": TensorLifetime.FRAME,
    }

    def weight(name: str, shape: tuple[int, ...]) -> LogicalTensor:
        return LogicalTensor(
            name=name, spec=TensorSpec(dtype="bfloat16", shape=shape), **weight_common
        )

    def activation(name: str, shape: tuple[int, ...]) -> LogicalTensor:
        return LogicalTensor(
            name=name, spec=TensorSpec(dtype="float32", shape=shape), **activation_common
        )

    padded_hidden_shape = (batch, conv3_width, hidden_size)
    hidden_shape = (hidden_rows, hidden_size)

    return Qwen3AsrAudioTowerTensors(
        input_features=LogicalTensor(
            name="qwen3_asr.audio_tower.input_features",
            spec=TensorSpec(dtype="float32", shape=(input_height, input_width)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        feature_lens=LogicalTensor(
            name="qwen3_asr.audio_tower.feature_lens",
            spec=TensorSpec(dtype="int64", shape=(1,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        padded_feature=activation(
            "qwen3_asr.audio_tower.padded_feature", (batch, 1, input_height, chunk_width)
        ),
        conv2d1_weight=weight(
            "thinker.audio_tower.conv2d1.weight", (downsample_hidden_size, 1, 3, 3)
        ),
        conv2d1_bias=weight("thinker.audio_tower.conv2d1.bias", (downsample_hidden_size,)),
        conv2d1_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d1.gelu",
            spec=TensorSpec(
                dtype="float32",
                shape=(batch, downsample_hidden_size, conv1_height, conv1_chunk_width),
            ),
            **output_common,
        ),
        conv2d2_weight=weight(
            "thinker.audio_tower.conv2d2.weight",
            (downsample_hidden_size, downsample_hidden_size, 3, 3),
        ),
        conv2d2_bias=weight("thinker.audio_tower.conv2d2.bias", (downsample_hidden_size,)),
        conv2d2_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d2.gelu",
            spec=TensorSpec(
                dtype="float32",
                shape=(batch, downsample_hidden_size, conv2_height, conv2_chunk_width),
            ),
            **output_common,
        ),
        conv2d3_weight=weight(
            "thinker.audio_tower.conv2d3.weight",
            (downsample_hidden_size, downsample_hidden_size, 3, 3),
        ),
        conv2d3_bias=weight("thinker.audio_tower.conv2d3.bias", (downsample_hidden_size,)),
        conv2d3_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d3.gelu",
            spec=TensorSpec(
                dtype="float32", shape=(batch, downsample_hidden_size, conv3_height, conv3_width)
            ),
            **output_common,
        ),
        conv_out_weight=weight(
            "thinker.audio_tower.conv_out.weight", (hidden_size, conv_out_input_features)
        ),
        conv_out_add_position=LogicalTensor(
            name="qwen3_asr.audio_tower.conv_out.add_position",
            spec=TensorSpec(dtype="float32", shape=padded_hidden_shape),
            **output_common,
        ),
        hidden_states=LogicalTensor(
            name="qwen3_asr.audio_tower.hidden_states",
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            compare=ComparePolicy(kind="tensor", rtol=2e-3, atol=2e-2),
            pytorch_probe=PyTorchProbe(
                kind="module_input",
                target="layers.0.self_attn_layer_norm" if encoder_layers else "ln_post",
                index=0,
            ),
            **output_common,
        ),
        cu_seqlens=LogicalTensor(
            name="qwen3_asr.audio_tower.cu_seqlens",
            spec=TensorSpec(dtype="uint32", shape=(cu_seqlens_count,)),
            compare=ComparePolicy(kind="tensor", rtol=0.0, atol=0.0) if encoder_layers else None,
            pytorch_probe=PyTorchProbe(kind="module_input", target="layers.0", index=1)
            if encoder_layers
            else None,
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
        ),
        layers=tuple(
            declare_qwen3_asr_audio_encoder_layer_tensors(
                layer=layer,
                hidden_shape=hidden_shape,
                hidden_size=hidden_size,
                encoder_ffn_dim=encoder_ffn_dim,
            )
            for layer in range(encoder_layers)
        ),
        ln_post_weight=weight("thinker.audio_tower.ln_post.weight", (hidden_size,)),
        ln_post_bias=weight("thinker.audio_tower.ln_post.bias", (hidden_size,)),
        ln_post=LogicalTensor(
            name="qwen3_asr.audio_tower.ln_post",
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            compare=ComparePolicy(kind="tensor", rtol=2e-3, atol=2e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="ln_post"),
            **output_common,
        ),
        proj1_weight=weight("thinker.audio_tower.proj1.weight", (hidden_size, hidden_size)),
        proj1_bias=weight("thinker.audio_tower.proj1.bias", (hidden_size,)),
        proj1_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.proj1.gelu",
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            **output_common,
        ),
        proj2_weight=weight("thinker.audio_tower.proj2.weight", (output_size, hidden_size)),
        proj2_bias=weight("thinker.audio_tower.proj2.bias", (output_size,)),
        last_hidden_state=LogicalTensor(
            name="qwen3_asr.audio_tower.last_hidden_state",
            spec=TensorSpec(dtype="float32", shape=(hidden_rows, output_size)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(
                kind="module_output", target="", selector="last_hidden_state"
            ),
        ),
    )


def _conv_downsample_len(length: int) -> int:
    return (((length + 1) // 2 + 1) // 2 + 1) // 2
