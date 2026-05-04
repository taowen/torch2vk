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


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioEncoderLayerWeights:
    self_attn_layer_norm_weight: LogicalTensor
    self_attn_layer_norm_bias: LogicalTensor
    q_proj_weight: LogicalTensor
    q_proj_bias: LogicalTensor
    k_proj_weight: LogicalTensor
    k_proj_bias: LogicalTensor
    v_proj_weight: LogicalTensor
    v_proj_bias: LogicalTensor
    out_proj_weight: LogicalTensor
    out_proj_bias: LogicalTensor
    final_layer_norm_weight: LogicalTensor
    final_layer_norm_bias: LogicalTensor
    fc1_weight: LogicalTensor
    fc1_bias: LogicalTensor
    fc2_weight: LogicalTensor
    fc2_bias: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerWeights:
    conv2d1_weight: LogicalTensor
    conv2d1_bias: LogicalTensor
    conv2d2_weight: LogicalTensor
    conv2d2_bias: LogicalTensor
    conv2d3_weight: LogicalTensor
    conv2d3_bias: LogicalTensor
    conv_out_weight: LogicalTensor
    ln_post_weight: LogicalTensor
    ln_post_bias: LogicalTensor
    proj1_weight: LogicalTensor
    proj1_bias: LogicalTensor
    proj2_weight: LogicalTensor
    proj2_bias: LogicalTensor
    layers: tuple[Qwen3AsrAudioEncoderLayerWeights, ...]


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioEncoderLayerTensors:
    self_attn_layer_norm: LogicalTensor
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    self_attn: LogicalTensor
    out_proj: LogicalTensor
    self_attn_residual: LogicalTensor
    final_layer_norm: LogicalTensor
    fc1_gelu: LogicalTensor
    fc2: LogicalTensor
    output: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerTensors:
    input_features: LogicalTensor
    feature_lens: LogicalTensor
    padded_feature: LogicalTensor
    weights: Qwen3AsrAudioTowerWeights
    conv2d1_gelu: LogicalTensor
    conv2d2_gelu: LogicalTensor
    conv2d3_gelu: LogicalTensor
    conv_out_add_position: LogicalTensor
    layers: tuple[Qwen3AsrAudioEncoderLayerTensors, ...]
    ln_post: LogicalTensor
    proj1_gelu: LogicalTensor
    last_hidden_state: LogicalTensor


def declare_qwen3_asr_audio_tower_tensors(
    *,
    input_shape: tuple[int, int, int, int],
    hidden_size: int = 896,
    output_size: int = 1024,
    downsample_hidden_size: int = 480,
    encoder_layers: int = 18,
    encoder_ffn_dim: int = 3584,
) -> Qwen3AsrAudioTowerTensors:
    batch, _, input_height, input_width = input_shape
    conv1_height, conv1_width = (input_height + 1) // 2, (input_width + 1) // 2
    conv2_height, conv2_width = (conv1_height + 1) // 2, (conv1_width + 1) // 2
    conv3_height, conv3_width = (conv2_height + 1) // 2, (conv2_width + 1) // 2
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
        return LogicalTensor(name=name, spec=TensorSpec(dtype="bfloat16", shape=shape), **weight_common)

    def activation(name: str, shape: tuple[int, ...]) -> LogicalTensor:
        return LogicalTensor(name=name, spec=TensorSpec(dtype="float32", shape=shape), **activation_common)

    def layer_weights(layer: int) -> Qwen3AsrAudioEncoderLayerWeights:
        prefix = f"thinker.audio_tower.layers.{layer}"
        return Qwen3AsrAudioEncoderLayerWeights(
            self_attn_layer_norm_weight=weight(f"{prefix}.self_attn_layer_norm.weight", (hidden_size,)),
            self_attn_layer_norm_bias=weight(f"{prefix}.self_attn_layer_norm.bias", (hidden_size,)),
            q_proj_weight=weight(f"{prefix}.self_attn.q_proj.weight", (hidden_size, hidden_size)),
            q_proj_bias=weight(f"{prefix}.self_attn.q_proj.bias", (hidden_size,)),
            k_proj_weight=weight(f"{prefix}.self_attn.k_proj.weight", (hidden_size, hidden_size)),
            k_proj_bias=weight(f"{prefix}.self_attn.k_proj.bias", (hidden_size,)),
            v_proj_weight=weight(f"{prefix}.self_attn.v_proj.weight", (hidden_size, hidden_size)),
            v_proj_bias=weight(f"{prefix}.self_attn.v_proj.bias", (hidden_size,)),
            out_proj_weight=weight(f"{prefix}.self_attn.out_proj.weight", (hidden_size, hidden_size)),
            out_proj_bias=weight(f"{prefix}.self_attn.out_proj.bias", (hidden_size,)),
            final_layer_norm_weight=weight(f"{prefix}.final_layer_norm.weight", (hidden_size,)),
            final_layer_norm_bias=weight(f"{prefix}.final_layer_norm.bias", (hidden_size,)),
            fc1_weight=weight(f"{prefix}.fc1.weight", (encoder_ffn_dim, hidden_size)),
            fc1_bias=weight(f"{prefix}.fc1.bias", (encoder_ffn_dim,)),
            fc2_weight=weight(f"{prefix}.fc2.weight", (hidden_size, encoder_ffn_dim)),
            fc2_bias=weight(f"{prefix}.fc2.bias", (hidden_size,)),
        )

    hidden_shape = (batch * conv3_width, hidden_size)

    def layer_tensors(layer: int) -> Qwen3AsrAudioEncoderLayerTensors:
        prefix = f"qwen3_asr.audio_tower.layers.{layer:02d}"
        return Qwen3AsrAudioEncoderLayerTensors(
            self_attn_layer_norm=activation(f"{prefix}.self_attn_layer_norm", hidden_shape),
            q_proj=activation(f"{prefix}.self_attn.q_proj", hidden_shape),
            k_proj=activation(f"{prefix}.self_attn.k_proj", hidden_shape),
            v_proj=activation(f"{prefix}.self_attn.v_proj", hidden_shape),
            self_attn=activation(f"{prefix}.self_attn", hidden_shape),
            out_proj=activation(f"{prefix}.self_attn.out_proj", hidden_shape),
            self_attn_residual=activation(f"{prefix}.self_attn_residual", hidden_shape),
            final_layer_norm=activation(f"{prefix}.final_layer_norm", hidden_shape),
            fc1_gelu=activation(f"{prefix}.fc1.gelu", (batch * conv3_width, encoder_ffn_dim)),
            fc2=activation(f"{prefix}.fc2", hidden_shape),
            output=activation(f"{prefix}.output", hidden_shape),
        )

    weights = Qwen3AsrAudioTowerWeights(
        conv2d1_weight=weight("thinker.audio_tower.conv2d1.weight", (downsample_hidden_size, 1, 3, 3)),
        conv2d1_bias=weight("thinker.audio_tower.conv2d1.bias", (downsample_hidden_size,)),
        conv2d2_weight=weight(
            "thinker.audio_tower.conv2d2.weight", (downsample_hidden_size, downsample_hidden_size, 3, 3)
        ),
        conv2d2_bias=weight("thinker.audio_tower.conv2d2.bias", (downsample_hidden_size,)),
        conv2d3_weight=weight(
            "thinker.audio_tower.conv2d3.weight", (downsample_hidden_size, downsample_hidden_size, 3, 3)
        ),
        conv2d3_bias=weight("thinker.audio_tower.conv2d3.bias", (downsample_hidden_size,)),
        conv_out_weight=weight("thinker.audio_tower.conv_out.weight", (hidden_size, conv_out_input_features)),
        ln_post_weight=weight("thinker.audio_tower.ln_post.weight", (hidden_size,)),
        ln_post_bias=weight("thinker.audio_tower.ln_post.bias", (hidden_size,)),
        proj1_weight=weight("thinker.audio_tower.proj1.weight", (hidden_size, hidden_size)),
        proj1_bias=weight("thinker.audio_tower.proj1.bias", (hidden_size,)),
        proj2_weight=weight("thinker.audio_tower.proj2.weight", (output_size, hidden_size)),
        proj2_bias=weight("thinker.audio_tower.proj2.bias", (output_size,)),
        layers=tuple(layer_weights(layer) for layer in range(encoder_layers)),
    )

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
            spec=TensorSpec(dtype="int64", shape=(batch,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        padded_feature=activation("qwen3_asr.audio_tower.padded_feature", (batch, 1, input_height, input_width)),
        weights=weights,
        conv2d1_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d1.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, downsample_hidden_size, conv1_height, conv1_width)),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
            pytorch_probe=PyTorchProbe(kind="module_input", target="conv2d2", index=0),
            **output_common,
        ),
        conv2d2_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d2.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, downsample_hidden_size, conv2_height, conv2_width)),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
            pytorch_probe=PyTorchProbe(kind="module_input", target="conv2d3", index=0),
            **output_common,
        ),
        conv2d3_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d3.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, downsample_hidden_size, conv3_height, conv3_width)),
            **output_common,
        ),
        conv_out_add_position=LogicalTensor(
            name="qwen3_asr.audio_tower.conv_out.add_position",
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            compare=ComparePolicy(kind="tensor", rtol=2e-3, atol=2e-2),
            pytorch_probe=PyTorchProbe(
                kind="module_input",
                target="layers.0.self_attn_layer_norm" if encoder_layers else "ln_post",
                index=0,
            ),
            **output_common,
        ),
        layers=tuple(layer_tensors(layer) for layer in range(encoder_layers)),
        ln_post=LogicalTensor(
            name="qwen3_asr.audio_tower.ln_post",
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            **output_common,
        ),
        proj1_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.proj1.gelu",
            spec=TensorSpec(dtype="float32", shape=hidden_shape),
            **output_common,
        ),
        last_hidden_state=LogicalTensor(
            name="qwen3_asr.audio_tower.last_hidden_state",
            spec=TensorSpec(dtype="float32", shape=(batch * conv3_width, output_size)),
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", selector="last_hidden_state"),
            **output_common,
        ),
    )
