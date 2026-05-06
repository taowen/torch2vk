"""Logical tensors for Qwen3-ASR text prefill/decode frames."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)
from models.qwen3_asr.tensors.text_layer import (
    Qwen3AsrTextLayerTensors,
    declare_qwen3_asr_text_layer_tensors,
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrTextPrefillTensors:
    input_ids: LogicalTensor
    attention_mask: LogicalTensor
    input_features: LogicalTensor | None
    feature_attention_mask: LogicalTensor | None
    position_ids: LogicalTensor
    rope_cos: LogicalTensor
    rope_sin: LogicalTensor
    audio_features: LogicalTensor
    audio_scatter_mask: LogicalTensor
    embed_tokens_weight: LogicalTensor
    inputs_embeds: LogicalTensor
    layers: tuple[Qwen3AsrTextLayerTensors, ...]
    norm_weight: LogicalTensor
    final_norm: LogicalTensor
    lm_head_weight: LogicalTensor
    logits: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3AsrTextDecodeTensors:
    input_ids: LogicalTensor
    attention_mask: LogicalTensor
    position_ids: LogicalTensor
    rope_cos: LogicalTensor
    rope_sin: LogicalTensor
    cache_position: LogicalTensor
    embed_tokens_weight: LogicalTensor
    inputs_embeds: LogicalTensor
    layers: tuple[Qwen3AsrTextLayerTensors, ...]
    norm_weight: LogicalTensor
    final_norm: LogicalTensor
    lm_head_weight: LogicalTensor
    lm_head_select_scratch: LogicalTensor
    logits: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3AsrTokenSelectTensors:
    eos_token_ids: LogicalTensor
    next_token: LogicalTensor
    done: LogicalTensor
    generated_tokens: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3AsrTextTensors:
    prefill: Qwen3AsrTextPrefillTensors
    decode: Qwen3AsrTextDecodeTensors
    token_select: Qwen3AsrTokenSelectTensors


def declare_qwen3_asr_text_tensors(
    *,
    prompt_length: int,
    audio_tokens: int,
    max_sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    decoder_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    eos_token_ids: tuple[int, ...] = (151645, 151643),
    audio_features: LogicalTensor | None = None,
    pytorch_input_features_shape: tuple[int, ...] | None = None,
    pytorch_feature_attention_mask_shape: tuple[int, ...] | None = None,
) -> Qwen3AsrTextTensors:
    if prompt_length <= 0:
        raise ValueError(f"prompt_length must be positive, got {prompt_length}")
    if max_sequence_length < prompt_length:
        raise ValueError(
            f"max_sequence_length {max_sequence_length} must cover prompt_length {prompt_length}"
        )
    if audio_features is not None:
        audio_features.validate_declaration()
        if audio_features.spec.dtype != "float32":
            raise ValueError(
                f"audio_features must be float32, got {audio_features.spec.dtype}"
            )
        if audio_features.concrete_shape != (audio_tokens, hidden_size):
            raise ValueError(
                "audio_features shape must match "
                f"(audio_tokens, hidden_size)=({audio_tokens}, {hidden_size}), "
                f"got {audio_features.concrete_shape}"
            )
    prefill_audio_features = audio_features or _activation(
        "qwen3_asr.text_prefill.audio_features", "float32", (audio_tokens, hidden_size)
    )
    shared_layer_caches = tuple(
        (
            _state(
                f"qwen3_asr.text.layers.{layer:02d}.key_cache",
                "float32",
                (1, num_key_value_heads, max_sequence_length, head_dim),
                semantic=TensorSemantic.KV_CACHE,
            ),
            _state(
                f"qwen3_asr.text.layers.{layer:02d}.value_cache",
                "float32",
                (1, num_key_value_heads, max_sequence_length, head_dim),
                semantic=TensorSemantic.KV_CACHE,
            ),
        )
        for layer in range(decoder_layers)
    )
    prefill = Qwen3AsrTextPrefillTensors(
        input_ids=_input("qwen3_asr.text_prefill.input_ids", "int64", (1, prompt_length)),
        attention_mask=_input("qwen3_asr.text_prefill.attention_mask", "int64", (1, prompt_length)),
        input_features=None if pytorch_input_features_shape is None else _input(
            "qwen3_asr.text_prefill.input_features", "float32", pytorch_input_features_shape,
        ),
        feature_attention_mask=None
        if pytorch_feature_attention_mask_shape is None
        else _input(
            "qwen3_asr.text_prefill.feature_attention_mask",
            "int64",
            pytorch_feature_attention_mask_shape,
        ),
        position_ids=_input("qwen3_asr.text_prefill.position_ids", "int64", (3, 1, prompt_length)),
        rope_cos=_state("qwen3_asr.text_prefill.rope_cos", "float32", (1, prompt_length, head_dim)),
        rope_sin=_state("qwen3_asr.text_prefill.rope_sin", "float32", (1, prompt_length, head_dim)),
        audio_features=prefill_audio_features,
        audio_scatter_mask=_activation(
            "qwen3_asr.text_prefill.audio_scatter_mask",
            "uint32",
            (1, prompt_length, hidden_size),
            semantic=TensorSemantic.MASK,
        ),
        embed_tokens_weight=_weight("thinker.model.embed_tokens.weight", (vocab_size, hidden_size)),
        inputs_embeds=_comparable_activation(
            "qwen3_asr.text_prefill.inputs_embeds",
            (1, prompt_length, hidden_size),
            probe=PyTorchProbe(kind="module_input", target="model.layers.0", index=0),
        ),
        layers=tuple(
            declare_qwen3_asr_text_layer_tensors(
                frame_prefix="qwen3_asr.text_prefill",
                layer=layer,
                sequence_length=prompt_length,
                max_sequence_length=max_sequence_length,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                key_cache=shared_layer_caches[layer][0],
                value_cache=shared_layer_caches[layer][1],
            )
            for layer in range(decoder_layers)
        ),
        norm_weight=_weight("thinker.model.norm.weight", (hidden_size,)),
        final_norm=_comparable_activation(
            "qwen3_asr.text_prefill.final_norm",
            (1, prompt_length, hidden_size),
            probe=PyTorchProbe(kind="module_output", target="model.norm"),
        ),
        lm_head_weight=_weight("thinker.lm_head.weight", (vocab_size, hidden_size)),
        logits=LogicalTensor(
            name="qwen3_asr.text_prefill.logits",
            spec=TensorSpec(dtype="float32", shape=(1, prompt_length, vocab_size)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
            semantic=TensorSemantic.LOGITS,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", selector="logits"),
        ),
    )
    decode = Qwen3AsrTextDecodeTensors(
        input_ids=_input("qwen3_asr.text_decode.input_ids", "int64", (1, 1)),
        attention_mask=_input("qwen3_asr.text_decode.attention_mask", "int64", (1, max_sequence_length)),
        position_ids=_input("qwen3_asr.text_decode.position_ids", "int64", (3, 1, 1)),
        rope_cos=_state("qwen3_asr.text_decode.rope_cos", "float32", (1, 1, head_dim)),
        rope_sin=_state("qwen3_asr.text_decode.rope_sin", "float32", (1, 1, head_dim)),
        cache_position=_input("qwen3_asr.text_decode.cache_position", "int64", (1,)),
        embed_tokens_weight=_weight("thinker.model.embed_tokens.weight", (vocab_size, hidden_size)),
        inputs_embeds=_activation("qwen3_asr.text_decode.inputs_embeds", "float32", (1, 1, hidden_size)),
        layers=tuple(
            declare_qwen3_asr_text_layer_tensors(
                frame_prefix="qwen3_asr.text_decode",
                layer=layer,
                sequence_length=1,
                max_sequence_length=max_sequence_length,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                key_cache=shared_layer_caches[layer][0],
                value_cache=shared_layer_caches[layer][1],
            )
            for layer in range(decoder_layers)
        ),
        norm_weight=_weight("thinker.model.norm.weight", (hidden_size,)),
        final_norm=_comparable_activation(
            "qwen3_asr.text_decode.final_norm",
            (1, 1, hidden_size),
            probe=PyTorchProbe(kind="module_output", target="model.norm"),
        ),
        lm_head_weight=_weight("thinker.lm_head.weight", (vocab_size, hidden_size)),
        lm_head_select_scratch=LogicalTensor(
            name="qwen3_asr.text_decode.lm_head_select_scratch",
            spec=TensorSpec(dtype="float32", shape=(2 * ((vocab_size + 3) // 4),)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
        ),
        logits=LogicalTensor(
            name="qwen3_asr.text_decode.logits",
            spec=TensorSpec(dtype="float32", shape=(1, 1, vocab_size)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
            semantic=TensorSemantic.LOGITS,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", selector="logits"),
        ),
    )
    token_select = Qwen3AsrTokenSelectTensors(
        eos_token_ids=_input(
            "qwen3_asr.token_select.eos_token_ids", "int64", (len(eos_token_ids),)
        ),
        next_token=LogicalTensor(
            name="qwen3_asr.token_select.next_token",
            spec=TensorSpec(dtype="int64", shape=(1,)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
            semantic=TensorSemantic.TOKEN,
            compare=ComparePolicy(kind="token", rtol=0.0, atol=0.0),
        ),
        done=LogicalTensor(
            name="qwen3_asr.token_select.done",
            spec=TensorSpec(dtype="uint32", shape=(1,)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
            semantic=TensorSemantic.TOKEN,
            compare=ComparePolicy(kind="token", rtol=0.0, atol=0.0),
        ),
        generated_tokens=_state(
            "qwen3_asr.generated_tokens",
            "int64",
            (1, 0),
            semantic=TensorSemantic.TOKEN,
        ),
    )
    return Qwen3AsrTextTensors(prefill=prefill, decode=decode, token_select=token_select)

def _weight(name: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="bfloat16", shape=shape),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
    )


def _input(name: str, dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _state(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=semantic,
    )


def _activation(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
        semantic=semantic,
    )


def _comparable_activation(
    name: str,
    shape: tuple[int, ...],
    *,
    probe: PyTorchProbe,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="float32", shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
        compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
        pytorch_probe=probe,
    )
