"""Generated model-level tensor wiring for quantized OmniVoice."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_omnivoice.tensors.audio_head import AudioHeadTensors, create_audio_head
from models.quantized_omnivoice.tensors.llm_forward import LlmForwardTensors, create_llm_forward
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    bind_logical_tensor_names,
)
from torch2vk.runtime.rope_table import RopeTableTensors, declare_rope_table_tensors
from torch2vk.vulkan.types import (
    CONTIGUOUS_LAYOUT,
    Q4KWordsLayout,
    Q8_0HalfwordsLayout,
    TensorLayout,
    TensorSpec,
    q4_k_words_layout,
    q8_0_halfwords_layout,
)


@dataclass(frozen=True, slots=True)
class QuantizedOmniVoiceTensors:
    text_embedding_weight: LogicalTensor
    audio_embedding_weight: LogicalTensor
    batch_input_ids: LogicalTensor
    batch_audio_mask: LogicalTensor
    attention_mask: LogicalTensor
    audio_mask_id: LogicalTensor
    rng_seed: LogicalTensor
    step_index: LogicalTensor
    unmask_count: LogicalTensor
    tokens: LogicalTensor
    candidate_tokens: LogicalTensor
    candidate_scores: LogicalTensor
    rope: RopeTableTensors
    llm_forward: LlmForwardTensors
    audio_head: AudioHeadTensors


_MODEL_TENSORS: QuantizedOmniVoiceTensors | None = None


def create_model_tensors(*, target_len: int) -> QuantizedOmniVoiceTensors:
    text_embedding_weight = _weight_tensor(
        "float32",
        (151676, 1024),
        "llm.embed_tokens.weight",
        quantize_q8_0=True,
    )
    audio_embedding_weight = _weight_tensor(
        "float32",
        (8200, 1024),
        "audio_embeddings.weight",
        quantize_q8_0=True,
    )
    batch_input_ids = _state_tensor("int64", (2, 8, 85))
    batch_audio_mask = _state_tensor("uint32", (2, 85))
    attention_mask = _state_tensor("float16", (2, 1, 85, 85))
    audio_mask_id = _state_tensor("int64", (1,))
    rng_seed = _state_tensor("uint32", (1,))
    step_index = _host_input_tensor("uint32", (1,))
    unmask_count = _host_input_tensor("uint32", (1,))
    tokens = _state_tensor("int64", (1, 8, target_len))
    candidate_tokens = _state_tensor("int64", (8, target_len))
    candidate_scores = _state_tensor("float32", (8, target_len))
    rope = declare_rope_table_tensors(
        "omnivoice.rope",
        batch=2,
        sequence_length=85,
        head_dim=128,
    )
    hidden_states = _activation_tensor(
        "float16",
        (2, 85, 1024),
    )
    llm_forward = create_llm_forward(
        "omnivoice.llm",
        hidden_states=hidden_states,
        cos=rope.cos,
        sin=rope.sin,
        attention_mask=attention_mask,
    )
    audio_head = create_audio_head(
        "omnivoice.audio_head",
        input=llm_forward.mul_365,
    )

    global _MODEL_TENSORS
    _MODEL_TENSORS = QuantizedOmniVoiceTensors(
        text_embedding_weight=text_embedding_weight,
        audio_embedding_weight=audio_embedding_weight,
        batch_input_ids=batch_input_ids,
        batch_audio_mask=batch_audio_mask,
        attention_mask=attention_mask,
        audio_mask_id=audio_mask_id,
        rng_seed=rng_seed,
        step_index=step_index,
        unmask_count=unmask_count,
        tokens=tokens,
        candidate_tokens=candidate_tokens,
        candidate_scores=candidate_scores,
        rope=rope,
        llm_forward=llm_forward,
        audio_head=audio_head,
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    return _MODEL_TENSORS


def model_tensors() -> QuantizedOmniVoiceTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors must be called before generated dispatch")
    return _MODEL_TENSORS


def _weight_tensor(
    dtype: str,
    shape: tuple[int, ...],
    checkpoint_key: str,
    *,
    quantize_q4_k: bool = True,
    quantize_q8_0: bool = False,
) -> LogicalTensor:
    quantized_layout: Q4KWordsLayout | Q8_0HalfwordsLayout | None = None
    if quantize_q8_0 and dtype == "float32" and len(shape) == 2:
        quantized_layout = q8_0_halfwords_layout(logical_k=shape[1])
    elif quantize_q4_k and dtype == "float32" and len(shape) == 2:
        quantized_layout = q4_k_words_layout(logical_k=shape[1])
    layout: TensorLayout = CONTIGUOUS_LAYOUT
    if quantized_layout is not None:
        if shape[1] % quantized_layout.block_size != 0:
            raise ValueError(f"Quantized weight {checkpoint_key} requires K divisible by {quantized_layout.block_size}")
        layout = quantized_layout
        if isinstance(quantized_layout, Q4KWordsLayout):
            dtype = "uint32"
            shape = (shape[0], shape[1] // quantized_layout.block_size * quantized_layout.words_per_block)
        elif isinstance(quantized_layout, Q8_0HalfwordsLayout):
            dtype = "uint16"
            shape = (shape[0], shape[1] // quantized_layout.block_size * quantized_layout.halfwords_per_block)
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
        checkpoint_key=checkpoint_key,
        layout=layout,
    )


def _host_input_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _state_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _activation_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
    )
