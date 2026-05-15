"""PyTorch/Vulkan comparison entry points for optimized Qwen3-ASR.

Run from project root:
    .venv/bin/python -m models.optimized_qwen3_asr.compare
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.optimized_qwen3_asr import reference
from models.optimized_qwen3_asr.export_gguf import export_qwen3_asr_q4_k_m_gguf
from models.optimized_qwen3_asr.dispatch.audio_encoder import run_audio_encoder
from models.optimized_qwen3_asr.dispatch.audio_inject import run_audio_inject
from models.optimized_qwen3_asr.dispatch.decode_embed import run_decode_embed
from models.optimized_qwen3_asr.dispatch.decode_layer import run_decode_layer
from models.optimized_qwen3_asr.dispatch.decode_norm import run_decode_norm
from models.optimized_qwen3_asr.dispatch.embed_tokens import run_embed_tokens
from models.optimized_qwen3_asr.dispatch.text_layer import run_text_last_layer_tail, run_text_layer
from models.optimized_qwen3_asr.dispatch.text_norm import run_text_norm
from models.optimized_qwen3_asr.pytorch_modules import (
    AudioEncoderReference,
    AudioInjectReference,
    TextLayerReference,
    TextReferenceState,
    TokenSelectReference,
    TokenStoreReference,
    audio_position_embedding_shape,
    preprocess_audio_inputs,
)
from models.optimized_qwen3_asr.shaders.lm_head_q6_k_argmax_partial_f16 import (
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16,
)
from models.optimized_qwen3_asr.shaders.qwen3_asr_token_store_eos import (
    QWEN3_ASR_TOKEN_STORE_EOS,
)
from models.optimized_qwen3_asr.shaders.qwen3_token_select_reduce_chunks_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
)
from models.optimized_qwen3_asr.shaders.qwen3_token_select_reduce_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_F32,
)
from models.optimized_qwen3_asr.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.host_array import as_float16_array, as_float16_attention_mask
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.rope_table import run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader

get_shader = make_shader_loader("models.optimized_qwen3_asr.shaders")


@dataclass(slots=True)
class _QwenCompareReferences:
    audio_encoder: AudioEncoderReference
    audio_inject: AudioInjectReference
    lm_head: nn.Module
    text_layers: tuple[TextLayerReference, ...]
    decode_layers: tuple[TextLayerReference, ...]
    token_select: TokenSelectReference
    token_store: TokenStoreReference


def _require_gpu_output(tensor: LogicalTensor) -> None:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} did not produce a GPU buffer")


def _vulkan_input(rt: RuntimeSession, tensor: LogicalTensor) -> reference.ReferenceInput:
    return np.ascontiguousarray(rt.readback(tensor))


def _vulkan_request_state(rt: RuntimeSession, tensor: LogicalTensor) -> np.ndarray:
    _require_gpu_output(tensor)
    return np.ascontiguousarray(rt.read_request_state(tensor))


def _reference_lm_head_logits(
    lm_head: nn.Module, hidden_states: reference.ReferenceInput
) -> np.ndarray:
    hidden = torch.from_numpy(np.ascontiguousarray(hidden_states)).cuda().float()
    with torch.no_grad():
        logits = lm_head(hidden)
    return np.ascontiguousarray(logits.detach().cpu().float().numpy())


def _run_lm_head_select(rt: RuntimeSession, *, x: LogicalTensor) -> None:
    tensors = model_tensors()
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16(
        rt,
        x=x,
        weight=tensors.lm_head.p_weight,
        partial_scores=tensors.lm_head_partial_scores,
        partial_tokens=tensors.lm_head_partial_tokens,
    )
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32(
        rt,
        scores=tensors.lm_head_partial_scores,
        tokens=tensors.lm_head_partial_tokens,
        chunk_scores=tensors.lm_head_chunk_scores,
        chunk_tokens=tensors.lm_head_chunk_tokens,
    )
    QWEN3_TOKEN_SELECT_REDUCE_F32(
        rt,
        partial_scores=tensors.lm_head_chunk_scores,
        partial_tokens=tensors.lm_head_chunk_tokens,
        eos_token_ids=tensors.eos_token_ids,
        next_token=tensors.next_token,
        done=tensors.done,
    )


def _run_token_store(
    rt: RuntimeSession,
    *,
    next_token: LogicalTensor,
    token_index: LogicalTensor,
    done: LogicalTensor,
    generated_tokens: LogicalTensor,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
    frame_name: str,
) -> None:
    with rt.frame(frame_name):
        QWEN3_ASR_TOKEN_STORE_EOS(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )


def _run_rope_table(
    rt: RuntimeSession,
    *,
    phase: str,
    rope_theta: float,
    frame_name: str,
) -> None:
    tensors = model_tensors()
    if phase == "prefill":
        rope_t = tensors.prefill_rope
    elif phase == "decode":
        rope_t = tensors.decode_rope
    else:
        raise ValueError(f"unknown rope phase: {phase}")
    run_rope_table_f32(
        rt,
        start_position=rope_t.start_position,
        theta=rope_theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


def _decode_step_inputs(
    *,
    cache_position: int,
    token_index_value: int,
) -> dict[LogicalTensor, np.ndarray]:
    tensors = model_tensors()
    if not tensors.decode_layers:
        raise ValueError("decode_layers must not be empty")
    return {
        tensors.decode_layers[0].cache_position: np.array([cache_position], dtype=np.int64),
        tensors.token_index: np.array([token_index_value], dtype=np.int64),
    }


def _read_selected_token(rt: RuntimeSession, next_token: LogicalTensor) -> int:
    _require_gpu_output(next_token)
    return int(rt.read_request_state(next_token).reshape(-1)[0])


def _load_qwen_reference_model(model_dir: Path) -> Qwen3ASRForConditionalGeneration:
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return cast(Qwen3ASRForConditionalGeneration, model)


def _build_compare_references(
    model: Qwen3ASRForConditionalGeneration,
) -> _QwenCompareReferences:
    thinker = cast(nn.Module, getattr(model, "thinker"))
    reference.set_model(model)
    text_state = TextReferenceState(thinker)
    decode_state = text_state
    audio_tower = cast(nn.Module, thinker.get_submodule("audio_tower"))
    lm_head = cast(nn.Module, thinker.get_submodule("lm_head"))
    return _QwenCompareReferences(
        audio_encoder=AudioEncoderReference(audio_tower),
        audio_inject=AudioInjectReference(),
        lm_head=lm_head,
        text_layers=tuple(
            TextLayerReference(text_state, layer_idx, prefill=True)
            for layer_idx in range(len(text_state.layers))
        ),
        decode_layers=tuple(
            TextLayerReference(decode_state, layer_idx, prefill=False)
            for layer_idx in range(len(decode_state.layers))
        ),
        token_select=TokenSelectReference(),
        token_store=TokenStoreReference(),
    )


def _compare_token_select(
    rt: RuntimeSession,
    *,
    frame_name: str,
    logits: reference.ReferenceInput,
    eos_token_ids: np.ndarray,
    refs: _QwenCompareReferences,
) -> None:
    reference.run_token_select(
        rt,
        refs.token_select,
        name=frame_name,
        logits=logits,
        eos_token_ids=eos_token_ids,
    )


def _compare_token_store(
    rt: RuntimeSession,
    *,
    frame_name: str,
    refs: _QwenCompareReferences,
    next_token: reference.ReferenceInput,
    token_index: reference.ReferenceInput,
    done: reference.ReferenceInput,
    generated_tokens: reference.ReferenceInput,
    generated_length: reference.ReferenceInput,
    stopped: reference.ReferenceInput,
) -> None:
    reference.run_token_store(
        rt,
        refs.token_store,
        name=frame_name,
        next_token=next_token,
        token_index=token_index,
        done=done,
        generated_tokens=generated_tokens,
        generated_length=generated_length,
        stopped=stopped,
    )


def _run_decode_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    cache_position: np.ndarray,
    eos_token_ids: np.ndarray,
    token_index: np.ndarray,
    refs: _QwenCompareReferences,
) -> int:
    tensors = model_tensors()
    decode_input = _vulkan_request_state(rt, tensors.next_token).astype(np.int64, copy=False)
    with rt.frame(f"spike.decode.{step:04d}"):
        run_decode_embed(rt)
        reference.run_decode_embed(
            rt,
            step=step,
            input=decode_input,
        )
        ref_hidden = _vulkan_input(rt, tensors.decode_embed.embedding)
        cos = _vulkan_input(rt, tensors.decode_rope.cos)
        sin = _vulkan_input(rt, tensors.decode_rope.sin)
        for layer_idx in range(len(tensors.decode_layers)):
            key_cache_before = _vulkan_request_state(rt, tensors.key_caches[layer_idx])
            value_cache_before = _vulkan_request_state(rt, tensors.value_caches[layer_idx])
            run_decode_layer(rt, layer_idx)
            reference.run_decode_layer(
                rt,
                refs.decode_layers[layer_idx],
                step=step,
                layer_idx=layer_idx,
                hidden_states=ref_hidden,
                position_embeddings_0=cos,
                position_embeddings_1=sin,
                cache_position=cache_position,
                key_cache=key_cache_before,
                value_cache=value_cache_before,
            )
            ref_hidden = _vulkan_input(rt, tensors.decode_layers[layer_idx].add_7)
        run_decode_norm(rt)
        reference.run_decode_norm(
            rt,
            step=step,
            hidden_states=ref_hidden,
        )
        ref_hidden = _vulkan_input(rt, tensors.decode_norm.mul_1)
        _run_lm_head_select(rt, x=tensors.decode_norm.mul_1)
        _compare_token_select(
            rt,
            frame_name=f"spike.decode.{step:04d}.token_select",
            logits=_reference_lm_head_logits(refs.lm_head, ref_hidden),
            eos_token_ids=eos_token_ids,
            refs=refs,
        )
        generated_tokens_before = _vulkan_request_state(rt, tensors.generated_tokens)
        generated_length_before = _vulkan_request_state(rt, tensors.generated_length)
        stopped_before = _vulkan_request_state(rt, tensors.stopped)
        QWEN3_ASR_TOKEN_STORE_EOS(
            rt,
            next_token=tensors.next_token,
            token_index=tensors.token_index,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )
        _compare_token_store(
            rt,
            frame_name=f"spike.decode.{step:04d}.token_store",
            refs=refs,
            next_token=_vulkan_request_state(rt, tensors.next_token),
            token_index=token_index,
            done=_vulkan_request_state(rt, tensors.done),
            generated_tokens=generated_tokens_before,
            generated_length=generated_length_before,
            stopped=stopped_before,
        )
    return int(_vulkan_request_state(rt, tensors.next_token).reshape(-1)[0])


# ==============================================================
# Main pipeline
# ==============================================================


def compare_decode_steps(
    *,
    max_new_tokens: int = 1,
    compare_prefill_layers: int = 2,
) -> str:
    if max_new_tokens <= 0 or max_new_tokens > 64:
        raise ValueError(f"max_new_tokens must be in [1, 64], got {max_new_tokens}")
    if compare_prefill_layers < 0:
        raise ValueError(
            f"compare_prefill_layers must be non-negative, got {compare_prefill_layers}"
        )
    wav_path = Path("tests/fixtures/qwen3_asr_asknot.wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"Test wav not found at {wav_path}")

    print("Preparing inputs...")
    model_dir = resolve_cached_model(REPO_ID)
    gguf_path = export_qwen3_asr_q4_k_m_gguf(model_dir=model_dir)
    config_payload = (Path(model_dir) / "config.json").read_text()

    devnull = open(os.devnull, "w")
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    try:
        config = Qwen3ASRConfig(**json.loads(config_payload))
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        devnull.close()

    ac = config.thinker_config.audio_config
    tc = config.thinker_config.text_config
    rope_theta = float(getattr(tc, "rope_theta", 5_000_000.0))
    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(wav_path))
    prompt_length = prepared.prompt_length
    max_sequence_length = prepared.prompt_length + 64
    audio_feature_length = int(
        np.asarray(prepared.feature_attention_mask).sum(axis=-1).reshape(-1)[0]
    )
    audio_position_shape = audio_position_embedding_shape(
        feature_length=audio_feature_length,
        d_model=ac.d_model,
    )
    preprocessed = preprocess_audio_inputs(
        prepared.input_ids,
        prepared.input_features,
        prepared.feature_attention_mask,
        position_embedding_shape=audio_position_shape,
        d_model=ac.d_model,
    )
    audio_chunk_count = int(preprocessed["padded_feature"].shape[0])
    audio_sequence_length = int(preprocessed["compact_index"].shape[0])

    # === Create all tensor objects upfront ===
    print("Declaring tensors...")
    eos_token_ids = (151645, 151643)
    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    create_model_tensors(
        input_ids_shape=tuple(int(dim) for dim in prepared.input_ids.shape),
        attention_mask_shape=tuple(int(dim) for dim in prepared.attention_mask.shape),
        input_features_shape=tuple(int(dim) for dim in prepared.input_features.shape),
        feature_attention_mask_shape=tuple(
            int(dim) for dim in prepared.feature_attention_mask.shape
        ),
        prompt_length=prompt_length,
        audio_chunk_count=audio_chunk_count,
        audio_sequence_length=audio_sequence_length,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=tc.num_hidden_layers,
        num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,
        max_new_tokens=max_new_tokens,
        eos_token_count=len(eos_token_ids),
        vocab_size=int(tc.vocab_size),
    )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_path.parent,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    rt.register_session_tensors({model_tensors().eos_token_ids: eos_token_array})
    print("Loading PyTorch reference for compare...")
    compare_refs = _build_compare_references(_load_qwen_reference_model(Path(model_dir)))

    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float16,
    )
    rt.initialize_request_state(
        {cache: zero_cache for cache in model_tensors().key_caches + model_tensors().value_caches}
    )

    # === Audio Tower ===
    print("\n=== Phase 1: Audio Tower ===")
    print(f"  hidden_states after compact: {model_tensors().audio_encoder.index_select.spec.shape}")
    print(f"  audio encoder ({model_tensors().audio_encoder.x.spec.shape})...")
    rt.register_inputs(
        {
            model_tensors().audio_encoder.x: as_float16_array(preprocessed["padded_feature"]),
            model_tensors().audio_encoder.position_embedding: as_float16_array(
                preprocessed["position_embedding"]
            ),
            model_tensors().audio_encoder.compact_index: preprocessed["compact_index"],
            model_tensors().audio_encoder.attention_mask: as_float16_attention_mask(
                preprocessed["audio_attention_mask"]
            ),
        }
    )
    with rt.frame("spike.audio"):
        run_audio_encoder(rt)
        reference.run_audio_encoder(
            rt,
            compare_refs.audio_encoder,
            x=as_float16_array(preprocessed["padded_feature"]),
            position_embedding=as_float16_array(preprocessed["position_embedding"]),
            compact_index=preprocessed["compact_index"],
            attention_mask=as_float16_attention_mask(preprocessed["audio_attention_mask"]),
        )
        ref_audio_features = _vulkan_input(rt, model_tensors().audio_encoder.linear_110)
    _require_gpu_output(model_tensors().audio_encoder.linear_110)
    print(f"  Audio tower output: {model_tensors().audio_encoder.linear_110.spec.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    audio_positions = preprocessed["audio_positions"]
    if len(audio_positions) > 0:
        audio_start = int(audio_positions[0])
        audio_end = audio_start + len(audio_positions)
        print(f"    Injecting audio [{audio_start}:{audio_end}]")

    rt.register_inputs(
        {
            model_tensors().prefill_rope.start_position: np.array([0], dtype=np.int64),
        }
    )
    _run_rope_table(
        rt,
        phase="prefill",
        rope_theta=rope_theta,
        frame_name="spike.text.prefill_rope",
    )

    # Embedding, audio injection, and decoder layers stay on GPU.
    print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
    prefill_position_ids = np.broadcast_to(
        np.arange(prompt_length, dtype=np.int64)[None, None, :],
        (3, 1, prompt_length),
    ).copy()
    prefill_cache_position = np.arange(prompt_length, dtype=np.int64)
    rt.initialize_request_state(
        {
            model_tensors().generated_tokens: np.zeros((1, max_new_tokens), dtype=np.int64),
            model_tensors().generated_length: np.zeros((1,), dtype=np.uint32),
            model_tensors().stopped: np.zeros((1,), dtype=np.uint32),
        }
    )
    with rt.frame("spike.text.prefill"):
        rt.register_inputs(
            {
                model_tensors().input_ids: np.ascontiguousarray(
                    prepared.input_ids,
                    dtype=np.int64,
                ),
                model_tensors().attention_mask: np.ascontiguousarray(
                    prepared.attention_mask,
                    dtype=np.int64,
                ),
                model_tensors().input_features: np.ascontiguousarray(
                    prepared.input_features,
                    dtype=np.float32,
                ),
                model_tensors().feature_attention_mask: np.ascontiguousarray(
                    prepared.feature_attention_mask,
                    dtype=np.int64,
                ),
                model_tensors().position_ids: prefill_position_ids,
                model_tensors().audio_inject.audio_positions: preprocessed["audio_positions"],
                model_tensors().text_layers[0].cache_position: prefill_cache_position,
            }
        )
        run_embed_tokens(rt)
        reference.run_embed_tokens(
            rt,
            input=prepared.input_ids,
        )
        ref_hidden = _vulkan_input(rt, model_tensors().embed_tokens.embedding)
        run_audio_inject(rt)
        reference.run_audio_inject(
            rt,
            compare_refs.audio_inject,
            inputs_embeds=ref_hidden,
            audio_positions=preprocessed["audio_positions"],
            audio_features=ref_audio_features,
        )
        ref_hidden = _vulkan_input(rt, model_tensors().audio_inject.index_copy)
        ref_cos = _vulkan_input(rt, model_tensors().prefill_rope.cos)
        ref_sin = _vulkan_input(rt, model_tensors().prefill_rope.sin)
        for layer_idx in range(len(model_tensors().text_layers) - 1):
            key_cache_before = _vulkan_request_state(rt, model_tensors().key_caches[layer_idx])
            value_cache_before = _vulkan_request_state(rt, model_tensors().value_caches[layer_idx])
            run_text_layer(rt, layer_idx)
            if layer_idx < compare_prefill_layers:
                reference.run_text_layer(
                    rt,
                    compare_refs.text_layers[layer_idx],
                    layer_idx=layer_idx,
                    hidden_states=ref_hidden,
                    position_embeddings_0=ref_cos,
                    position_embeddings_1=ref_sin,
                    cache_position=prefill_cache_position,
                    key_cache=key_cache_before,
                    value_cache=value_cache_before,
                )
            ref_hidden = _vulkan_input(rt, model_tensors().text_layers[layer_idx].add_7)
            if layer_idx % 7 == 6:
                print(f"    layer {layer_idx} done")
        run_text_last_layer_tail(rt)
        ref_hidden = _vulkan_input(rt, model_tensors().prefill_last_output)
        run_text_norm(rt)
        reference.run_text_norm(
            rt,
            hidden_states=ref_hidden,
        )
        ref_hidden = _vulkan_input(rt, model_tensors().text_norm.mul_1)
        _run_lm_head_select(rt, x=model_tensors().text_norm.mul_1)
        _compare_token_select(
            rt,
            frame_name="spike.text.token_select",
            logits=_reference_lm_head_logits(compare_refs.lm_head, ref_hidden),
            eos_token_ids=eos_token_array,
            refs=compare_refs,
        )

    print("  lm_head + token_select...")
    rt.register_inputs({model_tensors().token_index: np.array([0], dtype=np.int64)})
    generated_tokens_before = _vulkan_request_state(rt, model_tensors().generated_tokens)
    generated_length_before = _vulkan_request_state(rt, model_tensors().generated_length)
    stopped_before = _vulkan_request_state(rt, model_tensors().stopped)
    _run_token_store(
        rt,
        next_token=model_tensors().next_token,
        token_index=model_tensors().token_index,
        done=model_tensors().done,
        generated_tokens=model_tensors().generated_tokens,
        generated_length=model_tensors().generated_length,
        stopped=model_tensors().stopped,
        frame_name="spike.text.token_store",
    )
    _compare_token_store(
        rt,
        frame_name="spike.text.token_store",
        refs=compare_refs,
        next_token=_vulkan_request_state(rt, model_tensors().next_token),
        token_index=np.array([0], dtype=np.int64),
        done=_vulkan_request_state(rt, model_tensors().done),
        generated_tokens=generated_tokens_before,
        generated_length=generated_length_before,
        stopped=stopped_before,
    )
    first_token = _read_selected_token(rt, model_tensors().next_token)
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    eos_token_set = set(eos_token_ids)
    generated_tokens = [first_token]

    for step in range(max_new_tokens - 1):
        if generated_tokens[-1] in eos_token_set:
            print(f"  EOS at step {step}")
            break

        cache_pos = prompt_length + step

        rt.register_inputs(
            {
                model_tensors().decode_rope.start_position: np.array(
                    [cache_pos],
                    dtype=np.int64,
                ),
            }
        )
        _run_rope_table(
            rt,
            phase="decode",
            rope_theta=rope_theta,
            frame_name=f"spike.decode.rope.{step:04d}",
        )

        decode_step_inputs = _decode_step_inputs(
            cache_position=cache_pos,
            token_index_value=step + 1,
        )
        rt.register_inputs(decode_step_inputs)
        next_token = _run_decode_step_with_compare(
            rt,
            step=step,
            cache_position=np.array([cache_pos], dtype=np.int64),
            eos_token_ids=eos_token_array,
            token_index=np.array([step + 1], dtype=np.int64),
            refs=compare_refs,
        )
        generated_tokens.append(next_token)
        if step < 5 or step % 20 == 0:
            print(f"  Step {step}: token={generated_tokens[-1]}")

    # Decode text
    print("\n=== Result ===")
    stored_length = int(rt.read_request_state(model_tensors().generated_length).reshape(-1)[0])
    generated_tokens = [
        int(token)
        for token in rt.read_request_state(model_tensors().generated_tokens).reshape(-1)[
            :stored_length
        ]
    ]
    print(f"Generated {len(generated_tokens)} tokens")
    text = processor.batch_decode(
        np.array([generated_tokens], dtype=np.int64),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(f"Transcription: {text}")

    rt.close()
    return text


if __name__ == "__main__":
    result = compare_decode_steps()
    print(result)
