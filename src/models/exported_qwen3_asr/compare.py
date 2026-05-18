"""PyTorch/Vulkan comparison entry points for exported Qwen3-ASR.

Run from project root:
    .venv/bin/python -m models.exported_qwen3_asr.compare
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
from models.exported_qwen3_asr import reference
from models.exported_qwen3_asr.pytorch_modules import (
    AudioEncoderReference,
    AudioInjectReference,
    TextLayerReference,
    TextReferenceState,
    TokenSelectReference,
    TokenStoreReference,
    preprocess_audio_inputs,
)
from models.exported_qwen3_asr.shaders.lm_head_bf16_argmax_partial_f16 import (
    LM_HEAD_BF16_ARGMAX_PARTIAL_F16,
)
from models.exported_qwen3_asr.shaders.qwen3_asr_token_store_eos import (
    QWEN3_ASR_TOKEN_STORE_EOS,
)
from models.exported_qwen3_asr.shaders.qwen3_token_select_reduce_chunks_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
)
from models.exported_qwen3_asr.shaders.qwen3_token_select_reduce_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_F32,
)
from models.exported_qwen3_asr.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from models.exported_qwen3_asr.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.host_array import as_float16_array, as_float16_attention_mask
from torch2vk.runtime.logical import ComparePolicy, LogicalTensor
from torch2vk.runtime.rope_table import run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader
from torch2vk.runtime.streaming_compare import compare_vulkan_stage

get_shader = make_shader_loader("models.exported_qwen3_asr.shaders")


@dataclass(slots=True)
class _QwenCompareReferences:
    audio_encoder: AudioEncoderReference
    audio_inject: AudioInjectReference
    embed_tokens: nn.Module
    norm: nn.Module
    lm_head: nn.Module
    text_layers: tuple[TextLayerReference, ...]
    decode_layers: tuple[TextLayerReference, ...]
    token_select: TokenSelectReference
    token_store: TokenStoreReference


def _reference_lm_head_logits(
    lm_head: nn.Module, hidden_states: reference.ReferenceInput
) -> np.ndarray:
    hidden = torch.from_numpy(np.ascontiguousarray(hidden_states)).cuda().float()
    with torch.no_grad():
        logits = lm_head(hidden)
    return np.ascontiguousarray(logits.detach().cpu().float().numpy())


def _slice_prefill_lm_head_input(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    SLICE_LAST_TOKEN_F16(
        rt,
        x=tensors.text_norm.rms_norm,
        output=tensors.prefill_lm_head_input,
    )


def _run_lm_head_select(rt: RuntimeSession, *, x: LogicalTensor) -> None:
    tensors = model_tensors()
    LM_HEAD_BF16_ARGMAX_PARTIAL_F16(
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
    token_index: int,
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
    start_position: int,
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
        start_position=start_position,
        theta=rope_theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


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
    text_state = TextReferenceState(thinker)
    decode_state = text_state
    text_model = cast(nn.Module, thinker.get_submodule("model"))
    embed_tokens = cast(nn.Module, text_model.get_submodule("embed_tokens"))
    norm = cast(nn.Module, text_model.get_submodule("norm"))
    audio_tower = cast(nn.Module, thinker.get_submodule("audio_tower"))
    lm_head = cast(nn.Module, thinker.get_submodule("lm_head"))
    return _QwenCompareReferences(
        audio_encoder=AudioEncoderReference(audio_tower),
        audio_inject=AudioInjectReference(),
        embed_tokens=embed_tokens,
        norm=norm,
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


def _expected_array(expected: reference.ReferenceExpected, key: str) -> np.ndarray:
    return np.ascontiguousarray(as_numpy_array(expected[key]))


def _module_expected(
    module: nn.Module,
    output_name: str,
    *inputs: reference.ReferenceInput,
) -> reference.ReferenceExpected:
    args = []
    for value in inputs:
        tensor = torch.from_numpy(np.ascontiguousarray(as_numpy_array(value))).cuda()
        if tensor.is_floating_point():
            tensor = tensor.float()
        args.append(tensor)
    with torch.no_grad():
        output = module(*args)
    return {output_name: output}


def _token_select_expected(
    refs: _QwenCompareReferences,
    *,
    logits: reference.ReferenceInput,
    eos_token_ids: np.ndarray,
) -> reference.ReferenceExpected:
    return refs.token_select.execute(
        {
            "logits": np.ascontiguousarray(as_numpy_array(logits)),
            "eos_token_ids": eos_token_ids,
        }
    )


def _logical_tensor_path(field_path: str) -> LogicalTensor:
    value: object = model_tensors()
    for segment in field_path.split("."):
        value = getattr(value, segment)
    if not isinstance(value, LogicalTensor):
        raise TypeError(f"model_tensors().{field_path} is not a LogicalTensor")
    return value


def _compare_lm_head_select(
    rt: RuntimeSession,
    *,
    frame_name: str,
    x: reference.ReferenceInput,
    x_field: str,
    eos_token_ids: np.ndarray,
    refs: _QwenCompareReferences,
) -> reference.ReferenceExpected:
    expected = _token_select_expected(
        refs,
        logits=_reference_lm_head_logits(refs.lm_head, x),
        eos_token_ids=eos_token_ids,
    )
    compare_vulkan_stage(
        rt,
        name=frame_name,
        run=lambda: _run_lm_head_select(rt, x=_logical_tensor_path(x_field)),
        tensors=model_tensors(),
        input_bindings={"x": x_field},
        output_bindings={"next_token": "next_token", "done": "done"},
        inputs={"x": x},
        expected=expected,
        policy=ComparePolicy(kind="token"),
    )
    return expected


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
) -> reference.ReferenceExpected:
    expected = refs.token_store.execute(
        {
            "next_token": np.ascontiguousarray(as_numpy_array(next_token)),
            "token_index": np.asarray([token_index], dtype=np.int64),
            "done": np.ascontiguousarray(as_numpy_array(done)),
            "generated_tokens": np.ascontiguousarray(as_numpy_array(generated_tokens)),
            "generated_length": np.ascontiguousarray(as_numpy_array(generated_length)),
            "stopped": np.ascontiguousarray(as_numpy_array(stopped)),
        }
    )
    compare_vulkan_stage(
        rt,
        name=frame_name,
        run=lambda: QWEN3_ASR_TOKEN_STORE_EOS(
            rt,
            next_token=model_tensors().next_token,
            token_index=int(np.asarray(token_index).reshape(-1)[0]),
            done=model_tensors().done,
            generated_tokens=model_tensors().generated_tokens,
            generated_length=model_tensors().generated_length,
            stopped=model_tensors().stopped,
        ),
        tensors=model_tensors(),
        input_bindings={
            "next_token": "next_token",
            "done": "done",
            "generated_tokens": "generated_tokens",
            "generated_length": "generated_length",
            "stopped": "stopped",
        },
        output_bindings={
            "generated_tokens": "generated_tokens",
            "generated_length": "generated_length",
            "stopped": "stopped",
        },
        inputs={
            "next_token": next_token,
            "done": done,
            "generated_tokens": generated_tokens,
            "generated_length": generated_length,
            "stopped": stopped,
        },
        expected=expected,
        policy=ComparePolicy(kind="token"),
    )
    return expected


def _rope_arrays(
    rt: RuntimeSession,
    *,
    phase: str,
    start_position: int,
    rope_theta: float,
    frame_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    with rt.request():
        _run_rope_table(
            rt,
            phase=phase,
            start_position=start_position,
            rope_theta=rope_theta,
            frame_name=frame_name,
        )
        rope_t = model_tensors().prefill_rope if phase == "prefill" else model_tensors().decode_rope
        cos = np.ascontiguousarray(rt.read_request_state(rope_t.cos))
        sin = np.ascontiguousarray(rt.read_request_state(rope_t.sin))
    return cos, sin


def _compare_prefill_lm_head_input(
    rt: RuntimeSession,
    *,
    hidden_states: np.ndarray,
) -> np.ndarray:
    value = np.ascontiguousarray(hidden_states[:, -1:, :])
    expected: reference.ReferenceExpected = {"prefill_lm_head_input": value}
    compare_vulkan_stage(
        rt,
        name="spike.text.prefill_lm_head_input",
        run=lambda: _slice_prefill_lm_head_input(rt),
        tensors=model_tensors(),
        input_bindings={"x": "text_norm.rms_norm"},
        output_bindings={"prefill_lm_head_input": "prefill_lm_head_input"},
        inputs={"x": hidden_states},
        expected=expected,
        policy=ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    )
    return value


def _run_decode_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    cache_position: int,
    eos_token_ids: np.ndarray,
    token_index: int,
    refs: _QwenCompareReferences,
    next_token: np.ndarray,
    generated_tokens: np.ndarray,
    generated_length: np.ndarray,
    stopped: np.ndarray,
    key_caches: list[np.ndarray],
    value_caches: list[np.ndarray],
    rope_theta: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_expected = _module_expected(refs.embed_tokens, "embedding", next_token)
    hidden_expected = reference.compare_decode_embed(
        rt,
        step=step,
        expected=hidden_expected,
        input=next_token,
    )
    ref_hidden = _expected_array(hidden_expected, "embedding")
    cos, sin = _rope_arrays(
        rt,
        phase="decode",
        start_position=cache_position,
        rope_theta=rope_theta,
        frame_name=f"spike.decode.rope.{step:04d}",
    )
    for layer_idx in range(len(model_tensors().decode_layers)):
        layer_expected = refs.decode_layers[layer_idx].execute(
            {
                "hidden_states": ref_hidden,
                "position_embeddings_0": cos,
                "position_embeddings_1": sin,
                "cache_position": np.asarray([cache_position], dtype=np.int64),
                "key_cache": key_caches[layer_idx],
                "value_cache": value_caches[layer_idx],
            }
        )
        layer_expected = reference.compare_decode_layer(
            rt,
            step=step,
            layer_idx=layer_idx,
            expected=layer_expected,
            hidden_states=ref_hidden,
            position_embeddings_0=cos,
            position_embeddings_1=sin,
            cache_position=cache_position,
            key_cache=key_caches[layer_idx],
            value_cache=value_caches[layer_idx],
        )
        ref_hidden = _expected_array(layer_expected, "add_3")
        key_caches[layer_idx] = _expected_array(layer_expected, "index_copy")
        value_caches[layer_idx] = _expected_array(layer_expected, "index_copy_1")

    norm_expected = _module_expected(refs.norm, "rms_norm", ref_hidden)
    norm_expected = reference.compare_decode_norm(
        rt,
        step=step,
        expected=norm_expected,
        hidden_states=ref_hidden,
    )
    ref_hidden = _expected_array(norm_expected, "rms_norm")
    select_expected = _compare_lm_head_select(
        rt,
        frame_name=f"spike.decode.{step:04d}.token_select",
        x=ref_hidden,
        x_field="decode_norm.rms_norm",
        eos_token_ids=eos_token_ids,
        refs=refs,
    )
    next_token = _expected_array(select_expected, "next_token").astype(np.int64, copy=False)
    done = _expected_array(select_expected, "done").astype(np.uint32, copy=False)
    store_expected = _compare_token_store(
        rt,
        frame_name=f"spike.decode.{step:04d}.token_store",
        refs=refs,
        next_token=next_token,
        token_index=token_index,
        done=done,
        generated_tokens=generated_tokens,
        generated_length=generated_length,
        stopped=stopped,
    )
    return (
        int(next_token.reshape(-1)[0]),
        next_token,
        _expected_array(store_expected, "generated_tokens").astype(np.int64, copy=False),
        _expected_array(store_expected, "generated_length").astype(np.uint32, copy=False),
        _expected_array(store_expected, "stopped").astype(np.uint32, copy=False),
    )


# ==============================================================
# Main pipeline
# ==============================================================


def compare_decode_steps(
    *,
    max_new_tokens: int = 2,
) -> str:
    if max_new_tokens <= 0 or max_new_tokens > 64:
        raise ValueError(f"max_new_tokens must be in [1, 64], got {max_new_tokens}")
    wav_path = Path("tests/fixtures/qwen3_asr_asknot.wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"Test wav not found at {wav_path}")

    print("Preparing inputs...")
    model_dir = resolve_cached_model(REPO_ID)
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
        model_dir=model_dir,
        model_tensors=model_tensors(),
        session_tensors={model_tensors().eos_token_ids: eos_token_array},
        get_shader=get_shader,
    )
    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float16,
    )
    key_caches = [zero_cache.copy() for _ in range(tc.num_hidden_layers)]
    value_caches = [zero_cache.copy() for _ in range(tc.num_hidden_layers)]
    generated_tokens_state = np.zeros((1, max_new_tokens), dtype=np.int64)
    generated_length_state = np.zeros((1,), dtype=np.uint32)
    stopped_state = np.zeros((1,), dtype=np.uint32)
    preprocessed = preprocess_audio_inputs(
        prepared.input_ids,
        prepared.input_features,
        prepared.feature_attention_mask,
        position_embedding_shape=tuple(
            int(dim) for dim in model_tensors().audio_encoder.position_embedding.spec.shape
        ),
        d_model=ac.d_model,
    )
    prefill_cache_position = np.arange(prompt_length, dtype=np.int64)
    try:
        print("Loading PyTorch reference for compare...")
        compare_refs = _build_compare_references(_load_qwen_reference_model(Path(model_dir)))

        # === Audio Tower ===
        print("\n=== Phase 1: Audio Tower ===")
        print(f"  hidden_states after compact: {model_tensors().audio_encoder.index_select.spec.shape}")
        print(f"  audio encoder ({model_tensors().audio_encoder.x.spec.shape})...")
        audio_x = as_float16_array(preprocessed["padded_feature"])
        audio_position_embedding = as_float16_array(preprocessed["position_embedding"])
        audio_attention_mask = as_float16_attention_mask(preprocessed["audio_attention_mask"])
        audio_expected = compare_refs.audio_encoder.execute(
            {
                "x": audio_x,
                "position_embedding": audio_position_embedding,
                "compact_index": preprocessed["compact_index"],
                "attention_mask": audio_attention_mask,
            }
        )
        audio_expected = reference.compare_audio_encoder(
            rt,
            expected=audio_expected,
            x=audio_x,
            position_embedding=audio_position_embedding,
            compact_index=preprocessed["compact_index"],
            attention_mask=audio_attention_mask,
        )
        ref_audio_features = _expected_array(audio_expected, "linear_110")
        print(f"  Audio tower output: {model_tensors().audio_encoder.linear_110.spec.shape}")

        # === Text Prefill ===
        print("\n=== Phase 2: Text Prefill ===")
        prompt_length = prepared.prompt_length

        audio_positions = preprocessed["audio_positions"]
        if len(audio_positions) > 0:
            audio_start = int(audio_positions[0])
            audio_end = audio_start + len(audio_positions)
            print(f"    Injecting audio [{audio_start}:{audio_end}]")

        ref_cos, ref_sin = _rope_arrays(
            rt,
            phase="prefill",
            start_position=0,
            rope_theta=rope_theta,
            frame_name="spike.text.prefill_rope",
        )

        # Embedding, audio injection, and decoder layers stay on GPU.
        print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
        embed_expected = _module_expected(compare_refs.embed_tokens, "embedding", prepared.input_ids)
        embed_expected = reference.compare_embed_tokens(
            rt,
            expected=embed_expected,
            input=prepared.input_ids,
        )
        ref_hidden = _expected_array(embed_expected, "embedding")
        inject_expected = compare_refs.audio_inject.execute(
            {
                "inputs_embeds": ref_hidden,
                "audio_positions": preprocessed["audio_positions"],
                "audio_features": ref_audio_features,
            }
        )
        inject_expected = reference.compare_audio_inject(
            rt,
            expected=inject_expected,
            inputs_embeds=ref_hidden,
            audio_positions=preprocessed["audio_positions"],
            audio_features=ref_audio_features,
        )
        ref_hidden = _expected_array(inject_expected, "embedding")
        for layer_idx in range(len(model_tensors().text_layers)):
            layer_expected = compare_refs.text_layers[layer_idx].execute(
                {
                    "hidden_states": ref_hidden,
                    "position_embeddings_0": ref_cos,
                    "position_embeddings_1": ref_sin,
                    "cache_position": prefill_cache_position,
                    "key_cache": key_caches[layer_idx],
                    "value_cache": value_caches[layer_idx],
                }
            )
            layer_expected = reference.compare_text_layer(
                rt,
                layer_idx=layer_idx,
                expected=layer_expected,
                hidden_states=ref_hidden,
                position_embeddings_0=ref_cos,
                position_embeddings_1=ref_sin,
                cache_position=prefill_cache_position,
                key_cache=key_caches[layer_idx],
                value_cache=value_caches[layer_idx],
            )
            ref_hidden = _expected_array(layer_expected, "add_3")
            key_caches[layer_idx] = _expected_array(layer_expected, "index_copy")
            value_caches[layer_idx] = _expected_array(layer_expected, "index_copy_1")
            if layer_idx % 7 == 6:
                print(f"    layer {layer_idx} done")
        norm_expected = _module_expected(compare_refs.norm, "rms_norm", ref_hidden)
        norm_expected = reference.compare_text_norm(
            rt,
            expected=norm_expected,
            hidden_states=ref_hidden,
        )
        ref_hidden = _expected_array(norm_expected, "rms_norm")
        lm_head_input = _compare_prefill_lm_head_input(rt, hidden_states=ref_hidden)
        select_expected = _compare_lm_head_select(
            rt,
            frame_name="spike.text.token_select",
            x=lm_head_input,
            x_field="prefill_lm_head_input",
            eos_token_ids=eos_token_array,
            refs=compare_refs,
        )

        print("  lm_head + token_select...")
        next_token = _expected_array(select_expected, "next_token").astype(np.int64, copy=False)
        done = _expected_array(select_expected, "done").astype(np.uint32, copy=False)
        store_expected = _compare_token_store(
            rt,
            frame_name="spike.text.token_store",
            refs=compare_refs,
            next_token=next_token,
            token_index=0,
            done=done,
            generated_tokens=generated_tokens_state,
            generated_length=generated_length_state,
            stopped=stopped_state,
        )
        generated_tokens_state = _expected_array(store_expected, "generated_tokens").astype(
            np.int64,
            copy=False,
        )
        generated_length_state = _expected_array(store_expected, "generated_length").astype(
            np.uint32,
            copy=False,
        )
        stopped_state = _expected_array(store_expected, "stopped").astype(np.uint32, copy=False)
        first_token = int(next_token.reshape(-1)[0])
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
            next_token_value, next_token, generated_tokens_state, generated_length_state, stopped_state = _run_decode_step_with_compare(
                rt,
                step=step,
                cache_position=cache_pos,
                eos_token_ids=eos_token_array,
                token_index=step + 1,
                refs=compare_refs,
                next_token=next_token,
                generated_tokens=generated_tokens_state,
                generated_length=generated_length_state,
                stopped=stopped_state,
                key_caches=key_caches,
                value_caches=value_caches,
                rope_theta=rope_theta,
            )
            generated_tokens.append(next_token_value)
            if step < 5 or step % 20 == 0:
                print(f"  Step {step}: token={generated_tokens[-1]}")

        # Decode text
        print("\n=== Result ===")
        stored_length = int(generated_length_state.reshape(-1)[0])
        generated_tokens = [
            int(token)
            for token in generated_tokens_state.reshape(-1)[:stored_length]
        ]
        print(f"Generated {len(generated_tokens)} tokens")
        text = processor.batch_decode(
            np.array([generated_tokens], dtype=np.int64),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"Transcription: {text}")

    finally:
        rt.close()
    return text


if __name__ == "__main__":
    result = compare_decode_steps()
    print(result)
