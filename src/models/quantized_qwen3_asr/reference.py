"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

import numpy as np
import torch

from models.quantized_qwen3_asr.dispatch.audio_encoder import run_audio_encoder as _dispatch_audio_encoder
from models.quantized_qwen3_asr.dispatch.embed_tokens import run_embed_tokens as _dispatch_embed_tokens
from models.quantized_qwen3_asr.dispatch.audio_inject import run_audio_inject as _dispatch_audio_inject
from models.quantized_qwen3_asr.dispatch.text_layer import run_text_layer as _dispatch_text_layer
from models.quantized_qwen3_asr.dispatch.text_norm import run_text_norm as _dispatch_text_norm
from models.quantized_qwen3_asr.dispatch.decode_embed import run_decode_embed as _dispatch_decode_embed
from models.quantized_qwen3_asr.dispatch.decode_layer import run_decode_layer as _dispatch_decode_layer
from models.quantized_qwen3_asr.dispatch.decode_norm import run_decode_norm as _dispatch_decode_norm
from models.quantized_qwen3_asr.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.streaming_compare import compare_vulkan_stage


ReferenceInput = np.ndarray | torch.Tensor | int
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "q8_tensor": ComparePolicy(kind="tensor", rtol=2e-2, atol=8.0),
    "q4_tensor": ComparePolicy(kind="tensor", rtol=5e-2, atol=128.0),
    "token": ComparePolicy(kind="token"),
}


def _reference_int(value: ReferenceInput) -> int:
    if isinstance(value, int):
        return value
    return int(as_numpy_array(value).reshape(-1)[0])


def _policy(policy: str | dict[str, str]) -> ComparePolicy | dict[str, ComparePolicy]:
    if isinstance(policy, dict):
        return {key: _COMPARE_POLICIES[value] for key, value in policy.items()}
    return _COMPARE_POLICIES[policy]

def compare_audio_encoder(
    rt: RuntimeSession,
    *,
    expected: ReferenceExpected,
    x: ReferenceInput,
    position_embedding: ReferenceInput,
    compact_index: ReferenceInput,
    attention_mask: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name='spike.audio.encoder',
        run=lambda: _dispatch_audio_encoder(rt),
        tensors=model_tensors().audio_encoder,
        input_bindings={'x': 'x', 'position_embedding': 'position_embedding', 'compact_index': 'compact_index', 'attention_mask': 'attention_mask'},
        output_bindings={'linear_110': 'linear_110'},
        policy=_policy('q4_tensor'),
        inputs={
            "x": x,
            "position_embedding": position_embedding,
            "compact_index": compact_index,
            "attention_mask": attention_mask,
        },
        expected=expected,
    )

def compare_embed_tokens(
    rt: RuntimeSession,
    *,
    expected: ReferenceExpected,
    input: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name='spike.text.embed',
        run=lambda: _dispatch_embed_tokens(rt),
        tensors=model_tensors().embed_tokens,
        input_bindings={'input': 'input'},
        output_bindings={'embedding': 'embedding'},
        policy=_policy('q8_tensor'),
        inputs={
            "input": input,
        },
        expected=expected,
    )

def compare_audio_inject(
    rt: RuntimeSession,
    *,
    expected: ReferenceExpected,
    inputs_embeds: ReferenceInput,
    audio_positions: ReferenceInput,
    audio_features: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name='spike.text.audio_inject',
        run=lambda: _dispatch_audio_inject(rt),
        tensors=model_tensors().audio_inject,
        input_bindings={'inputs_embeds': 'index_copy', 'audio_positions': 'audio_positions', 'audio_features': 'audio_features'},
        output_bindings={'embedding': 'index_copy'},
        policy=_policy('tensor'),
        inputs={
            "inputs_embeds": inputs_embeds,
            "audio_positions": audio_positions,
            "audio_features": audio_features,
        },
        expected=expected,
    )

def compare_text_layer(
    rt: RuntimeSession,
    *,
    layer_idx: int,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
    position_embeddings_0: ReferenceInput,
    position_embeddings_1: ReferenceInput,
    cache_position: ReferenceInput,
    key_cache: ReferenceInput,
    value_cache: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'spike.text.layer.{layer_idx}',
        run=lambda: _dispatch_text_layer(rt, layer_idx),
        tensors=model_tensors().text_layers[layer_idx],
        input_bindings={'hidden_states': 'hidden_states', 'position_embeddings_0': 'position_embeddings_0', 'position_embeddings_1': 'position_embeddings_1', 'cache_position': 'cache_position', 'key_cache': 'index_copy', 'value_cache': 'index_copy_1'},
        output_bindings={'add_3': 'add_3', 'index_copy': 'index_copy', 'index_copy_1': 'index_copy_1'},
        policy=_policy('q4_tensor'),
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "cache_position": cache_position,
            "key_cache": key_cache,
            "value_cache": value_cache,
        },
        expected=expected,
    )

def compare_text_norm(
    rt: RuntimeSession,
    *,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name='spike.text.norm',
        run=lambda: _dispatch_text_norm(rt),
        tensors=model_tensors().text_norm,
        input_bindings={'hidden_states': 'hidden_states'},
        output_bindings={'rms_norm': 'rms_norm'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
        },
        expected=expected,
    )

def compare_decode_embed(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    input: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'spike.decode.{step:04d}.embed',
        run=lambda: _dispatch_decode_embed(rt),
        tensors=model_tensors().decode_embed,
        input_bindings={'input': 'input'},
        output_bindings={'embedding': 'embedding'},
        policy=_policy('q8_tensor'),
        inputs={
            "input": input,
        },
        expected=expected,
    )

def compare_decode_layer(
    rt: RuntimeSession,
    *,
    step: int,
    layer_idx: int,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
    position_embeddings_0: ReferenceInput,
    position_embeddings_1: ReferenceInput,
    key_cache: ReferenceInput,
    value_cache: ReferenceInput,
    cache_position: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'spike.decode.{step:04d}.layer.{layer_idx}',
        run=lambda: _dispatch_decode_layer(rt, layer_idx, cache_position=_reference_int(cache_position)),
        tensors=model_tensors().decode_layers[layer_idx],
        input_bindings={'hidden_states': 'hidden_states', 'position_embeddings_0': 'position_embeddings_0', 'position_embeddings_1': 'position_embeddings_1', 'key_cache': 'index_copy', 'value_cache': 'index_copy_1'},
        output_bindings={'add_3': 'add_3', 'index_copy': 'index_copy', 'index_copy_1': 'index_copy_1'},
        policy=_policy('q4_tensor'),
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "key_cache": key_cache,
            "value_cache": value_cache,
        },
        expected=expected,
    )

def compare_decode_norm(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'spike.decode.{step:04d}.norm',
        run=lambda: _dispatch_decode_norm(rt),
        tensors=model_tensors().decode_norm,
        input_bindings={'hidden_states': 'hidden_states'},
        output_bindings={'rms_norm': 'rms_norm'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
        },
        expected=expected,
    )
