"""PyTorch/Vulkan comparison entry points for quantized OmniVoice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

from models.exported_omnivoice.pytorch_modules import (
    AudioDecodeReference,
    AudioHeadReference,
    InputEmbedReference,
    LlmForwardReference,
    TokenScoreReference,
    TokenUpdateReference,
)
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from models.quantized_omnivoice import reference
from models.quantized_omnivoice.export_gguf import export_omnivoice_q4_k_m_gguf
from models.quantized_omnivoice.input_prep import DEFAULT_TEXT, prepare_omnivoice_inputs
from models.quantized_omnivoice.run import (
    _get_time_steps,
    _run_rope_table,
    _run_token_score,
    _run_token_update,
)
from models.quantized_omnivoice.tensors.model import create_model_tensors, model_tensors
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.host_array import as_float16_attention_mask
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader
from torch2vk.runtime.streaming_compare import compare_vulkan_stage

get_shader = make_shader_loader("models.quantized_omnivoice.shaders")


@dataclass(slots=True)
class _OmniVoiceCompareState:
    input_embed: InputEmbedReference
    llm_forward: LlmForwardReference
    audio_head: AudioHeadReference
    token_score: TokenScoreReference
    token_update: TokenUpdateReference
    audio_decode: AudioDecodeReference


def _build_compare_references(
    model: OmniVoice,
    *,
    audio_tokenizer: HiggsAudioV2TokenizerModel,
) -> _OmniVoiceCompareState:
    return _OmniVoiceCompareState(
        input_embed=InputEmbedReference(model),
        llm_forward=LlmForwardReference(model),
        audio_head=AudioHeadReference(model),
        token_score=TokenScoreReference(model),
        token_update=TokenUpdateReference(),
        audio_decode=AudioDecodeReference(audio_tokenizer),
    )


def _expected_array(expected: reference.ReferenceExpected, key: str) -> np.ndarray:
    return np.ascontiguousarray(as_numpy_array(expected[key]))


def _run_generation_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    unmask_count: int,
    rng_seed: int,
    audio_mask_id: int,
    batch_input_ids: np.ndarray,
    batch_audio_mask: np.ndarray,
    attention_mask: np.ndarray,
    tokens: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    refs: _OmniVoiceCompareState,
) -> tuple[np.ndarray, np.ndarray]:
    embed_expected = refs.input_embed.execute(
        {
            "input_ids": batch_input_ids,
            "audio_mask": batch_audio_mask,
        }
    )
    embed_expected = reference.compare_input_embed(
        rt,
        step=step,
        expected=embed_expected,
        input_ids=batch_input_ids,
        audio_mask=batch_audio_mask,
    )
    hidden_states = _expected_array(embed_expected, "hidden_states")

    llm_expected = refs.llm_forward.execute(
        {
            "hidden_states": hidden_states,
            "cos": cos,
            "sin": sin,
            "attention_mask": attention_mask,
        }
    )
    llm_expected = reference.compare_llm_forward(
        rt,
        step=step,
        expected=llm_expected,
        hidden_states=hidden_states,
        cos=cos,
        sin=sin,
        attention_mask=attention_mask,
    )
    llm_output = _expected_array(llm_expected, "rms_norm_112")

    head_expected = refs.audio_head.execute({"input": llm_output})
    head_expected = reference.compare_audio_head(
        rt,
        step=step,
        expected=head_expected,
        input=llm_output,
    )
    logits = _expected_array(head_expected, "linear")

    score_expected = refs.token_score.execute(
        {
            "logits": logits,
            "tokens": tokens,
            "audio_mask_id": np.array([audio_mask_id], dtype=np.int64),
            "rng_seed": np.array([rng_seed], dtype=np.uint32),
            "step_index": np.array([step], dtype=np.int64),
        }
    )
    compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.token_score",
        run=lambda: _run_token_score(
            rt,
            step=step,
            rng_seed=rng_seed,
            audio_mask_id=audio_mask_id,
        ),
        tensors=model_tensors(),
        input_bindings={
            "logits": "audio_head.linear",
            "tokens": "tokens",
        },
        output_bindings={"candidate_scores": "candidate_scores"},
        inputs={
            "logits": logits,
            "tokens": tokens,
        },
        expected=score_expected,
        policy={"candidate_scores": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5)},
    )
    candidate_tokens = _expected_array(score_expected, "candidate_tokens")
    candidate_scores = _expected_array(score_expected, "candidate_scores")

    update_expected = refs.token_update.execute(
        {
            "tokens": tokens,
            "batch_input_ids": batch_input_ids,
            "candidate_tokens": candidate_tokens,
            "candidate_scores": candidate_scores,
            "unmask_count": np.array([unmask_count], dtype=np.uint32),
        }
    )
    compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.token_update",
        run=lambda: _run_token_update(rt, unmask_count=unmask_count),
        tensors=model_tensors(),
        input_bindings={
            "tokens": "tokens",
            "batch_input_ids": "batch_input_ids",
            "candidate_tokens": "candidate_tokens",
            "candidate_scores": "candidate_scores",
        },
        output_bindings={
            "tokens": "tokens",
            "batch_input_ids": "batch_input_ids",
        },
        inputs={
            "tokens": tokens,
            "batch_input_ids": batch_input_ids,
            "candidate_tokens": candidate_tokens,
            "candidate_scores": candidate_scores,
        },
        expected=update_expected,
        policy=ComparePolicy(kind="token"),
    )
    return (
        _expected_array(update_expected, "tokens").astype(np.int64, copy=False),
        _expected_array(update_expected, "batch_input_ids").astype(np.int64, copy=False),
    )


def compare_generation_steps(
    *,
    text: str = DEFAULT_TEXT,
    num_steps: int = 2,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    gguf_path = export_omnivoice_q4_k_m_gguf(model_dir=model_dir)
    config = OmniVoiceConfig(**json.loads((model_dir / "config.json").read_text()))
    llm_config = config.llm_config
    if llm_config is None:
        raise ValueError("OmniVoice config requires llm_config")

    text_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prepared = prepare_omnivoice_inputs(
        text=text,
        tokenizer=text_tokenizer,
        config=config,
    )
    create_model_tensors(target_len=prepared.target_len)
    expected_seq_len = model_tensors().batch_input_ids.spec.shape[2]
    if expected_seq_len != prepared.seq_len:
        raise ValueError(
            f"quantized OmniVoice seq_len is {expected_seq_len}, "
            f"but prepared inputs require {prepared.seq_len}; regenerate quantized_omnivoice"
        )

    model = cast(
        OmniVoice,
        OmniVoice.from_pretrained(
            str(model_dir),
            dtype=torch.float32,
            device_map="cuda",
            train=True,
        ).eval(),
    )
    audio_tokenizer = cast(
        HiggsAudioV2TokenizerModel,
        HiggsAudioV2TokenizerModel.from_pretrained(
            str(model_dir / "audio_tokenizer"),
            device_map="cuda",
        ).eval(),
    )
    tokens = np.full(
        (1, config.num_audio_codebook, prepared.target_len),
        config.audio_mask_id,
        dtype=np.int64,
    )
    attention_mask = as_float16_attention_mask(prepared.attention_mask)
    rng_seed = 0x1234ABCD
    refs = _build_compare_references(
        model,
        audio_tokenizer=audio_tokenizer,
    )

    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_path.parent,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    try:
        with rt.request():
            _run_rope_table(rt, frame_name="omnivoice.rope")
            cos = np.ascontiguousarray(rt.read_request_state(model_tensors().rope.cos))
            sin = np.ascontiguousarray(rt.read_request_state(model_tensors().rope.sin))

        batch_input_ids = np.ascontiguousarray(prepared.batch_input_ids, dtype=np.int64)
        batch_audio_mask = np.ascontiguousarray(prepared.batch_audio_mask)
        tokens = np.ascontiguousarray(tokens)
        timesteps = _get_time_steps(0.0, 1.0, num_steps, t_shift=0.1)
        total_mask = prepared.target_len * config.num_audio_codebook
        remaining = total_mask
        for step in range(num_steps):
            if step == num_steps - 1:
                unmask_count = remaining
            else:
                unmask_count = min(
                    math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                    remaining,
                )
            remaining -= int(unmask_count)
            if unmask_count <= 0:
                continue
            tokens, batch_input_ids = _run_generation_step_with_compare(
                rt,
                step=step,
                unmask_count=int(unmask_count),
                rng_seed=rng_seed,
                audio_mask_id=config.audio_mask_id,
                batch_input_ids=batch_input_ids,
                batch_audio_mask=batch_audio_mask,
                attention_mask=attention_mask,
                tokens=tokens,
                cos=cos,
                sin=sin,
                refs=refs,
            )

        audio_expected = refs.audio_decode.execute({"audio_codes": tokens})
        audio_expected = reference.compare_audio_decode(
            rt,
            expected=audio_expected,
            audio_codes=tokens,
        )
    finally:
        rt.close()
