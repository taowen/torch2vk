"""PyTorch/Vulkan comparison entry points for exported OmniVoice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

from models.exported_omnivoice import reference
from models.exported_omnivoice.dispatch.audio_decode import run_audio_decode
from models.exported_omnivoice.dispatch.audio_head import run_audio_head
from models.exported_omnivoice.dispatch.llm_forward import run_llm_forward
from models.exported_omnivoice.input_prep import DEFAULT_TEXT, prepare_omnivoice_inputs
from models.exported_omnivoice.pytorch_modules import (
    InputEmbedReference,
    AudioDecodeReference,
    LlmForwardReference,
    TokenScoreReference,
    TokenUpdateReference,
)
from models.exported_omnivoice.run import (
    _get_time_steps,
    _run_input_embed,
    _run_rope_table,
    _run_token_score,
    _run_token_update,
)
from models.exported_omnivoice.tensors.model import create_model_tensors, model_tensors
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.runtime.host_array import as_float16_attention_mask
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader

get_shader = make_shader_loader("models.exported_omnivoice.shaders")


@dataclass(slots=True)
class _OmniVoiceCompareState:
    input_embed: InputEmbedReference
    llm_forward: LlmForwardReference
    token_score: TokenScoreReference
    token_update: TokenUpdateReference
    audio_decode: AudioDecodeReference


def _build_compare_references(
    model: OmniVoice,
    *,
    audio_tokenizer: HiggsAudioV2TokenizerModel,
) -> _OmniVoiceCompareState:
    reference.set_model(model)
    return _OmniVoiceCompareState(
        input_embed=InputEmbedReference(model),
        llm_forward=LlmForwardReference(model),
        token_score=TokenScoreReference(model),
        token_update=TokenUpdateReference(),
        audio_decode=AudioDecodeReference(audio_tokenizer),
    )


def _vulkan_tensor(rt: RuntimeSession, tensor: LogicalTensor) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rt.readback(tensor))).cuda()


def _run_generation_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    unmask_count: int,
    rng_seed: int,
    audio_mask_id: int,
    refs: _OmniVoiceCompareState,
) -> None:
    with rt.frame(f"omnivoice.step.{step:04d}"):
        batch_input_ids = _vulkan_tensor(rt, model_tensors().batch_input_ids).long()
        batch_audio_mask = _vulkan_tensor(rt, model_tensors().batch_audio_mask).to(torch.bool)
        _run_input_embed(rt)
        reference.run_input_embed(
            rt,
            refs.input_embed,
            step=step,
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
        )
        hidden_states = _vulkan_tensor(rt, model_tensors().llm_forward.hidden_states).float()

        run_llm_forward(rt)
        reference.run_llm_forward(
            rt,
            refs.llm_forward,
            step=step,
            hidden_states=hidden_states,
            cos=_vulkan_tensor(rt, model_tensors().rope.cos).float(),
            sin=_vulkan_tensor(rt, model_tensors().rope.sin).float(),
            attention_mask=_vulkan_tensor(rt, model_tensors().attention_mask).float(),
        )
        llm_output = _vulkan_tensor(rt, model_tensors().llm_forward.mul_365).float()

        run_audio_head(rt)
        reference.run_audio_head(
            rt,
            step=step,
            input=llm_output,
        )
        logits = _vulkan_tensor(rt, model_tensors().audio_head.linear).float()

        step_index = torch.tensor([step], dtype=torch.int64, device="cuda")
        tokens = _vulkan_tensor(rt, model_tensors().tokens).long()
        _run_token_score(rt, step=step, rng_seed=rng_seed, audio_mask_id=audio_mask_id)
        reference.run_token_score(
            rt,
            refs.token_score,
            step=step,
            logits=logits,
            tokens=tokens,
            audio_mask_id=audio_mask_id,
            rng_seed=np.array([rng_seed], dtype=np.uint32),
            step_index=step_index,
        )
        candidate_tokens = _vulkan_tensor(rt, model_tensors().candidate_tokens).long()
        candidate_scores = _vulkan_tensor(rt, model_tensors().candidate_scores).float()

        _run_token_update(rt, unmask_count=unmask_count)
        unmask_count_t = torch.tensor([unmask_count], dtype=torch.uint32, device="cuda")
        reference.run_token_update(
            rt,
            refs.token_update,
            step=step,
            tokens=tokens,
            batch_input_ids=batch_input_ids,
            candidate_tokens=candidate_tokens,
            candidate_scores=candidate_scores,
            unmask_count=unmask_count_t,
        )


def compare_generation_steps(
    *,
    text: str = DEFAULT_TEXT,
    num_steps: int = 2,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
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
            f"exported OmniVoice seq_len is {expected_seq_len}, "
            f"but prepared inputs require {prepared.seq_len}; regenerate exported_omnivoice"
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
        model_dir=model_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    try:
        rt.initialize_request_state(
            {
                model_tensors().batch_input_ids: prepared.batch_input_ids,
                model_tensors().batch_audio_mask: prepared.batch_audio_mask,
                model_tensors().attention_mask: attention_mask,
                model_tensors().tokens: tokens,
            }
        )
        _run_rope_table(rt, frame_name="omnivoice.rope")

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
            _run_generation_step_with_compare(
                rt,
                step=step,
                unmask_count=int(unmask_count),
                rng_seed=rng_seed,
                audio_mask_id=config.audio_mask_id,
                refs=refs,
            )
        with rt.frame("omnivoice.audio_decode"):
            run_audio_decode(rt)
            reference.run_audio_decode(
                rt,
                refs.audio_decode,
                audio_codes=_vulkan_tensor(rt, model_tensors().tokens).long(),
            )
    finally:
        rt.close()
