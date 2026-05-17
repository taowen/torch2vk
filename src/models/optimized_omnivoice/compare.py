"""PyTorch/Vulkan comparison entry points for optimized OmniVoice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

from models.exported_omnivoice.pytorch_modules import (
    AudioDecodeReference,
    AudioHeadReference,
    InputEmbedReference,
    LlmForwardReference,
)
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.dispatch.audio_decode import run_audio_decode_with_tensors
from models.optimized_omnivoice.dispatch.audio_head import run_audio_head
from models.optimized_omnivoice.dispatch.llm_forward import run_llm_forward
from models.optimized_omnivoice.export_gguf import export_omnivoice_q4_k_m_gguf
from models.optimized_omnivoice.input_prep import (
    DEFAULT_TEXT,
    TARGET_CAPACITY,
    prepare_omnivoice_inputs,
)
from models.optimized_omnivoice.pytorch.example import REPO_ID
from models.optimized_omnivoice.run import (
    _audio_decode_topology,
    _get_time_steps,
    _run_input_embed,
    _run_rope_table,
    _run_token_score,
    _run_token_update,
)
from models.optimized_omnivoice.tensors.model import create_model_tensors, model_tensors
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.host_array import as_float16_attention_mask
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader
from torch2vk.runtime.streaming_compare import compare_vulkan_stage

get_shader = make_shader_loader("models.optimized_omnivoice.shaders")

ReferenceExpected: TypeAlias = dict[str, object]
_Q4_COMPARE_POLICY = ComparePolicy(kind="tensor", rtol=5e-2, atol=128.0)
_AUDIO_DECODE_COMPARE_POLICY = ComparePolicy(kind="waveform", rtol=0.0, atol=0.2, max_abs=0.2)


@dataclass(slots=True)
class _OmniVoiceCompareState:
    input_embed: InputEmbedReference
    llm_forward: LlmForwardReference
    audio_head: AudioHeadReference
    token_score: "_ActiveTokenScoreReference"
    token_update: "_ActiveTokenUpdateReference"
    audio_decode: AudioDecodeReference


def _torch_input(inputs: dict[str, np.ndarray], name: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(inputs[name])).cuda()


class _ActiveTokenScoreReference:
    def __init__(self, model: OmniVoice) -> None:
        self.num_audio_codebook = model.config.num_audio_codebook
        self.audio_vocab_size = model.config.audio_vocab_size
        self.guidance_scale = 2.0
        self.layer_penalty = 5.0
        self.position_temperature = 5.0

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        logits = _torch_input(inputs, "logits").float()
        tokens = _torch_input(inputs, "tokens").long()
        audio_mask_id = _torch_input(inputs, "audio_mask_id").long()
        rng_seed = _torch_input(inputs, "rng_seed").long()
        step_index = _torch_input(inputs, "step_index").long()
        target_len = int(inputs["active_target_len"].reshape(()))
        cond_target_start = int(inputs["cond_target_start"].reshape(()))

        seq_len = logits.shape[1]
        logits = logits.view(2, seq_len, self.num_audio_codebook, self.audio_vocab_size)
        cond_logits = logits[0:1, cond_target_start : cond_target_start + target_len].permute(
            0, 2, 1, 3
        )
        uncond_logits = logits[1:2, :target_len].permute(0, 2, 1, 3)
        cond_log_probs = F.log_softmax(cond_logits, dim=-1)
        uncond_log_probs = F.log_softmax(uncond_logits, dim=-1)
        guided = cond_log_probs + self.guidance_scale * (cond_log_probs - uncond_log_probs)
        log_probs = F.log_softmax(guided, dim=-1)
        log_probs[..., int(audio_mask_id.reshape(()))] = -float("inf")

        candidate_tokens = torch.full(
            (self.num_audio_codebook, TARGET_CAPACITY),
            int(audio_mask_id.reshape(())),
            dtype=torch.int64,
            device="cuda",
        )
        candidate_scores = torch.full(
            (self.num_audio_codebook, TARGET_CAPACITY),
            -torch.finfo(torch.float32).max,
            dtype=torch.float32,
            device="cuda",
        )
        candidate_tokens[:, :target_len] = log_probs.argmax(dim=-1)[0].to(torch.int64)
        active_scores = log_probs.max(dim=-1)[0][0]
        layer_ids = torch.arange(
            self.num_audio_codebook,
            device="cuda",
            dtype=torch.float32,
        ).view(self.num_audio_codebook, 1)
        active_scores = active_scores - layer_ids * self.layer_penalty
        if self.position_temperature > 0.0:
            flat_pos = torch.arange(
                self.num_audio_codebook * target_len,
                device="cuda",
                dtype=torch.int64,
            ).view(self.num_audio_codebook, target_len)
            active_scores = active_scores / self.position_temperature + self._gumbel_noise(
                flat_pos,
                rng_seed,
                step_index,
            )
        active_scores = torch.where(
            tokens[0, :, :target_len] != audio_mask_id.reshape(()),
            torch.full_like(active_scores, -torch.finfo(torch.float32).max),
            active_scores,
        )
        candidate_scores[:, :target_len] = active_scores
        return {"candidate_tokens": candidate_tokens, "candidate_scores": candidate_scores}

    def _gumbel_noise(
        self,
        flat_pos: torch.Tensor,
        rng_seed: torch.Tensor,
        step_index: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.bitwise_xor(
            self._u32(rng_seed.to(torch.int64).reshape(())),
            self._u32(step_index.to(torch.int64).reshape(()) * 0x9E3779B9),
        )
        x = torch.bitwise_xor(x, self._u32(flat_pos.to(torch.int64) * 0x85EBCA6B))
        hashed = self._hash_u32(x)
        u = (torch.bitwise_and(hashed, 0x00FFFFFF).to(torch.float32) + 0.5) * (1.0 / 16777216.0)
        u = torch.clamp(u, 1.0e-7, 1.0 - 1.0e-7)
        return -torch.log(-torch.log(u))

    def _hash_u32(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
        x = self._u32(x * 0x7FEB352D)
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 15))
        x = self._u32(x * 0x846CA68B)
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
        return self._u32(x)

    def _u32(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(x, 0xFFFFFFFF)


class _ActiveTokenUpdateReference:
    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        tokens = _torch_input(inputs, "tokens").long()
        batch_input_ids = _torch_input(inputs, "batch_input_ids").long()
        candidate_tokens = _torch_input(inputs, "candidate_tokens").long()
        candidate_scores = _torch_input(inputs, "candidate_scores").float()
        unmask_count = int(inputs["unmask_count"].reshape(()))
        target_len = int(inputs["active_target_len"].reshape(()))
        cond_target_start = int(inputs["cond_target_start"].reshape(()))

        flat_scores = candidate_scores[:, :target_len].reshape(-1)
        flat_candidates = candidate_tokens[:, :target_len].reshape(-1)
        active_tokens = tokens[:, :, :target_len].clone()
        flat_tokens = active_tokens.reshape(-1)
        if unmask_count > 0:
            _, topk_idx = torch.topk(flat_scores, min(unmask_count, flat_scores.numel()))
            flat_tokens[topk_idx] = flat_candidates[topk_idx]
        updated_active = flat_tokens.view_as(active_tokens)
        updated_tokens = tokens.clone()
        updated_tokens[:, :, :target_len] = updated_active
        updated_batch_input_ids = batch_input_ids.clone()
        updated_batch_input_ids[0:1, :, cond_target_start : cond_target_start + target_len] = (
            updated_active
        )
        updated_batch_input_ids[1:2, :, :target_len] = updated_active
        return {"tokens": updated_tokens, "batch_input_ids": updated_batch_input_ids}


def _build_compare_references(
    model: OmniVoice,
    *,
    audio_tokenizer: HiggsAudioV2TokenizerModel,
) -> _OmniVoiceCompareState:
    return _OmniVoiceCompareState(
        input_embed=InputEmbedReference(model),
        llm_forward=LlmForwardReference(model),
        audio_head=AudioHeadReference(model),
        token_score=_ActiveTokenScoreReference(model),
        token_update=_ActiveTokenUpdateReference(),
        audio_decode=AudioDecodeReference(audio_tokenizer),
    )


def _expected_array(expected: ReferenceExpected, key: str) -> np.ndarray:
    return np.ascontiguousarray(as_numpy_array(expected[key]))


def _run_generation_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    unmask_count: int,
    rng_seed: int,
    audio_mask_id: int,
    active_target_len: int,
    cond_target_start: int,
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
    embed_expected = compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.input_embed",
        run=lambda: _run_input_embed(rt),
        tensors=model_tensors(),
        input_bindings={
            "input_ids": "batch_input_ids",
            "audio_mask": "batch_audio_mask",
        },
        output_bindings={"hidden_states": "llm_forward.hidden_states"},
        inputs={
            "input_ids": batch_input_ids,
            "audio_mask": batch_audio_mask,
        },
        expected=embed_expected,
        policy=_Q4_COMPARE_POLICY,
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
    llm_expected = compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.llm_forward",
        run=lambda: run_llm_forward(rt),
        tensors=model_tensors(),
        input_bindings={
            "hidden_states": "llm_forward.hidden_states",
            "cos": "rope.cos",
            "sin": "rope.sin",
            "attention_mask": "attention_mask",
        },
        output_bindings={"mul_365": "llm_forward.mul_365"},
        inputs={
            "hidden_states": hidden_states,
            "cos": cos,
            "sin": sin,
            "attention_mask": attention_mask,
        },
        expected=llm_expected,
        policy=_Q4_COMPARE_POLICY,
    )
    llm_output = _expected_array(llm_expected, "mul_365")

    head_expected = compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.audio_head",
        run=lambda: run_audio_head(rt),
        tensors=model_tensors().audio_head,
        input_bindings={"input": "input"},
        output_bindings={"linear": "linear"},
        inputs={"input": llm_output},
        expected=refs.audio_head.execute({"input": llm_output}),
        policy=_Q4_COMPARE_POLICY,
    )
    logits = _expected_array(head_expected, "linear")

    score_expected = refs.token_score.execute(
        {
            "logits": logits,
            "tokens": tokens,
            "audio_mask_id": np.array([audio_mask_id], dtype=np.int64),
            "rng_seed": np.array([rng_seed], dtype=np.uint32),
            "step_index": np.array([step], dtype=np.int64),
            "active_target_len": np.array(active_target_len, dtype=np.int64),
            "cond_target_start": np.array(cond_target_start, dtype=np.int64),
        }
    )
    score_expected = compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.token_score",
        run=lambda: _run_token_score(
            rt,
            step=step,
            rng_seed=rng_seed,
            audio_mask_id=audio_mask_id,
            active_target_len=active_target_len,
            cond_target_start=cond_target_start,
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
            "unmask_count": np.array(unmask_count, dtype=np.uint32),
            "active_target_len": np.array(active_target_len, dtype=np.int64),
            "cond_target_start": np.array(cond_target_start, dtype=np.int64),
        }
    )
    update_expected = compare_vulkan_stage(
        rt,
        name=f"omnivoice.step.{step:04d}.token_update",
        run=lambda: _run_token_update(
            rt,
            unmask_count=unmask_count,
            active_target_len=active_target_len,
            cond_target_start=cond_target_start,
        ),
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
    create_model_tensors()

    model = cast(
        OmniVoice,
        OmniVoice.from_pretrained(
            str(model_dir),
            dtype=torch.float32,
            device_map="cuda",
            attn_implementation="eager",
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
        (1, config.num_audio_codebook, TARGET_CAPACITY),
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
                active_target_len=prepared.target_len,
                cond_target_start=prepared.cond_target_start,
                batch_input_ids=batch_input_ids,
                batch_audio_mask=batch_audio_mask,
                attention_mask=attention_mask,
                tokens=tokens,
                cos=cos,
                sin=sin,
                refs=refs,
            )

        generated_tokens = np.ascontiguousarray(tokens[:, :, : prepared.target_len])
        audio_topology = _audio_decode_topology(
            prepared.target_len,
            config.num_audio_codebook,
        )
        audio_expected = refs.audio_decode.execute({"audio_codes": generated_tokens})
        compare_vulkan_stage(
            rt,
            name="omnivoice.audio_decode",
            run=lambda: run_audio_decode_with_tensors(rt, audio_topology.tensors),
            tensors=audio_topology.tensors,
            input_bindings={"audio_codes": "audio_codes"},
            output_bindings={"conv1d_31": "conv1d_31"},
            inputs={"audio_codes": generated_tokens},
            expected=audio_expected,
            policy=_AUDIO_DECODE_COMPARE_POLICY,
        )
    finally:
        rt.close()
