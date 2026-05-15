"""OmniVoice TTS inference using exported Vulkan shaders.

32-step iterative masked decoding with classifier-free guidance.
Embedding, LLM, audio_head, CFG scoring, and token updates run on Vulkan.

Run from project root:
    .venv/bin/python -m models.optimized_omnivoice.run
"""

from __future__ import annotations

import atexit
import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers import AutoTokenizer

from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.dispatch.audio_decode import run_audio_decode
from models.optimized_omnivoice.dispatch.audio_head import run_audio_head
from models.optimized_omnivoice.dispatch.llm_forward import run_llm_forward
from models.optimized_omnivoice.export_gguf import export_omnivoice_q4_k_m_gguf
from models.optimized_omnivoice.input_prep import (
    DEFAULT_TEXT,
    OmniVoiceTokenizer,
    prepare_omnivoice_inputs,
)
from models.optimized_omnivoice.shaders.omnivoice_cfg_score_f32 import OMNIVOICE_CFG_SCORE_F32
from models.optimized_omnivoice.shaders.omnivoice_input_embed_q8_0_f32 import (
    OMNIVOICE_INPUT_EMBED_Q8_0_F32,
)
from models.optimized_omnivoice.shaders.omnivoice_token_update_topk_f32 import (
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
)
from models.optimized_omnivoice.tensors.model import create_model_tensors, model_tensors
from omnivoice.models.omnivoice import OmniVoiceConfig
from models.optimized_omnivoice.pytorch.example import REPO_ID, save_audio_wav
from torch2vk.runtime.host_array import as_float16_attention_mask
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.replay_cache_key import (
    build_cached_replay_plan,
    cached_replay_plan,
    replay_cache_namespace,
    source_tree_digest,
)
from torch2vk.runtime.rope_table import run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader

DEFAULT_OUTPUT_WAV = Path("/tmp/torch2vk_omnivoice_optimized.wav")
_GENERATION_REPLAY_CACHE = "optimized_omnivoice_generation_step:v8"
_AUDIO_DECODE_REPLAY_CACHE = "optimized_omnivoice_audio_decode:v5"
get_shader = make_shader_loader("models.optimized_omnivoice.shaders")


_REPLAY_SOURCE_DIGEST = source_tree_digest(__file__)


@dataclass(slots=True)
class _RuntimeCache:
    target_len: int
    gguf_dir: Path
    profile_dir: str | None
    generation_cache_namespace: str
    audio_decode_cache_namespace: str
    rt: RuntimeSession
    generation_replay_plan: ReplayPlan | None = None
    audio_decode_replay_plan: ReplayPlan | None = None

    def close(self) -> None:
        self.rt.close()


_RUNTIME_CACHE: _RuntimeCache | None = None


@lru_cache(maxsize=2)
def _cached_gguf_path(model_dir: str) -> Path:
    return export_omnivoice_q4_k_m_gguf(model_dir=Path(model_dir))


@lru_cache(maxsize=2)
def _cached_config(model_dir: str) -> OmniVoiceConfig:
    config_data = json.loads((Path(model_dir) / "config.json").read_text())
    return OmniVoiceConfig(**config_data)


@lru_cache(maxsize=2)
def _cached_tokenizer(model_dir: str) -> OmniVoiceTokenizer:
    return cast(OmniVoiceTokenizer, AutoTokenizer.from_pretrained(Path(model_dir)))


def _runtime_for(
    *,
    target_len: int,
    seq_len: int,
    gguf_path: Path,
    profile_dir: str | Path | None,
) -> _RuntimeCache:
    profile_key = None if profile_dir is None else str(Path(profile_dir).expanduser().resolve())
    gguf_dir = gguf_path.parent

    global _RUNTIME_CACHE
    if (
        profile_key is None
        and _RUNTIME_CACHE is not None
        and _RUNTIME_CACHE.target_len == target_len
        and _RUNTIME_CACHE.gguf_dir == gguf_dir
    ):
        return _RUNTIME_CACHE

    if _RUNTIME_CACHE is not None:
        _RUNTIME_CACHE.close()
        _RUNTIME_CACHE = None

    create_model_tensors(target_len=target_len)
    expected_seq_len = model_tensors().batch_input_ids.spec.shape[2]
    if expected_seq_len != seq_len:
        raise ValueError(
            f"optimized OmniVoice seq_len is {expected_seq_len}, "
            f"but prepared inputs require {seq_len}; regenerate optimized_omnivoice"
        )
    runtime = _RuntimeCache(
        target_len=target_len,
        gguf_dir=gguf_dir,
        profile_dir=profile_key,
        generation_cache_namespace=_generation_replay_cache_namespace(gguf_dir),
        audio_decode_cache_namespace=_audio_decode_replay_cache_namespace(gguf_dir),
        rt=RuntimeSession.open(
            device_index=0,
            model_dir=gguf_dir,
            profile_dir=profile_dir,
            model_tensors=model_tensors(),
            get_shader=get_shader,
        ),
    )
    if profile_key is None:
        _RUNTIME_CACHE = runtime
    return runtime


def _close_runtime_cache() -> None:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        _RUNTIME_CACHE.close()
        _RUNTIME_CACHE = None


atexit.register(_close_runtime_cache)


def _get_time_steps(t_start: float, t_end: float, num_step: int, t_shift: float) -> np.ndarray:
    t = np.linspace(t_start, t_end, num_step + 1, dtype=np.float64)
    t = t_shift * t / (1.0 + (t_shift - 1.0) * t)
    return t.astype(np.float32)



def _run_rope_table(rt: RuntimeSession, *, frame_name: str) -> None:
    rope_t = model_tensors().rope
    run_rope_table_f32(
        rt,
        start_position=rope_t.start_position,
        theta=rope_t.theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


def _run_input_embed(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    OMNIVOICE_INPUT_EMBED_Q8_0_F32(
        rt,
        text_weight=tensors.text_embedding_weight,
        audio_weight=tensors.audio_embedding_weight,
        batch_input_ids=tensors.batch_input_ids,
        batch_audio_mask=tensors.batch_audio_mask,
        hidden_states=tensors.llm_forward.hidden_states,
    )


def _run_token_score(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    OMNIVOICE_CFG_SCORE_F32(
        rt,
        logits=tensors.audio_head.linear,
        tokens=tensors.tokens,
        audio_mask_id=tensors.audio_mask_id,
        rng_seed=tensors.rng_seed,
        step_index=tensors.step_index,
        candidate_tokens=tensors.candidate_tokens,
        candidate_scores=tensors.candidate_scores,
    )


def _run_token_update(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32(
        rt,
        candidate_tokens=tensors.candidate_tokens,
        candidate_scores=tensors.candidate_scores,
        unmask_count=tensors.unmask_count,
        tokens=tensors.tokens,
        batch_input_ids=tensors.batch_input_ids,
    )


def _run_generation_step(rt: RuntimeSession, *, step: int) -> None:
    with rt.frame(f"omnivoice.step.{step:04d}"):
        _run_input_embed(rt)
        run_llm_forward(rt)
        run_audio_head(rt)
        _run_token_score(rt)
        _run_token_update(rt)


def _generation_step_inputs(step: int, unmask_count: int) -> dict[LogicalTensor, np.ndarray]:
    return {
        model_tensors().step_index: np.array([step], dtype=np.uint32),
        model_tensors().unmask_count: np.array([unmask_count], dtype=np.uint32),
    }


def _build_generation_replay_plan(
    rt: RuntimeSession,
    *,
    frame: str,
    cache_namespace: str,
) -> ReplayPlan:
    return build_cached_replay_plan(
        rt,
        namespace=cache_namespace,
        name="optimized_omnivoice_generation_step",
        frame=frame,
        readback_error="OmniVoice generation replay must not use readback slots",
    )


def _build_audio_decode_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan:
    return build_cached_replay_plan(
        rt,
        namespace=cache_namespace,
        name="optimized_omnivoice_audio_decode",
        frame="omnivoice.audio_decode",
        readback_error="OmniVoice audio decode replay must not use readback slots",
    )


def _cached_generation_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    return cached_replay_plan(rt, namespace=cache_namespace)


def _cached_audio_decode_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    return cached_replay_plan(rt, namespace=cache_namespace)


def _generation_replay_cache_namespace(model_dir: Path) -> str:
    return replay_cache_namespace(
        name=_GENERATION_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
    )


def _audio_decode_replay_cache_namespace(model_dir: Path) -> str:
    return replay_cache_namespace(
        name=_AUDIO_DECODE_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
    )


def main(
    *,
    text: str = DEFAULT_TEXT,
    output: str | Path = DEFAULT_OUTPUT_WAV,
    num_steps: int = 32,
    profile_dir: str | Path | None = None,
) -> Path:
    output_path = Path(output)
    model_dir = resolve_cached_model(REPO_ID)
    model_dir_key = str(model_dir)
    gguf_path = _cached_gguf_path(model_dir_key)
    config = _cached_config(model_dir_key)

    print("Loading tokenizer...")
    text_tokenizer = _cached_tokenizer(model_dir_key)

    audio_mask_id = config.audio_mask_id

    # Prepare inputs (host-side _prepare_inference_inputs)
    print("Preparing inputs...")
    prepared = prepare_omnivoice_inputs(
        text=text,
        tokenizer=text_tokenizer,
        config=config,
    )
    B = 1
    num_audio_codebook = config.num_audio_codebook
    target_len = prepared.target_len
    seq_len = prepared.seq_len
    batch_input_ids = prepared.batch_input_ids
    batch_audio_mask = prepared.batch_audio_mask
    attn_mask_np = as_float16_attention_mask(prepared.attention_mask)

    print(
        f"  seq_len={seq_len}, target_len={target_len}, cond_audio_start={prepared.cond_audio_start}"
    )

    # Create runtime and tensors
    print("Initializing Vulkan runtime...")

    print("Declaring tensors...")
    runtime = _runtime_for(
        target_len=target_len,
        seq_len=seq_len,
        gguf_path=gguf_path,
        profile_dir=profile_dir,
    )
    rt = runtime.rt

    # Iterative decoding
    print(f"\n=== Iterative Decoding ({num_steps} steps) ===")
    tokens = np.full(
        (B, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=np.int64,
    )

    timesteps = _get_time_steps(0.0, 1.0, num_steps, t_shift=0.1)
    total_mask = target_len * num_audio_codebook
    schedule = []
    rem = total_mask
    for step in range(num_steps):
        if step == num_steps - 1:
            num = rem
        else:
            num = min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    rng_seed = 0x1234ABCD
    rt.initialize_request_state(
        {
            model_tensors().batch_input_ids: batch_input_ids,
            model_tensors().batch_audio_mask: batch_audio_mask,
            model_tensors().attention_mask: attn_mask_np,
            model_tensors().audio_mask_id: np.array([audio_mask_id], dtype=np.int64),
            model_tensors().rng_seed: np.array([rng_seed], dtype=np.uint32),
            model_tensors().tokens: tokens,
        }
    )

    # Compute RoPE once on GPU (positions are fixed for masked decoding)
    rt.register_inputs(
        {
            model_tensors().rope.start_position: np.array([0], dtype=np.int64),
            model_tensors().rope.theta: np.array([1_000_000.0], dtype=np.float32),
        }
    )
    _run_rope_table(rt, frame_name="omnivoice.rope")

    unmasked = 0
    generation_replay_plan = runtime.generation_replay_plan
    generation_start = time.perf_counter()
    for step in range(num_steps):
        k = schedule[step]
        if k <= 0:
            continue

        step_inputs = _generation_step_inputs(step, k)
        if generation_replay_plan is None:
            rt.register_inputs(step_inputs)
            generation_replay_plan = _cached_generation_replay_plan(
                rt,
                cache_namespace=runtime.generation_cache_namespace,
            )
            if generation_replay_plan is None:
                _run_generation_step(rt, step=step)
                generation_replay_plan = _build_generation_replay_plan(
                    rt,
                    frame=f"omnivoice.step.{step:04d}",
                    cache_namespace=runtime.generation_cache_namespace,
                )
                runtime.generation_replay_plan = generation_replay_plan
            else:
                runtime.generation_replay_plan = generation_replay_plan
                stage_replay_step_inputs(
                    rt,
                    plan=generation_replay_plan,
                    inputs=step_inputs,
                    write_through=(
                        model_tensors().step_index,
                        model_tensors().unmask_count,
                    ),
                )
                execute_replay(generation_replay_plan)
        else:
            stage_replay_step_inputs(
                rt,
                plan=generation_replay_plan,
                inputs=step_inputs,
                write_through=(
                    model_tensors().step_index,
                    model_tensors().unmask_count,
                ),
            )
            execute_replay(generation_replay_plan)
        unmasked += k

        if step % 8 == 0 or step == num_steps - 1:
            total = num_audio_codebook * target_len
            print(f"  Step {step}: unmasked {unmasked}/{total} ({100 * unmasked / total:.0f}%)")
    generation_elapsed = time.perf_counter() - generation_start
    print(
        f"  Generation: {num_steps} steps in {generation_elapsed:.3f}s "
        f"({generation_elapsed / num_steps * 1000:.1f} ms/step)"
    )

    # Decode audio tokens
    print("\nDecoding audio tokens...")
    decode_start = time.perf_counter()
    audio_decode_replay_plan = runtime.audio_decode_replay_plan
    if audio_decode_replay_plan is None:
        audio_decode_replay_plan = _cached_audio_decode_replay_plan(
            rt,
            cache_namespace=runtime.audio_decode_cache_namespace,
        )
        if audio_decode_replay_plan is None:
            with rt.frame("omnivoice.audio_decode"):
                run_audio_decode(rt)
            audio_decode_replay_plan = _build_audio_decode_replay_plan(
                rt,
                cache_namespace=runtime.audio_decode_cache_namespace,
            )
        else:
            execute_replay(audio_decode_replay_plan)
        runtime.audio_decode_replay_plan = audio_decode_replay_plan
    else:
        execute_replay(audio_decode_replay_plan)
    waveform = torch.from_numpy(
        np.ascontiguousarray(rt.read_request_state(model_tensors().audio_decode.conv1d_31)[0])
    )
    decode_elapsed = time.perf_counter() - decode_start
    if runtime.profile_dir is not None:
        runtime.close()
    print(f"  Audio decode: {decode_elapsed:.3f}s")

    # Save wav
    output_path = save_audio_wav(waveform, output_path)
    print(f"\nOutput: {output_path}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
