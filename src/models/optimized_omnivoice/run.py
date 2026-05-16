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
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers import AutoTokenizer

from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.dispatch.audio_decode import run_audio_decode_with_tensors
from models.optimized_omnivoice.dispatch.audio_head import run_audio_head
from models.optimized_omnivoice.dispatch.llm_forward import run_llm_forward
from models.optimized_omnivoice.export_gguf import export_omnivoice_q4_k_m_gguf
from models.optimized_omnivoice.input_prep import (
    DEFAULT_TEXT,
    OMNIVOICE_FRAME_RATE,
    OmniVoiceTokenizer,
    SEQ_CAPACITY,
    TARGET_CAPACITY,
    PreparedOmniVoiceInputs,
    chunk_omnivoice_text_for_capacity,
    prepare_omnivoice_inputs,
)
from models.optimized_omnivoice.shaders.omnivoice_cfg_score_f32 import OMNIVOICE_CFG_SCORE_F32
from models.optimized_omnivoice.shaders.omnivoice_input_embed_q8_0_f32 import (
    OMNIVOICE_INPUT_EMBED_Q8_0_F32,
)
from models.optimized_omnivoice.shaders.omnivoice_token_update_topk_f32 import (
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
)
from models.optimized_omnivoice.tensors.audio_decode import AudioDecodeTensors, create_audio_decode
from models.optimized_omnivoice.tensors.model import create_model_tensors, model_tensors
from omnivoice.models.omnivoice import OmniVoiceConfig
from omnivoice.utils.audio import cross_fade_chunks
from models.optimized_omnivoice.pytorch.example import REPO_ID, save_audio_wav
from torch2vk.runtime.host_array import as_float16_attention_mask
from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
from torch2vk.runtime.replay import ReplayPlan, execute_replay
from torch2vk.runtime.replay_cache_key import (
    build_cached_replay_plan,
    cached_replay_plan,
    replay_cache_namespace,
    source_tree_digest,
)
from torch2vk.runtime.rope_table import run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader
from torch2vk.vulkan.types import TensorSpec

DEFAULT_OUTPUT_WAV = Path("/tmp/torch2vk_omnivoice_optimized.wav")
_GENERATION_REPLAY_CACHE = "optimized_omnivoice_generation_step:v11"
get_shader = make_shader_loader("models.optimized_omnivoice.shaders")


_REPLAY_SOURCE_DIGEST = source_tree_digest(__file__)


@dataclass(slots=True)
class _RuntimeCache:
    gguf_dir: Path
    profile_dir: str | None
    generation_cache_namespace: str
    rt: RuntimeSession
    generation_replay_plan: ReplayPlan | None = None
    audio_decode_replay_plans: dict[int, ReplayPlan] | None = None

    def close(self) -> None:
        self.rt.close()


_RUNTIME_CACHE: _RuntimeCache | None = None


@dataclass(frozen=True, slots=True)
class _GeneratedChunk:
    text: str
    tokens: np.ndarray
    waveform: np.ndarray


@dataclass(frozen=True, slots=True)
class _AudioDecodeTopology:
    audio_codes: LogicalTensor
    tensors: AudioDecodeTensors


_AUDIO_DECODE_TOPOLOGIES: dict[int, _AudioDecodeTopology] = {}


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
    gguf_path: Path,
    profile_dir: str | Path | None,
) -> _RuntimeCache:
    profile_key = None if profile_dir is None else str(Path(profile_dir).expanduser().resolve())
    gguf_dir = gguf_path.parent

    global _RUNTIME_CACHE
    if profile_key is None and _RUNTIME_CACHE is not None and _RUNTIME_CACHE.gguf_dir == gguf_dir:
        return _RUNTIME_CACHE

    if _RUNTIME_CACHE is not None:
        _RUNTIME_CACHE.close()
        _RUNTIME_CACHE = None

    create_model_tensors()
    _AUDIO_DECODE_TOPOLOGIES.clear()
    runtime = _RuntimeCache(
        gguf_dir=gguf_dir,
        profile_dir=profile_key,
        generation_cache_namespace=_generation_replay_cache_namespace(gguf_dir),
        rt=RuntimeSession.open(
            device_index=0,
            model_dir=gguf_dir,
            profile_dir=profile_dir,
            model_tensors=model_tensors(),
            get_shader=get_shader,
        ),
        audio_decode_replay_plans={},
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
        start_position=0,
        theta=1_000_000.0,
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


def _run_token_score(rt: RuntimeSession, *, step: int, rng_seed: int) -> None:
    tensors = model_tensors()
    OMNIVOICE_CFG_SCORE_F32(
        rt,
        logits=tensors.audio_head.linear,
        tokens=tensors.tokens,
        audio_mask_id=tensors.audio_mask_id,
        step_index=step,
        rng_seed=rng_seed,
        active_target_len=tensors.active_target_len,
        cond_target_start=tensors.cond_target_start,
        candidate_tokens=tensors.candidate_tokens,
        candidate_scores=tensors.candidate_scores,
    )


def _run_token_update(rt: RuntimeSession, *, unmask_count: int) -> None:
    tensors = model_tensors()
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32(
        rt,
        candidate_tokens=tensors.candidate_tokens,
        candidate_scores=tensors.candidate_scores,
        unmask_count=unmask_count,
        active_target_len=tensors.active_target_len,
        cond_target_start=tensors.cond_target_start,
        tokens=tensors.tokens,
        batch_input_ids=tensors.batch_input_ids,
    )


def _run_generation_step(
    rt: RuntimeSession, *, step: int, unmask_count: int, rng_seed: int
) -> None:
    with rt.frame(f"omnivoice.step.{step:04d}"):
        _run_input_embed(rt)
        run_llm_forward(rt)
        run_audio_head(rt)
        _run_token_score(rt, step=step, rng_seed=rng_seed)
        _run_token_update(rt, unmask_count=unmask_count)


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
    frame: str,
) -> ReplayPlan:
    plan = rt.build_replay_plan(
        name="optimized_omnivoice_audio_decode",
        frame=frame,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("OmniVoice audio decode replay must not use readback slots")
    return plan


def _cached_generation_replay_plan(
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


def _generation_schedule(
    *,
    target_len: int,
    num_audio_codebook: int,
    num_steps: int,
) -> list[int]:
    timesteps = _get_time_steps(0.0, 1.0, num_steps, t_shift=0.1)
    total_mask = target_len * num_audio_codebook
    schedule: list[int] = []
    remaining = total_mask
    for step in range(num_steps):
        if step == num_steps - 1:
            unmask_count = remaining
        else:
            unmask_count = min(
                math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                remaining,
            )
        schedule.append(int(unmask_count))
        remaining -= int(unmask_count)
    return schedule


def _audio_decode_topology(target_len: int, num_audio_codebook: int) -> _AudioDecodeTopology:
    cached = _AUDIO_DECODE_TOPOLOGIES.get(target_len)
    if cached is not None:
        return cached
    audio_codes = LogicalTensor(
        name=f"omnivoice.audio_decode.active_codes.{target_len}",
        spec=TensorSpec(dtype="int64", shape=(1, num_audio_codebook, target_len)),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )
    base = model_tensors().audio_decode
    weight_kwargs = {
        field.name: getattr(base, field.name)
        for field in fields(AudioDecodeTensors)
        if getattr(base, field.name).role is TensorRole.WEIGHT
    }
    tensors = create_audio_decode(
        f"omnivoice.audio_decode.active.{target_len}",
        target_len=target_len,
        audio_codes=audio_codes,
        request_state_outputs=frozenset(("conv1d_31",)),
        **weight_kwargs,
    )
    topology = _AudioDecodeTopology(audio_codes=audio_codes, tensors=tensors)
    _AUDIO_DECODE_TOPOLOGIES[target_len] = topology
    return topology


def _decode_current_audio(
    rt: RuntimeSession,
    runtime: _RuntimeCache,
    *,
    tokens: np.ndarray,
    num_audio_codebook: int,
) -> np.ndarray:
    active_target_len = int(tokens.shape[-1])
    topology = _audio_decode_topology(active_target_len, num_audio_codebook)
    rt.initialize_request_state({topology.audio_codes: tokens[None, :, :]})

    replay_plans = runtime.audio_decode_replay_plans
    if replay_plans is None:
        replay_plans = {}
        runtime.audio_decode_replay_plans = replay_plans
    audio_decode_replay_plan = replay_plans.get(active_target_len)
    if audio_decode_replay_plan is None:
        with rt.frame(f"omnivoice.audio_decode.{active_target_len}"):
            run_audio_decode_with_tensors(rt, topology.tensors)
        audio_decode_replay_plan = _build_audio_decode_replay_plan(
            rt,
            frame=f"omnivoice.audio_decode.{active_target_len}",
        )
        replay_plans[active_target_len] = audio_decode_replay_plan
    else:
        execute_replay(audio_decode_replay_plan)
    waveform = np.ascontiguousarray(rt.read_request_state(topology.tensors.conv1d_31)[0])
    return waveform


def _run_prepared_chunk(
    rt: RuntimeSession,
    runtime: _RuntimeCache,
    *,
    text: str,
    prepared: PreparedOmniVoiceInputs,
    config: OmniVoiceConfig,
    num_steps: int,
) -> _GeneratedChunk:
    num_audio_codebook = config.num_audio_codebook
    target_len = prepared.target_len
    audio_mask_id = config.audio_mask_id
    print(
        "  "
        f"seq_len={prepared.seq_len}/{SEQ_CAPACITY}, "
        f"target_len={target_len}/{TARGET_CAPACITY}, "
        f"cond_audio_start={prepared.cond_audio_start}, "
        f"cond_target_start={prepared.cond_target_start}"
    )

    tokens = np.full(
        (1, num_audio_codebook, TARGET_CAPACITY),
        audio_mask_id,
        dtype=np.int64,
    )
    rt.initialize_request_state(
        {
            model_tensors().batch_input_ids: prepared.batch_input_ids,
            model_tensors().batch_audio_mask: prepared.batch_audio_mask,
            model_tensors().attention_mask: as_float16_attention_mask(prepared.attention_mask),
            model_tensors().audio_mask_id: np.array([audio_mask_id], dtype=np.int64),
            model_tensors().active_target_len: np.array([target_len], dtype=np.uint32),
            model_tensors().cond_target_start: np.array(
                [prepared.cond_target_start],
                dtype=np.uint32,
            ),
            model_tensors().tokens: tokens,
        }
    )

    schedule = _generation_schedule(
        target_len=target_len,
        num_audio_codebook=num_audio_codebook,
        num_steps=num_steps,
    )
    unmasked = 0
    generation_replay_plan = runtime.generation_replay_plan
    generation_start = time.perf_counter()
    rng_seed = 0x1234ABCD
    for step, unmask_count in enumerate(schedule):
        if unmask_count <= 0:
            continue
        if generation_replay_plan is None:
            generation_replay_plan = _cached_generation_replay_plan(
                rt,
                cache_namespace=runtime.generation_cache_namespace,
            )
            if generation_replay_plan is None:
                _run_generation_step(
                    rt,
                    step=step,
                    unmask_count=unmask_count,
                    rng_seed=rng_seed,
                )
                generation_replay_plan = _build_generation_replay_plan(
                    rt,
                    frame=f"omnivoice.step.{step:04d}",
                    cache_namespace=runtime.generation_cache_namespace,
                )
            else:
                print("  Generation replay cache hit")
                execute_replay(
                    generation_replay_plan,
                    dynamic_push_constants={
                        "step_index": step,
                        "unmask_count": unmask_count,
                        "rng_seed": rng_seed,
                    },
                )
            runtime.generation_replay_plan = generation_replay_plan
        else:
            execute_replay(
                generation_replay_plan,
                dynamic_push_constants={
                    "step_index": step,
                    "unmask_count": unmask_count,
                    "rng_seed": rng_seed,
                },
            )
        unmasked += unmask_count
        if step % 8 == 0 or step == num_steps - 1:
            total = num_audio_codebook * target_len
            print(f"  Step {step}: unmasked {unmasked}/{total} ({100 * unmasked / total:.0f}%)")
    generation_elapsed = time.perf_counter() - generation_start
    print(
        f"  Generation: {num_steps} steps in {generation_elapsed:.3f}s "
        f"({generation_elapsed / num_steps * 1000:.1f} ms/step)"
    )
    generated_tokens = np.ascontiguousarray(
        rt.read_request_state(model_tensors().tokens)[0, :, :target_len]
    )

    print("  Decoding audio tokens...")
    decode_start = time.perf_counter()
    waveform = _decode_current_audio(
        rt,
        runtime,
        tokens=generated_tokens,
        num_audio_codebook=num_audio_codebook,
    )
    print(f"  Audio decode: {time.perf_counter() - decode_start:.3f}s")

    return _GeneratedChunk(text=text, tokens=generated_tokens, waveform=waveform)


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

    print("Preparing inputs...")
    text_chunks = chunk_omnivoice_text_for_capacity(
        text,
        tokenizer=text_tokenizer,
        config=config,
        seq_capacity=SEQ_CAPACITY,
        target_capacity=TARGET_CAPACITY,
    )
    print(f"  chunks={len(text_chunks)}, capacity=seq:{SEQ_CAPACITY}, target:{TARGET_CAPACITY}")

    print("Initializing Vulkan runtime...")
    runtime = _runtime_for(
        gguf_path=gguf_path,
        profile_dir=profile_dir,
    )
    rt = runtime.rt

    _run_rope_table(rt, frame_name="omnivoice.rope")

    generated_chunks: list[_GeneratedChunk] = []
    first_chunk_tokens: np.ndarray | None = None
    first_chunk_text: str | None = None
    for chunk_idx, chunk in enumerate(text_chunks):
        print(f"\n=== Chunk {chunk_idx + 1}/{len(text_chunks)} ===")
        prepared = prepare_omnivoice_inputs(
            text=chunk.text,
            tokenizer=text_tokenizer,
            config=config,
            ref_text=first_chunk_text,
            ref_audio_tokens=first_chunk_tokens,
            seq_capacity=SEQ_CAPACITY,
            target_capacity=TARGET_CAPACITY,
        )
        generated = _run_prepared_chunk(
            rt,
            runtime,
            text=chunk.text,
            prepared=prepared,
            config=config,
            num_steps=num_steps,
        )
        generated_chunks.append(generated)
        if first_chunk_tokens is None:
            first_chunk_tokens = generated.tokens
            first_chunk_text = generated.text

    if runtime.profile_dir is not None:
        runtime.close()

    if len(generated_chunks) == 1:
        waveform = generated_chunks[0].waveform
    else:
        waveform = cross_fade_chunks(
            [chunk.waveform for chunk in generated_chunks],
            OMNIVOICE_FRAME_RATE * 960,
        )
    output_path = save_audio_wav(torch.from_numpy(np.ascontiguousarray(waveform)), output_path)
    print(f"\nOutput: {output_path}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
