"""OmniVoice TTS inference using exported Vulkan shaders.

32-step iterative masked decoding with classifier-free guidance.
Embedding, LLM, audio_head, CFG scoring, and token updates run on Vulkan.

Run from project root:
    .venv/bin/python -m models.exported_omnivoice.run
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from models.hf_cache import resolve_cached_model
from models.exported_omnivoice.dispatch.audio_decode import run_audio_decode
from models.exported_omnivoice.dispatch.audio_head import run_audio_head
from models.exported_omnivoice.dispatch.llm_forward import run_llm_forward
from models.exported_omnivoice.input_prep import DEFAULT_TEXT, prepare_omnivoice_inputs
from models.exported_omnivoice.shaders.omnivoice_cfg_score_f32 import OMNIVOICE_CFG_SCORE_F32
from models.exported_omnivoice.shaders.omnivoice_input_embed_f32 import OMNIVOICE_INPUT_EMBED_F32
from models.exported_omnivoice.shaders.omnivoice_token_update_topk_f32 import (
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
)
from models.exported_omnivoice.tensors.model import create_model_tensors, model_tensors
from omnivoice.models.omnivoice import OmniVoiceConfig
from models.optimized_omnivoice.pytorch.example import REPO_ID, save_audio_wav
from torch2vk.runtime.host_array import as_float16_attention_mask
from torch2vk.runtime.replay import ReplayPlan, execute_replay
from torch2vk.runtime.replay_cache_key import source_tree_digest
from torch2vk.runtime.rope_table import run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader

DEFAULT_OUTPUT_WAV = Path("/tmp/torch2vk_omnivoice_exported.wav")
_GENERATION_REPLAY_CACHE = "exported_omnivoice_generation_step:v1"
get_shader = make_shader_loader("models.exported_omnivoice.shaders")


_REPLAY_SOURCE_DIGEST = source_tree_digest(__file__)


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
    OMNIVOICE_INPUT_EMBED_F32(
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
    plan = rt.build_replay_plan(
        name="exported_omnivoice_generation_step",
        frame=frame,
    )
    if plan.readback_slots:
        raise RuntimeError("OmniVoice generation replay must not use readback slots")
    rt.cache_replay_plan(cache_namespace, plan)
    return plan


def _cached_generation_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    for plan in rt.cached_replay_plans(cache_namespace):
        return plan
    return None


def _generation_replay_cache_namespace(model_dir: Path) -> str:
    return f"{_GENERATION_REPLAY_CACHE}:{_REPLAY_SOURCE_DIGEST}:{model_dir.resolve()}"


def main(
    *,
    text: str = DEFAULT_TEXT,
    output: str | Path = DEFAULT_OUTPUT_WAV,
    num_steps: int = 32,
) -> Path:
    output_path = Path(output)
    model_dir = resolve_cached_model(REPO_ID)
    replay_cache_namespace = _generation_replay_cache_namespace(model_dir)
    config_data = json.loads((model_dir / "config.json").read_text())
    config = OmniVoiceConfig(**config_data)

    print("Loading tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
    create_model_tensors(target_len=target_len)
    expected_seq_len = model_tensors().batch_input_ids.spec.shape[2]
    if expected_seq_len != seq_len:
        raise ValueError(
            f"exported OmniVoice seq_len is {expected_seq_len}, "
            f"but prepared inputs require {seq_len}; regenerate exported_omnivoice"
        )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=model_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )

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
            model_tensors().tokens: tokens,
        }
    )

    # Compute RoPE once on GPU (positions are fixed for masked decoding)
    _run_rope_table(rt, frame_name="omnivoice.rope")

    unmasked = 0
    generation_replay_plan: ReplayPlan | None = None
    for step in range(num_steps):
        k = schedule[step]
        if k <= 0:
            continue

        if generation_replay_plan is None:
            generation_replay_plan = _cached_generation_replay_plan(
                rt,
                cache_namespace=replay_cache_namespace,
            )
            if generation_replay_plan is None:
                _run_generation_step(rt, step=step, unmask_count=k, rng_seed=rng_seed)
                generation_replay_plan = _build_generation_replay_plan(
                    rt,
                    frame=f"omnivoice.step.{step:04d}",
                    cache_namespace=replay_cache_namespace,
                )
            else:
                execute_replay(
                    generation_replay_plan,
                    dynamic_push_constants={
                        "step_index": step,
                        "unmask_count": k,
                        "rng_seed": rng_seed,
                    },
                )
        else:
            execute_replay(
                generation_replay_plan,
                dynamic_push_constants={
                    "step_index": step,
                    "unmask_count": k,
                    "rng_seed": rng_seed,
                },
            )
        unmasked += k

        if step % 8 == 0 or step == num_steps - 1:
            total = num_audio_codebook * target_len
            print(f"  Step {step}: unmasked {unmasked}/{total} ({100 * unmasked / total:.0f}%)")

    # Decode audio tokens
    print("\nDecoding audio tokens...")
    with rt.frame("omnivoice.audio_decode"):
        run_audio_decode(rt)
    waveform = torch.from_numpy(
        np.ascontiguousarray(rt.read_request_state(model_tensors().audio_decode.conv1d_31)[0])
    )
    rt.close()

    # Save wav
    output_path = save_audio_wav(waveform, output_path)
    print(f"\nOutput: {output_path}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
