"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.exported_qwen3_asr.run
"""

from __future__ import annotations

import json
import os
import time
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
from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from models.exported_qwen3_asr import dispatch, reference, reference_setup
from models.exported_qwen3_asr.pytorch_modules import (
    preprocess_audio_inputs,
)
from models.exported_qwen3_asr.shaders import model_shaders
from models.exported_qwen3_asr.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.session import RuntimeSession


def _reference_rope(
    refs: reference_setup.CompareReferences,
    hidden: torch.Tensor,
    position_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids_t = torch.from_numpy(np.ascontiguousarray(position_ids)).cuda().long()
    rotary_emb = cast(nn.Module, refs.text_state.text_model.get_submodule("rotary_emb"))
    with torch.no_grad():
        cos, sin = rotary_emb(hidden, position_ids_t)
    return cos.float(), sin.float()


def _expected_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.from_numpy(np.ascontiguousarray(as_numpy_array(value))).cuda().float()


def _reference_input(value: object) -> reference.ReferenceInput:
    if isinstance(value, torch.Tensor):
        return value
    return as_numpy_array(value)


def _require_gpu_output(tensor: LogicalTensor) -> None:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} did not produce a GPU buffer")


def _run_token_select(
    rt: RuntimeSession,
    *,
    logits: LogicalTensor,
    eos_token_ids: LogicalTensor,
    next_token: LogicalTensor,
    done: LogicalTensor,
    frame_name: str,
) -> LogicalTensor:
    with rt.frame(frame_name):
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=logits,
            eos_token_ids=eos_token_ids,
            next_token=next_token,
            done=done,
        )
    return next_token


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
        QWEN3_ASR_TOKEN_STORE_EOS_F32(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )


def _read_selected_token(rt: RuntimeSession, next_token: LogicalTensor) -> int:
    _require_gpu_output(next_token)
    return int(rt.read_request_state(next_token).reshape(-1)[0])


def _build_decode_replay_plan(
    rt: RuntimeSession,
    *,
    frame: str,
) -> ReplayPlan:
    plan = rt.build_replay_plan(
        name="exported_qwen3_asr_decode_step",
        frame=frame,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("Spike Qwen3-ASR decode replay must not use readback slots")
    rt.cache_replay_plan("exported_qwen3_asr_decode_step:v1", plan)
    return plan


def _load_qwen_reference_model(model_dir: Path) -> Qwen3ASRForConditionalGeneration:
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return cast(Qwen3ASRForConditionalGeneration, model)


def _compare_token_select(
    rt: RuntimeSession,
    *,
    frame_name: str,
    logits: object,
    eos_token_ids: np.ndarray,
    refs: reference_setup.CompareReferences,
) -> dict[str, object]:
    expected = reference.run_token_select(
        rt,
        refs.token_select,
        name=frame_name,
        logits=_reference_input(logits),
        eos_token_ids=eos_token_ids,
    )
    refs.next_token = as_numpy_array(expected["next_token"]).astype(np.int64, copy=False)
    refs.done = as_numpy_array(expected["done"]).astype(np.uint32, copy=False)
    return expected


def _compare_token_store(
    rt: RuntimeSession,
    *,
    frame_name: str,
    refs: reference_setup.CompareReferences,
    next_token: object,
    token_index: object,
    done: object,
) -> dict[str, object]:
    return reference.run_token_store(
        rt,
        refs.token_store,
        name=frame_name,
        next_token=_reference_input(next_token),
        token_index=_reference_input(token_index),
        done=_reference_input(done),
    )


def _run_decode_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    cache_position: np.ndarray,
    eos_token_ids: np.ndarray,
    token_index: np.ndarray,
    refs: reference_setup.CompareReferences,
) -> int:
    tensors = model_tensors()
    if refs.next_token is None:
        raise RuntimeError("decode compare requires a reference next_token from prefill")
    with rt.frame(f"spike.decode.{step:04d}"):
        dispatch.run_decode_embed(rt)
        embed_expected = reference.run_decode_embed(
            rt,
            refs.decode_embed,
            step=step,
            input=refs.next_token,
        )
        ref_hidden = _expected_tensor(embed_expected["embedding"])
        cos, sin = _reference_rope(
            refs,
            ref_hidden,
            np.array([[int(as_numpy_array(cache_position).reshape(-1)[0])]], dtype=np.int64),
        )
        for layer_idx in range(len(tensors.decode_layers)):
            dispatch.run_decode_layer(rt, layer_idx)
            layer_expected = reference.run_decode_layer(
                rt,
                refs.decode_layers[layer_idx],
                step=step,
                layer_idx=layer_idx,
                hidden_states=ref_hidden,
                position_embeddings_0=cos,
                position_embeddings_1=sin,
                cache_position=cache_position,
            )
            ref_hidden = _expected_tensor(layer_expected["add_7"])
        dispatch.run_decode_norm(rt)
        norm_expected = reference.run_decode_norm(
            rt,
            refs.decode_norm,
            step=step,
            hidden_states=ref_hidden,
        )
        ref_hidden = _expected_tensor(norm_expected["mul_1"])
        dispatch.run_decode_lm_head(rt)
        lm_head_expected = reference.run_decode_lm_head(
            rt,
            refs.decode_lm_head,
            step=step,
            input=ref_hidden,
        )
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=tensors.decode_lm_head.linear,
            eos_token_ids=tensors.eos_token_ids,
            next_token=tensors.next_token,
            done=tensors.done,
        )
        token_expected = _compare_token_select(
            rt,
            frame_name=f"spike.decode.{step:04d}.token_select",
            logits=lm_head_expected["linear"],
            eos_token_ids=eos_token_ids,
            refs=refs,
        )
        QWEN3_ASR_TOKEN_STORE_EOS_F32(
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
            next_token=token_expected["next_token"],
            token_index=token_index,
            done=token_expected["done"],
        )
    return int(as_numpy_array(refs.next_token).reshape(-1)[0])


# ==============================================================
# Main pipeline
# ==============================================================

def main(
    *,
    use_replay: bool = True,
    pytorch_compare: bool = False,
    max_new_tokens: int = 64,
) -> str:
    if max_new_tokens <= 0 or max_new_tokens > 64:
        raise ValueError(f"max_new_tokens must be in [1, 64], got {max_new_tokens}")
    if pytorch_compare:
        use_replay = False
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
    )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=model_dir,
        model_tensors=model_tensors(),
        model_shaders=model_shaders(),
    )
    compare_refs: reference_setup.CompareReferences | None = None
    if pytorch_compare:
        print("Loading PyTorch reference for compare...")
        compare_refs = reference_setup.build_compare_references(
            _load_qwen_reference_model(Path(model_dir)),
            base_dir=Path(__file__).parent,
            max_new_tokens=max_new_tokens,
        )

    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float32,
    )
    rt.initialize_request_state(
        {cache: zero_cache for cache in model_tensors().key_caches + model_tensors().value_caches}
    )

    # === Audio Tower ===
    print("\n=== Phase 1: Audio Tower ===")
    preprocessed = preprocess_audio_inputs(
        prepared.input_ids,
        prepared.input_features,
        prepared.feature_attention_mask,
        position_embedding_shape=tuple(
            int(dim) for dim in model_tensors().audio_encoder.position_embedding.spec.shape
        ),
        d_model=ac.d_model,
    )
    print(f"  hidden_states after compact: {model_tensors().audio_encoder.index_select.spec.shape}")
    print(f"  audio encoder ({model_tensors().audio_encoder.x.spec.shape})...")
    rt.register_inputs(
        {
            model_tensors().audio_encoder.x: preprocessed["padded_feature"],
            model_tensors().audio_encoder.position_embedding: preprocessed["position_embedding"],
            model_tensors().audio_encoder.compact_index: preprocessed["compact_index"],
            model_tensors().audio_encoder.attention_mask: preprocessed["audio_attention_mask"],
        }
    )
    ref_audio_features: reference.ReferenceInput | None = None
    with rt.frame("spike.audio"):
        dispatch.run_audio_encoder(rt)
        if compare_refs is not None:
            audio_expected = reference.run_audio_encoder(
                rt,
                compare_refs.audio_encoder,
                x=preprocessed["padded_feature"],
                position_embedding=preprocessed["position_embedding"],
                compact_index=preprocessed["compact_index"],
                attention_mask=preprocessed["audio_attention_mask"],
            )
            ref_audio_features = _reference_input(audio_expected["linear_110"])
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
            model_tensors().prefill_rope.theta: np.array([rope_theta], dtype=np.float32),
        }
    )
    dispatch.run_rope_table(
        rt,
        phase="prefill",
        frame_name="spike.text.prefill_rope",
    )

    # Embedding, audio injection, and decoder layers stay on GPU.
    print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
    prefill_position_ids = np.broadcast_to(
        np.arange(prompt_length, dtype=np.int64)[None, None, :],
        (3, 1, prompt_length),
    ).copy()
    prefill_cache_position = np.arange(prompt_length, dtype=np.int64)
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
        dispatch.run_embed_tokens(rt)
        ref_hidden: reference.ReferenceInput | None = None
        ref_cos: reference.ReferenceInput | None = None
        ref_sin: reference.ReferenceInput | None = None
        if compare_refs is not None:
            embed_expected = reference.run_embed_tokens(
                rt,
                compare_refs.embed_tokens,
                input=prepared.input_ids,
            )
            ref_hidden = _reference_input(embed_expected["embedding"])
        dispatch.run_audio_inject(rt)
        if compare_refs is not None:
            if ref_hidden is None:
                raise RuntimeError("audio inject compare requires reference embed output")
            if ref_audio_features is None:
                raise RuntimeError("audio encoder reference output is required for audio inject compare")
            audio_inject_expected = reference.run_audio_inject(
                rt,
                compare_refs.audio_inject,
                inputs_embeds=ref_hidden,
                audio_positions=preprocessed["audio_positions"],
                audio_features=ref_audio_features,
            )
            ref_hidden = _reference_input(audio_inject_expected["embedding"])
            ref_cos, ref_sin = _reference_rope(
                compare_refs,
                _expected_tensor(ref_hidden),
                prefill_position_ids,
            )
        for layer_idx in range(len(model_tensors().text_layers)):
            dispatch.run_text_layer(rt, layer_idx)
            if compare_refs is not None:
                if ref_hidden is None:
                    raise RuntimeError("text layer compare requires reference hidden state")
                if ref_cos is None or ref_sin is None:
                    raise RuntimeError("text layer compare requires reference rope tensors")
                layer_expected = reference.run_text_layer(
                    rt,
                    compare_refs.text_layers[layer_idx],
                    layer_idx=layer_idx,
                    hidden_states=ref_hidden,
                    position_embeddings_0=ref_cos,
                    position_embeddings_1=ref_sin,
                    cache_position=prefill_cache_position,
                )
                ref_hidden = _reference_input(layer_expected["add_7"])
            if layer_idx % 7 == 6:
                print(f"    layer {layer_idx} done")
        dispatch.run_text_norm(rt)
        if compare_refs is not None:
            if ref_hidden is None:
                raise RuntimeError("text norm compare requires reference hidden state")
            norm_expected = reference.run_text_norm(
                rt,
                compare_refs.text_norm,
                hidden_states=ref_hidden,
            )
            ref_hidden = _reference_input(norm_expected["mul_1"])
        dispatch.run_lm_head(rt)
        if compare_refs is not None:
            if ref_hidden is None:
                raise RuntimeError("lm_head compare requires reference hidden state")
            lm_head_expected = reference.run_lm_head(
                rt,
                compare_refs.lm_head,
                input=ref_hidden,
            )
        else:
            lm_head_expected = None

    print("  lm_head + token_select...")
    _require_gpu_output(model_tensors().lm_head.linear)

    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    rt.initialize_request_state(
        {
            model_tensors().generated_tokens: np.zeros((1, max_new_tokens), dtype=np.int64),
            model_tensors().generated_length: np.zeros((1,), dtype=np.uint32),
            model_tensors().stopped: np.zeros((1,), dtype=np.uint32),
        }
    )
    rt.register_inputs({model_tensors().eos_token_ids: eos_token_array})
    _run_token_select(
        rt,
        logits=model_tensors().lm_head.linear,
        eos_token_ids=model_tensors().eos_token_ids,
        next_token=model_tensors().next_token,
        done=model_tensors().done,
        frame_name="spike.text.token_select",
    )
    if compare_refs is not None:
        if lm_head_expected is None:
            raise RuntimeError("token select compare requires reference lm_head output")
        token_select_expected = _compare_token_select(
            rt,
            frame_name="spike.text.token_select",
            logits=lm_head_expected["linear"],
            eos_token_ids=eos_token_array,
            refs=compare_refs,
        )
    else:
        token_select_expected = None
    rt.register_inputs({model_tensors().token_index: np.array([0], dtype=np.int64)})
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
    if compare_refs is not None:
        if token_select_expected is None:
            raise RuntimeError("token store compare requires token select expected output")
        _compare_token_store(
            rt,
            frame_name="spike.text.token_store",
            refs=compare_refs,
            next_token=token_select_expected["next_token"],
            token_index=np.array([0], dtype=np.int64),
            done=token_select_expected["done"],
        )
    first_token = _read_selected_token(rt, model_tensors().next_token)
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    eos_token_set = set(eos_token_ids)
    generated_tokens = [first_token]
    decode_replay_plan: ReplayPlan | None = None

    # Memory sampling
    memory_trace: list[tuple[int, float, float, float]] = []
    baseline_stats = rt.device.allocation_stats()
    print(f"  Baseline GPU memory: device_local={baseline_stats.device_local_live_bytes / 1024**2:.1f} MB, "
          f"reserved={baseline_stats.device_local_reserved_bytes / 1024**2:.1f} MB")

    decode_start = time.perf_counter()
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
                model_tensors().decode_rope.theta: np.array(
                    [rope_theta],
                    dtype=np.float32,
                ),
            }
        )
        dispatch.run_rope_table(
            rt,
            phase="decode",
            frame_name=f"spike.decode.rope.{step:04d}",
        )

        decode_step_inputs = dispatch.decode_step_inputs(
            cache_position=cache_pos,
            eos_token_array=eos_token_array,
            token_index_value=step + 1,
        )
        if compare_refs is not None:
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
        elif decode_replay_plan is None:
            rt.register_inputs(decode_step_inputs)
            next_token = dispatch.run_decode_step(
                rt,
                step=step,
            )
            generated_tokens.append(next_token)
            if use_replay:
                decode_replay_plan = _build_decode_replay_plan(
                    rt,
                    frame=f"spike.decode.{step:04d}",
                )
        else:
            stage_replay_step_inputs(
                rt,
                plan=decode_replay_plan,
                inputs=decode_step_inputs,
                write_through=(
                    model_tensors().decode_layers[0].cache_position,
                    model_tensors().token_index,
                ),
            )
            execute_replay(decode_replay_plan)

        stats = rt.device.allocation_stats()
        memory_trace.append(
            (
                step,
                stats.device_local_live_bytes / 1024**2,
                stats.device_local_reserved_bytes / 1024**2,
                stats.host_upload_live_bytes / 1024**2,
            )
        )

        if step < 5 or step % 20 == 0:
            if use_replay:
                print(f"  Step {step}: gpu={stats.device_local_live_bytes / 1024**2:.1f}MB")
            else:
                print(f"  Step {step}: token={generated_tokens[-1]}  "
                      f"gpu={stats.device_local_live_bytes / 1024**2:.1f}MB")

    decode_elapsed = time.perf_counter() - decode_start
    decode_steps = len(memory_trace)
    print(f"\n  Decode: {decode_steps} steps in {decode_elapsed:.3f}s "
          f"({decode_elapsed / decode_steps * 1000:.1f} ms/token)" if decode_steps > 0 else "")

    # Memory summary
    print("\n=== GPU Memory Trace ===")
    if memory_trace:
        final_stats = rt.device.allocation_stats()
        print(f"  Peak device_local live: {final_stats.device_local_peak_live_bytes / 1024**2:.1f} MB")
        print(f"  Peak device_local reserved: {final_stats.device_local_peak_reserved_bytes / 1024**2:.1f} MB")
        print(f"  Final device_local live: {final_stats.device_local_live_bytes / 1024**2:.1f} MB")
        print(f"  Steps sampled: {len(memory_trace)}")
        print()
        print("  Step | GPU Live (MB) | GPU Reserved (MB) | Host Upload (MB)")
        print("  -----|---------------|-------------------|------------------")
        for step, device_local_live_mb, device_local_reserved_mb, host_upload_live_mb in memory_trace:
            print(f"  {step:4d} | {device_local_live_mb:13.1f} | "
                  f"{device_local_reserved_mb:17.1f} | {host_upload_live_mb:.1f}")

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
    result = main()
    print(result)
