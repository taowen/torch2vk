"""Generated Qwen3-ASR Vulkan adapter coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.generated_qwen3_asr.audio_tower import run_generated_qwen3_asr_audio_tower
from models.generated_qwen3_asr.execution import (
    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
    run_generated_qwen3_asr_greedy_decode_loop,
)
from models.generated_qwen3_asr.export import load_qwen3_asr_export_config
from models.generated_qwen3_asr.tensors.text import declare_generated_qwen3_asr_text_tensors
from models.generated_qwen3_asr.tensors.audio_tower import (
    declare_generated_qwen3_asr_audio_tower_tensors,
)
from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.runtime.session import RuntimeSession


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"


def test_generated_qwen3_asr_audio_tower_runs_audio_tower_and_matches_pytorch(
    tmp_path: Path,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    config = load_qwen3_asr_export_config(model_dir=model_dir)
    _, prepared = prepare_qwen3_asr_inputs(
        model_dir=model_dir,
        wav=_ASKNOT_WAV,
        language="English",
    )
    audio_feature_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(
        prepared.input_features[0, :, :audio_feature_len],
        dtype=np.float32,
    )
    feature_lens = np.array([audio_feature_len], dtype=np.int64)

    tensors = declare_generated_qwen3_asr_audio_tower_tensors(
        input_features_shape=input_features.shape,
        hidden_size=config.audio_hidden_size,
        output_size=config.audio_output_size,
        downsample_hidden_size=config.audio_downsample_hidden_size,
        encoder_layers=config.audio_encoder_layers,
        encoder_ffn_dim=config.audio_encoder_ffn_dim,
    )

    with RuntimeSession.open(
        artifact_dir=tmp_path,
        model_dir=model_dir,
    ) as rt:
        rt.register_inputs(
            {
                tensors.input_features: input_features,
                tensors.feature_lens: feature_lens,
            }
        )

        assert (
            run_generated_qwen3_asr_audio_tower(
                rt, tensors, pytorch_compare=True, max_ops=None
            )
            is tensors.last_hidden_state
        )

        assert [record.shader for record in rt.dispatch_records[:7]] == [
            "qwen3_asr_pad_feature_f32",
            "qwen3_asr_audio_tower_conv2d_gelu_f32",
            "qwen3_asr_audio_tower_conv2d_gelu_f32",
            "qwen3_asr_audio_tower_conv2d_gelu_f32",
            "qwen3_asr_audio_tower_conv_out_f32",
            "qwen3_asr_audio_tower_add_position_f32",
            "qwen3_asr_compact_after_cnn_f32",
        ]
        assert len(rt.dispatch_records) > 7
        assert rt.compare_results
        assert len(rt.compare_results) == 1
        assert all(result.passed for result in rt.compare_results)

        max_new_tokens = 1
        max_sequence_new_tokens = 2
        text_tensors = declare_generated_qwen3_asr_text_tensors(
            prompt_length=prepared.prompt_length,
            audio_tokens=tensors.last_hidden_state.concrete_shape[0],
            max_sequence_length=prepared.prompt_length + max_sequence_new_tokens,
            hidden_size=config.text_hidden_size,
            intermediate_size=config.text_intermediate_size,
            vocab_size=config.text_vocab_size,
            decoder_layers=config.text_decoder_layers,
            num_attention_heads=config.text_num_attention_heads,
            num_key_value_heads=config.text_num_key_value_heads,
            head_dim=config.text_head_dim,
            audio_features=tensors.last_hidden_state,
            pytorch_input_features_shape=prepared.input_features.shape,
            pytorch_feature_attention_mask_shape=prepared.feature_attention_mask.shape,
        )
        rt.register_inputs(
            {
                text_tensors.prefill.input_ids: prepared.input_ids,
                text_tensors.prefill.attention_mask: prepared.attention_mask,
                text_tensors.prefill.input_features: prepared.input_features,
                text_tensors.prefill.feature_attention_mask: prepared.feature_attention_mask,
                text_tensors.token_select.eos_token_ids: np.array(
                    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
                    dtype=np.int64,
                ),
            }
        )
        generated = run_generated_qwen3_asr_greedy_decode_loop(
            rt,
            text_tensors,
            max_new_tokens=max_new_tokens,
            rope_theta=config.text_rope_theta,
            mrope_section=config.text_mrope_section,
            pytorch_compare=True,
            stop_on_eos=False,
        )
        produced = rt.read_request_state(generated)
        assert produced.shape == (1, prepared.prompt_length + max_sequence_new_tokens)
        generated_length = int(rt.read_request_state(text_tensors.token_store.generated_length)[0])
        assert generated_length == max_new_tokens
        assert rt.compare_results
        assert all(result.passed for result in rt.compare_results)

        generated_no_compare = run_generated_qwen3_asr_greedy_decode_loop(
            rt,
            text_tensors,
            max_new_tokens=max_sequence_new_tokens,
            rope_theta=config.text_rope_theta,
            mrope_section=config.text_mrope_section,
            pytorch_compare=False,
            stop_on_eos=False,
        )
        produced_no_compare = rt.read_request_state(generated_no_compare)
        assert produced_no_compare.shape == (1, prepared.prompt_length + max_sequence_new_tokens)
        generated_length_no_compare = int(
            rt.read_request_state(text_tensors.token_store.generated_length)[0]
        )
        assert generated_length_no_compare == max_sequence_new_tokens
