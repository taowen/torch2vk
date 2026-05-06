"""Generated OmniVoice Vulkan adapter coverage."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from models.hf_cache import resolve_cached_model
from models.omnivoice.export import load_omnivoice_export_config
from models.omnivoice.input_embeddings import (
    INPUT_EMBEDDINGS_TORCH_OPS,
    run_omnivoice_input_embeddings,
)
from models.omnivoice.text_prefill import run_omnivoice_text_prefill
from models.omnivoice.token_select import run_omnivoice_token_select
from models.omnivoice.pytorch.example import REPO_ID
from models.omnivoice.tensors.text import declare_omnivoice_text_tensors
from torch2vk.runtime.session import RuntimeSession


def test_generated_omnivoice_input_embeddings_runs_full_chain_and_matches_pytorch(
    tmp_path: Path,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    config = load_omnivoice_export_config(model_dir=model_dir)
    text_tensors = declare_omnivoice_text_tensors(
        batch_size=1,
        prompt_length=4,
        max_sequence_length=4,
        hidden_size=config.llm_hidden_size,
        vocab_size=config.llm_vocab_size,
        audio_vocab_size=config.audio_vocab_size,
        num_audio_codebook=config.num_audio_codebook,
    )
    tensors = text_tensors.prefill
    token_select = text_tensors.token_select
    input_ids = _sample_input_ids(
        codebooks=config.num_audio_codebook,
        sequence_length=tensors.input_ids.concrete_shape[-1],
        audio_vocab_size=config.audio_vocab_size,
    )
    audio_mask_bool = np.array([[False, True, False, True]], dtype=np.bool_)
    assert [op.target for op in INPUT_EMBEDDINGS_TORCH_OPS[:2]] == [
        "aten.select.int",
        "aten.embedding.default",
    ]

    with RuntimeSession.open(artifact_dir=tmp_path, model_dir=model_dir) as rt:
        rt.register_inputs({tensors.input_ids: input_ids, tensors.audio_mask: audio_mask_bool})

        assert run_omnivoice_input_embeddings(
            rt,
            tensors,
            max_ops=10,
            pytorch_compare=True,
        ) is tensors.inputs_embeds
        assert [record.shader for record in rt.dispatch_records[-6:]] == [
            "omnivoice_aten_select_int_i64",
            "omnivoice_aten_embedding_f32",
            "omnivoice_aten_shifted_ids_i64",
            "omnivoice_aten_embedding_3d_f32",
            "omnivoice_aten_sum_dim1_f32",
            "omnivoice_aten_where_f32",
        ]

        codebook_offsets = np.arange(
            0,
            config.audio_vocab_size * config.num_audio_codebook,
            config.audio_vocab_size,
            dtype=np.int32,
        )
        guided_logits = np.random.default_rng(0).standard_normal(
            token_select.guided_logits.concrete_shape,
            dtype=np.float32,
        )
        current_tokens = np.zeros(token_select.current_tokens.concrete_shape, dtype=np.int32)
        penalty = np.linspace(
            0.0, 0.3, config.num_audio_codebook, dtype=np.float32
        ).reshape(1, config.num_audio_codebook)
        rt.register_inputs(
            {
                token_select.layer_ids: codebook_offsets,
                token_select.guided_logits: guided_logits,
                token_select.current_tokens: current_tokens,
                token_select.position_scores: penalty,
            }
        )
        assert run_omnivoice_token_select(rt, token_select) is token_select.updated_tokens
        assert [record.shader for record in rt.dispatch_records[-3:]] == [
            "omnivoice_codebook_argmax_f32",
            "omnivoice_codebook_argmax_scores_f32",
            "omnivoice_argmax_select_apply_fused_s",
        ]
        long_text_tensors = declare_omnivoice_text_tensors(
            batch_size=1,
            prompt_length=36,
            max_sequence_length=36,
            hidden_size=config.llm_hidden_size,
            vocab_size=config.llm_vocab_size,
            audio_vocab_size=config.audio_vocab_size,
            num_audio_codebook=config.num_audio_codebook,
        )
        long_token_select = long_text_tensors.token_select
        long_guided_logits = np.random.default_rng(1).standard_normal(
            long_token_select.guided_logits.concrete_shape,
            dtype=np.float32,
        )
        long_current_tokens = np.zeros(long_token_select.current_tokens.concrete_shape, dtype=np.int32)
        long_penalty = np.linspace(
            0.0, 0.3, config.num_audio_codebook, dtype=np.float32
        ).reshape(1, config.num_audio_codebook)
        rt.register_inputs(
            {
                long_token_select.layer_ids: codebook_offsets,
                long_token_select.guided_logits: long_guided_logits,
                long_token_select.current_tokens: long_current_tokens,
                long_token_select.position_scores: long_penalty,
            }
        )
        assert run_omnivoice_token_select(rt, long_token_select) is long_token_select.updated_tokens
        assert [record.shader for record in rt.dispatch_records[-3:]] == [
            "omnivoice_codebook_argmax_f32",
            "omnivoice_codebook_argmax_scores_f32",
            "omnivoice_argmax_select_apply_fused_l",
        ]
        assert rt.compare_results
        assert len(rt.compare_results) == 1
        assert all(result.passed for result in rt.compare_results)

        assert (
            run_omnivoice_text_prefill(
                rt,
                tensors,
                pytorch_compare=True,
            )
            is tensors.inputs_embeds
        )
        assert [record.shader for record in rt.dispatch_records[-6:]] == [
            "omnivoice_aten_select_int_i64",
            "omnivoice_aten_embedding_f32",
            "omnivoice_aten_shifted_ids_i64",
            "omnivoice_aten_embedding_3d_f32",
            "omnivoice_aten_sum_dim1_f32",
            "omnivoice_aten_where_f32",
        ]
        assert rt.compare_results
        assert len(rt.compare_results) == 2
        assert all(result.passed for result in rt.compare_results)

        long_prefill = long_text_tensors.prefill
        long_input_ids = _sample_input_ids(
            codebooks=config.num_audio_codebook,
            sequence_length=long_prefill.input_ids.concrete_shape[-1],
            audio_vocab_size=config.audio_vocab_size,
        )
        long_audio_mask_bool = np.zeros(long_prefill.audio_mask.concrete_shape, dtype=np.bool_)
        long_audio_mask_bool[:, 1::2] = True
        rt.register_inputs(
            {
                long_prefill.input_ids: long_input_ids,
                long_prefill.audio_mask: long_audio_mask_bool,
            }
        )
        assert run_omnivoice_text_prefill(
            rt,
            long_prefill,
            pytorch_compare=True,
        ) is long_prefill.inputs_embeds
        assert rt.compare_results
        assert len(rt.compare_results) == 3
        assert all(result.passed for result in rt.compare_results)

        assert run_omnivoice_input_embeddings(
            rt,
            tensors,
            max_ops=None,
            pytorch_compare=False,
        ) is tensors.inputs_embeds

        assert [record.shader for record in rt.dispatch_records[-6:]] == [
            "omnivoice_aten_select_int_i64",
            "omnivoice_aten_embedding_f32",
            "omnivoice_aten_shifted_ids_i64",
            "omnivoice_aten_embedding_3d_f32",
            "omnivoice_aten_sum_dim1_f32",
            "omnivoice_aten_where_f32",
        ]


def _sample_input_ids(
    *,
    codebooks: int,
    sequence_length: int,
    audio_vocab_size: int,
) -> np.ndarray:
    ids = np.zeros((1, codebooks, sequence_length), dtype=np.int64)
    seed_tokens = np.array([42, 7, 314, 9], dtype=np.int64)
    copy_len = min(sequence_length, seed_tokens.shape[0])
    ids[:, 0, :copy_len] = seed_tokens[:copy_len]
    for codebook in range(codebooks):
        ids[0, codebook, 1] = (codebook * 17 + 3) % audio_vocab_size
        ids[0, codebook, 3] = (codebook * 29 + 5) % audio_vocab_size
    return ids
