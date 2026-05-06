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
from models.omnivoice.pytorch.example import REPO_ID
from models.omnivoice.tensors.text import declare_omnivoice_text_tensors
from torch2vk.runtime.session import RuntimeSession


def test_generated_omnivoice_input_embeddings_first_two_ops_match_pytorch(
    tmp_path: Path,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    config = load_omnivoice_export_config(model_dir=model_dir)
    tensors = declare_omnivoice_text_tensors(
        batch_size=1,
        prompt_length=4,
        max_sequence_length=4,
        hidden_size=config.llm_hidden_size,
        vocab_size=config.llm_vocab_size,
        audio_vocab_size=config.audio_vocab_size,
        num_audio_codebook=config.num_audio_codebook,
    ).prefill
    input_ids = _sample_input_ids(
        codebooks=config.num_audio_codebook,
        sequence_length=tensors.input_ids.concrete_shape[-1],
        audio_vocab_size=config.audio_vocab_size,
    )
    audio_mask_bool = np.array([[False, True, False, True]], dtype=np.bool_)
    audio_mask_u32 = np.array([[0, 1, 0, 1]], dtype=np.uint32)
    assert [op.target for op in INPUT_EMBEDDINGS_TORCH_OPS[:2]] == [
        "aten.select.int",
        "aten.embedding.default",
    ]

    with RuntimeSession.open(artifact_dir=tmp_path, model_dir=model_dir) as rt:
        rt.register_inputs({tensors.input_ids: input_ids, tensors.audio_mask: audio_mask_bool})

        assert (
            run_omnivoice_input_embeddings(
                rt,
                tensors,
                max_ops=1,
            )
            is tensors.text_token_ids
        )
        assert [record.shader for record in rt.dispatch_records[-1:]] == [
            "omnivoice_aten_select_int_i64",
        ]
        assert len(rt.compare_results) == 1
        assert all(result.passed for result in rt.compare_results)

    with RuntimeSession.open(artifact_dir=tmp_path, model_dir=model_dir) as rt:
        rt.register_inputs({tensors.input_ids: input_ids, tensors.audio_mask: audio_mask_u32})

        assert (
            run_omnivoice_input_embeddings(
                rt,
                tensors,
                max_ops=4,
                pytorch_compare=False,
            )
            is tensors.audio_mask_for_shift
        )

        assert [record.shader for record in rt.dispatch_records[-2:]] == [
            "omnivoice_aten_select_int_i64",
            "omnivoice_aten_embedding_f32",
        ]


def _sample_input_ids(
    *,
    codebooks: int,
    sequence_length: int,
    audio_vocab_size: int,
) -> np.ndarray:
    ids = np.zeros((1, codebooks, sequence_length), dtype=np.int64)
    ids[:, 0, :] = np.array([42, 7, 314, 9], dtype=np.int64)
    for codebook in range(codebooks):
        ids[0, codebook, 1] = (codebook * 17 + 3) % audio_vocab_size
        ids[0, codebook, 3] = (codebook * 29 + 5) % audio_vocab_size
    return ids
