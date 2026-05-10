"""Generated PyTorch reference setup."""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path
from typing import cast

import numpy as np
from torch import nn
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

from models.exported_qwen3_asr import reference_specs
from models.exported_qwen3_asr.pytorch_modules import (
    AudioEncoderReference,
    AudioInjectReference,
    TextLayerReference,
    TextReferenceState,
    TokenSelectReference,
    TokenStoreReference,
)
from torch2vk.runtime.reference import ExportedProgramReference, load_exported_reference


@dataclass(slots=True)
class CompareReferences:
    text_state: TextReferenceState
    audio_encoder: AudioEncoderReference
    embed_tokens: ExportedProgramReference
    audio_inject: AudioInjectReference
    text_layers: tuple[TextLayerReference, ...]
    text_norm: ExportedProgramReference
    lm_head: ExportedProgramReference
    decode_embed: ExportedProgramReference
    decode_layers: tuple[TextLayerReference, ...]
    decode_norm: ExportedProgramReference
    decode_lm_head: ExportedProgramReference
    token_select: TokenSelectReference
    token_store: TokenStoreReference
    next_token: np.ndarray | None = None
    done: np.ndarray | None = None


def build_compare_references(
    model: Qwen3ASRForConditionalGeneration,
    *,
    base_dir: Path,
    max_new_tokens: int,
) -> CompareReferences:
    thinker = cast(nn.Module, getattr(model, "thinker"))
    text_state = TextReferenceState(thinker)
    decode_state = text_state
    text_model = cast(nn.Module, thinker.get_submodule("model"))
    embed_tokens = cast(nn.Module, text_model.get_submodule("embed_tokens"))
    norm = cast(nn.Module, text_model.get_submodule("norm"))
    lm_head = cast(nn.Module, thinker.get_submodule("lm_head"))
    audio_tower = cast(nn.Module, thinker.get_submodule("audio_tower"))
    return CompareReferences(
        text_state=text_state,
        audio_encoder=AudioEncoderReference(audio_tower),
        embed_tokens=load_exported_reference(
            base_dir,
            reference_specs.EMBED_TOKENS_SPEC,
            state_dict=embed_tokens.state_dict(),
        ),
        audio_inject=AudioInjectReference(),
        text_layers=tuple(
            TextLayerReference(text_state, layer_idx, prefill=True)
            for layer_idx in range(len(text_state.layers))
        ),
        text_norm=load_exported_reference(
            base_dir,
            reference_specs.TEXT_NORM_SPEC,
            state_dict=norm.state_dict(),
        ),
        lm_head=load_exported_reference(
            base_dir,
            reference_specs.LM_HEAD_SPEC,
            state_dict=lm_head.state_dict(),
        ),
        decode_embed=load_exported_reference(
            base_dir,
            reference_specs.DECODE_EMBED_SPEC,
            state_dict=embed_tokens.state_dict(),
        ),
        decode_layers=tuple(
            TextLayerReference(decode_state, layer_idx, prefill=False)
            for layer_idx in range(len(decode_state.layers))
        ),
        decode_norm=load_exported_reference(
            base_dir,
            reference_specs.DECODE_NORM_SPEC,
            state_dict=norm.state_dict(),
        ),
        decode_lm_head=load_exported_reference(
            base_dir,
            reference_specs.DECODE_LM_HEAD_SPEC,
            state_dict=lm_head.state_dict(),
        ),
        token_select=TokenSelectReference(),
        token_store=TokenStoreReference(max_new_tokens),
    )
