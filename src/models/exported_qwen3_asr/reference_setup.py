"""Generated PyTorch exported graph reference loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)
from models.exported_qwen3_asr import reference_specs
from torch2vk.runtime.reference import ExportedProgramReference, load_exported_reference


@dataclass(frozen=True, slots=True)
class LoadedReferences:
    decode_embed: ExportedProgramReference
    decode_lm_head: ExportedProgramReference
    decode_norm: ExportedProgramReference
    embed_tokens: ExportedProgramReference
    lm_head: ExportedProgramReference
    text_norm: ExportedProgramReference


def load_references(model: Qwen3ASRForConditionalGeneration, *, base_dir: Path) -> LoadedReferences:
    return LoadedReferences(
        decode_embed=load_exported_reference(
            base_dir,
            reference_specs.DECODE_EMBED_SPEC,
            state_dict=model.get_submodule('thinker.model.embed_tokens').state_dict(),
        ),
        decode_lm_head=load_exported_reference(
            base_dir,
            reference_specs.DECODE_LM_HEAD_SPEC,
            state_dict=model.get_submodule('thinker.lm_head').state_dict(),
        ),
        decode_norm=load_exported_reference(
            base_dir,
            reference_specs.DECODE_NORM_SPEC,
            state_dict=model.get_submodule('thinker.model.norm').state_dict(),
        ),
        embed_tokens=load_exported_reference(
            base_dir,
            reference_specs.EMBED_TOKENS_SPEC,
            state_dict=model.get_submodule('thinker.model.embed_tokens').state_dict(),
        ),
        lm_head=load_exported_reference(
            base_dir,
            reference_specs.LM_HEAD_SPEC,
            state_dict=model.get_submodule('thinker.lm_head').state_dict(),
        ),
        text_norm=load_exported_reference(
            base_dir,
            reference_specs.TEXT_NORM_SPEC,
            state_dict=model.get_submodule('thinker.model.norm').state_dict(),
        ),
    )
