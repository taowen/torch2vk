"""Generated PyTorch exported graph reference loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omnivoice.models.omnivoice import OmniVoice
from models.exported_omnivoice import reference_specs
from torch2vk.runtime.reference import ExportedProgramReference, load_exported_reference


@dataclass(frozen=True, slots=True)
class LoadedReferences:
    audio_head: ExportedProgramReference


def load_references(model: OmniVoice, *, base_dir: Path) -> LoadedReferences:
    return LoadedReferences(
        audio_head=load_exported_reference(
            base_dir,
            reference_specs.AUDIO_HEAD_SPEC,
            state_dict=model.get_submodule('audio_heads').state_dict(),
        ),
    )
