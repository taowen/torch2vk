"""OmniVoice Qwen3 LogicalTensor tree aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch2vk.models.omnivoice_safetensor.spec import OmniVoiceSpec
from torch2vk.models.qwen3_safetensor.tensors.prefill import (
    Qwen3PrefillTensors,
    qwen3_prefill_tensors,
)


@dataclass(frozen=True, slots=True)
class OmniVoiceQwen3RowTensors:
    row: Literal["cond", "uncond"]
    prefill: Qwen3PrefillTensors


def omnivoice_qwen3_row_tensors(
    *,
    row: Literal["cond", "uncond"],
    batch: int,
    steps: int,
    spec: OmniVoiceSpec,
    max_seq_len: int | None = None,
) -> OmniVoiceQwen3RowTensors:
    return OmniVoiceQwen3RowTensors(
        row=row,
        prefill=qwen3_prefill_tensors(
            batch=batch,
            steps=steps,
            spec=spec.qwen3,
            max_seq_len=max_seq_len,
        ),
    )
