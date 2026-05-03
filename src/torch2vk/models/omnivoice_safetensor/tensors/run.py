"""OmniVoice end-to-end LogicalTensor run tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import LogicalTensor, input_tensor, output_tensor
from torch2vk.models.omnivoice_safetensor.spec import OmniVoiceSpec

from .case import OmniVoiceDebugCase
from .qwen3 import OmniVoiceQwen3RowTensors, omnivoice_qwen3_row_tensors
from .stage0 import OmniVoiceStage0Tensors, omnivoice_stage0_tensors
from .stage1 import OmniVoiceStage1Tensors, omnivoice_stage1_tensors


@dataclass(frozen=True, slots=True)
class OmniVoiceStepTensors:
    tokens_before: LogicalTensor
    stage0: OmniVoiceStage0Tensors
    qwen3_cond: OmniVoiceQwen3RowTensors
    qwen3_uncond: OmniVoiceQwen3RowTensors
    tokens_after: LogicalTensor


@dataclass(frozen=True, slots=True)
class OmniVoiceRunTensors:
    prompt_ids: LogicalTensor
    steps: tuple[OmniVoiceStepTensors, ...]
    stage1: OmniVoiceStage1Tensors
    wav_pcm16: LogicalTensor


def omnivoice_run_tensors(
    *,
    case: OmniVoiceDebugCase,
    spec: OmniVoiceSpec,
    batch: int = 1,
) -> OmniVoiceRunTensors:
    steps = tuple(
        OmniVoiceStepTensors(
            tokens_before=input_tensor(
                "tokens.before",
                dtype="int32",
                shape=(batch, spec.num_audio_codebook, case.target_steps),
            ),
            stage0=omnivoice_stage0_tensors(batch=batch, steps=case.target_steps, spec=spec),
            qwen3_cond=omnivoice_qwen3_row_tensors(
                row="cond",
                batch=batch,
                steps=case.target_steps,
                spec=spec,
                max_seq_len=case.target_steps,
            ),
            qwen3_uncond=omnivoice_qwen3_row_tensors(
                row="uncond",
                batch=batch,
                steps=case.target_steps,
                spec=spec,
                max_seq_len=case.target_steps,
            ),
            tokens_after=output_tensor(
                "tokens.after",
                dtype="int32",
                shape=(batch, spec.num_audio_codebook, case.target_steps),
            ),
        )
        for _ in range(case.num_steps)
    )
    return OmniVoiceRunTensors(
        prompt_ids=input_tensor("input.prompt_ids", dtype="int32", shape=(batch, "prompt")),
        steps=steps,
        stage1=omnivoice_stage1_tensors(batch=batch, steps=case.target_steps),
        wav_pcm16=output_tensor(
            "output.wav_pcm16",
            dtype="int16",
            shape=(batch, "samples"),
        ),
    )
