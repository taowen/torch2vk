"""OmniVoice end-to-end debug boundary order."""

from __future__ import annotations

from torch2vk.e2e_debug import DebugBoundary
from torch2vk.logical import ComparePolicy


def omnivoice_debug_boundaries(*, llm_layers: int = 28) -> tuple[DebugBoundary, ...]:
    return (
        DebugBoundary("tokens.before", 100, ComparePolicy(kind="token")),
        DebugBoundary("stage0.audio_embedding.output", 200, ComparePolicy(kind="tensor")),
        DebugBoundary(
            "stage0.llm.input",
            300,
            ComparePolicy(kind="tensor"),
            artifacts=("stage0.llm.input.cond", "stage0.llm.input.uncond"),
        ),
        *tuple(
            DebugBoundary(
                f"stage0.llm.layer_{layer:02d}.output",
                400 + layer,
                ComparePolicy(kind="tensor"),
                artifacts=(
                    f"stage0.llm.layer_{layer:02d}.output.cond",
                    f"stage0.llm.layer_{layer:02d}.output.uncond",
                ),
            )
            for layer in range(llm_layers)
        ),
        DebugBoundary(
            "stage0.final_norm.input",
            500,
            ComparePolicy(kind="tensor"),
            artifacts=("stage0.final_norm.input.cond", "stage0.final_norm.input.uncond"),
        ),
        DebugBoundary(
            "stage0.final_norm",
            600,
            ComparePolicy(kind="tensor"),
            artifacts=("stage0.final_norm.cond", "stage0.final_norm.uncond"),
        ),
        DebugBoundary(
            "stage0.audio_head.logits",
            700,
            ComparePolicy(kind="tensor"),
            artifacts=("stage0.audio_head.logits.cond", "stage0.audio_head.logits.uncond"),
        ),
        DebugBoundary("state.selection_scores", 800, ComparePolicy(kind="tensor")),
        DebugBoundary("state.update_mask", 900, ComparePolicy(kind="token")),
        DebugBoundary("tokens.after", 1000, ComparePolicy(kind="token")),
        DebugBoundary(
            "generate.final.audio_tokens",
            1100,
            ComparePolicy(kind="token"),
            step_scoped=False,
        ),
        DebugBoundary(
            "stage1.decoder.waveform",
            1200,
            ComparePolicy(kind="tensor"),
            step_scoped=False,
        ),
        DebugBoundary(
            "output.wav_pcm16",
            1300,
            ComparePolicy(kind="tensor"),
            step_scoped=False,
        ),
    )
