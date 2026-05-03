"""OmniVoice debug/generation case parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OmniVoiceDebugCase:
    text: str = "hello world"
    language: str = "English"
    target_steps: int = 8
    num_steps: int = 8
    seed: int = 20260501
    guidance_scale: float = 2.0
    layer_penalty_factor: float = 5.0
    t_shift: float = 0.1
    position_temperature: float = 0.0
    class_temperature: float = 0.0
    denoise: bool = False


def default_omnivoice_debug_case() -> OmniVoiceDebugCase:
    return OmniVoiceDebugCase()
