"""Spike Qwen3-ASR full pipeline transcription test."""

from __future__ import annotations

from models.exported_qwen3_asr.compare import compare_decode_steps
from models.exported_qwen3_asr.run import main

_EXPECTED_ASKNOT_TEXT = (
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)


def test_exported_qwen3_asr_transcribes_asknot() -> None:
    assert main() == _EXPECTED_ASKNOT_TEXT


def test_exported_qwen3_asr_pytorch_compare_decode_step() -> None:
    assert compare_decode_steps(max_new_tokens=2) == "And so"
