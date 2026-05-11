"""Qwen3-ASR fixture transcription coverage."""

from __future__ import annotations

from pathlib import Path

from models.optimized_qwen3_asr.run import main as transcribe_qwen3_asr

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"
_EXPECTED_ASKNOT_TEXT = (
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)


def test_qwen3_asr_transcribes_asknot_fixture() -> None:
    assert transcribe_qwen3_asr(wav_path=_ASKNOT_WAV, language="English") == _EXPECTED_ASKNOT_TEXT
