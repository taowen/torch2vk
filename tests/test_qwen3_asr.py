"""Qwen3-ASR fixture transcription coverage."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from models.optimized_qwen3_asr import Qwen3AsrRecognizer

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"
_ZH_WAV = _FIXTURE_DIR / "qwen3_asr_zh.wav"
_EXPECTED_ASKNOT_TEXT = (
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)
_EXPECTED_ZH_TEXT = "甚至出现交易几乎停滞的情况。"


@pytest.fixture(scope="module")
def recognizer(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Qwen3AsrRecognizer]:
    artifact_dir = tmp_path_factory.mktemp("qwen3_asr")
    with Qwen3AsrRecognizer.open(artifact_dir=artifact_dir, pytorch_compare=True) as asr:
        yield asr


@pytest.mark.parametrize(
    ("wav", "language", "expected_text"),
    (
        (_ASKNOT_WAV, "English", _EXPECTED_ASKNOT_TEXT),
        (_ZH_WAV, "Chinese", _EXPECTED_ZH_TEXT),
    ),
)
def test_qwen3_asr_transcribes_fixture_wav(
    recognizer: Qwen3AsrRecognizer,
    wav: Path,
    language: str,
    expected_text: str,
) -> None:
    assert recognizer.transcribe(wav, language=language) == expected_text
