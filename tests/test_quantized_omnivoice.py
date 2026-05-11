"""OmniVoice Q4_K_M GGUF integration tests."""

from __future__ import annotations

from pathlib import Path

from models.optimized_qwen3_asr import Qwen3AsrRecognizer
from models.quantized_omnivoice.compare import compare_generation_steps
from models.quantized_omnivoice.run import main

_EXPECTED_TEXT = "Hello world. This is a speech recognition test."


def test_quantized_omnivoice_generation_compare() -> None:
    compare_generation_steps(num_steps=1)


def test_quantized_omnivoice_wav_transcribes_prompt(tmp_path: Path) -> None:
    wav = main(
        num_steps=32,
        output=tmp_path / "quantized_omnivoice.wav",
    )
    with Qwen3AsrRecognizer.open(
        artifact_dir=tmp_path / "qwen3_asr",
        pytorch_compare=False,
    ) as asr:
        assert asr.transcribe(wav, language="English") == _EXPECTED_TEXT
