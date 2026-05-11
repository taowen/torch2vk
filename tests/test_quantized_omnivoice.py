"""OmniVoice Q4_K_M GGUF integration tests."""

from __future__ import annotations

from pathlib import Path

from models.quantized_omnivoice.compare import compare_generation_steps
from models.quantized_omnivoice.run import main


def test_quantized_omnivoice_generation_compare() -> None:
    compare_generation_steps(num_steps=1)


def test_quantized_omnivoice_wav_is_written(tmp_path: Path) -> None:
    wav = main(
        num_steps=32,
        output=tmp_path / "quantized_omnivoice.wav",
    )
    assert wav.exists()
    assert wav.stat().st_size > 0
