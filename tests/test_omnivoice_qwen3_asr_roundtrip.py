from __future__ import annotations

from models.omnivoice.pytorch.example import run_inference_to_wav
from models.qwen3_asr.pytorch.example import run_inference


def test_omnivoice_tts_then_qwen3_asr_roundtrip(tmp_path) -> None:
    text = "hello world this is a speech recognition test"
    wav_path = run_inference_to_wav(text=text, output=tmp_path / "omnivoice.wav")

    transcripts = run_inference(audio=wav_path, language="English")

    assert transcripts
    transcript = transcripts[0].text.casefold()
    assert "hello world" in transcript
    assert "speech recognition test" in transcript
