from __future__ import annotations

import numpy as np

from models.omnivoice.pytorch.example import normalize_audio
from models.qwen3_asr.pytorch.example import Qwen3AsrTranscript, normalize_transcripts


def test_omnivoice_audio_normalization() -> None:
    audio = normalize_audio([np.zeros(4, dtype=np.float32)])
    assert tuple(audio.shape) == (4,)


def test_qwen3_asr_transcript_normalization() -> None:
    class Result:
        language = "English"
        text = "hello"

    assert normalize_transcripts([Result()]) == [Qwen3AsrTranscript(language="English", text="hello")]
