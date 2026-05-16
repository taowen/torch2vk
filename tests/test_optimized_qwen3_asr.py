"""Optimized Qwen3-ASR fixture transcription coverage."""

from __future__ import annotations

from pathlib import Path

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.export_gguf import export_qwen3_asr_q4_k_m_gguf
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.optimized_qwen3_asr.run import main as transcribe_qwen3_asr
from models.optimized_qwen3_asr.run import get_shader
from torch2vk.runtime.session import RuntimeSession

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"
_ZH_WAV = _FIXTURE_DIR / "qwen3_asr_zh.wav"
_EXPECTED_ASKNOT_TEXT = (
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)
_EXPECTED_ZH_TEXT = "甚至出现交易几乎停滞的情况。"


def test_optimized_qwen3_asr_transcribes_fixtures_with_shared_runtime() -> None:
    model_dir = resolve_cached_model(REPO_ID)
    gguf_path = export_qwen3_asr_q4_k_m_gguf(model_dir=model_dir)
    with RuntimeSession.open(
        device_index=0,
        model_dir=gguf_path.parent,
        get_shader=get_shader,
    ) as rt:
        assert (
            transcribe_qwen3_asr(wav_path=_ASKNOT_WAV, language="English", rt=rt)
            == _EXPECTED_ASKNOT_TEXT
        )
        assert (
            transcribe_qwen3_asr(wav_path=_ZH_WAV, language="Chinese", rt=rt)
            == _EXPECTED_ZH_TEXT
        )
