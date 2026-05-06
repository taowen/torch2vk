"""Generated Qwen3-ASR Vulkan adapter coverage."""

from __future__ import annotations

from pathlib import Path

from models.generated_qwen3_asr.transcribe import transcribe_wav_generated
from models.hf_cache import resolve_cached_model
from models.qwen3_asr.transcribe import transcribe_wav
from models.qwen3_asr.pytorch.example import REPO_ID


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"


def test_generated_qwen3_asr_audio_tower_runs_audio_tower_and_matches_pytorch(
    tmp_path: Path,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    generated_text_compare = transcribe_wav_generated(
        _ASKNOT_WAV,
        language="English",
        max_new_tokens=16,
        artifact_dir=tmp_path / "generated_transcribe_compare",
        model_dir=model_dir,
        pytorch_compare=True,
    )
    assert generated_text_compare.strip()

    generated_text = transcribe_wav_generated(
        _ASKNOT_WAV,
        language="English",
        max_new_tokens=32,
        artifact_dir=tmp_path / "generated_transcribe",
        model_dir=model_dir,
        pytorch_compare=False,
    )
    reference_text = transcribe_wav(
        _ASKNOT_WAV,
        language="English",
        max_new_tokens=32,
        artifact_dir=tmp_path / "reference_transcribe",
        model_dir=model_dir,
        pytorch_compare=False,
    )
    generated_text_norm = generated_text.strip()
    reference_text_norm = reference_text.strip()
    assert generated_text_norm == reference_text_norm
