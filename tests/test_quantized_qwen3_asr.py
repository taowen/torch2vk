"""Quantized Qwen3-ASR integration coverage."""

from __future__ import annotations

import json
from pathlib import Path

from models.quantized_qwen3_asr.run import main

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"
_ZH_WAV = _FIXTURE_DIR / "qwen3_asr_zh.wav"


def test_quantized_qwen3_asr_replay_cache_hits_across_wav_lengths(tmp_path: Path) -> None:
    first_text = main(
        max_new_tokens=2,
        wav_path=_ASKNOT_WAV,
        profile_dir=tmp_path / "first",
    )
    assert first_text == "And so"

    second_text = main(
        max_new_tokens=2,
        wav_path=_ZH_WAV,
        profile_dir=tmp_path / "second",
    )
    assert second_text

    second_rows = _profile_rows(tmp_path / "second")
    assert not any(
        row.get("phase") == "record" and row.get("frame") == "spike.decode.0000"
        for row in second_rows
    )
    assert any(
        row.get("phase") == "replay"
        and row.get("replay_plan") == "quantized_qwen3_asr_decode_step"
        for row in second_rows
    )


def _profile_rows(profile_dir: Path) -> tuple[dict[str, object], ...]:
    dispatches_path = profile_dir / "dispatches.jsonl"
    return tuple(
        json.loads(line)
        for line in dispatches_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
