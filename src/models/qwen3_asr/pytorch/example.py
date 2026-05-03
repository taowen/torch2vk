"""Qwen3-ASR official-source PyTorch inference example.

The official `qwen_asr` source is vendored in `src/qwen_asr`. This wrapper keeps
our public type boundary small: official objects are accepted through protocols
and normalized into `Qwen3AsrTranscript`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch

from models.hf_cache import resolve_cached_model
from models.omnivoice.pytorch.example import DEFAULT_OUTPUT_WAV as OMNIVOICE_OUTPUT_WAV

REPO_ID = "Qwen/Qwen3-ASR-0.6B"


class Qwen3AsrResultLike(Protocol):
    language: str
    text: str


class Qwen3AsrRuntime(Protocol):
    def transcribe(
        self,
        *,
        audio: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> Sequence[Qwen3AsrResultLike]: ...


class Qwen3AsrModelFactory(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *,
        dtype: torch.dtype,
        device_map: str,
        attn_implementation: str,
        max_inference_batch_size: int,
        max_new_tokens: int,
    ) -> Qwen3AsrRuntime: ...


@dataclass(frozen=True, slots=True)
class Qwen3AsrTranscript:
    language: str
    text: str


def default_audio_path() -> Path:
    if not OMNIVOICE_OUTPUT_WAV.is_file():
        raise FileNotFoundError(
            "OmniVoice wav is missing. Generate it first with: "
            "uv run python -m models.omnivoice.pytorch.example"
        )
    return OMNIVOICE_OUTPUT_WAV


def normalize_transcripts(results: Sequence[Qwen3AsrResultLike]) -> list[Qwen3AsrTranscript]:
    return [Qwen3AsrTranscript(language=item.language, text=item.text) for item in results]


def run_inference(
    *,
    model_dir: str | Path | None = None,
    audio: str | Path | None = None,
    language: str | None = "English",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> list[Qwen3AsrTranscript]:
    """Run the official Qwen3-ASR transformers backend from vendored source."""
    from qwen_asr import Qwen3ASRModel

    factory: Qwen3AsrModelFactory = Qwen3ASRModel
    resolved = resolve_cached_model(REPO_ID, model_dir)
    audio_path = default_audio_path() if audio is None else Path(audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input wav does not exist: {audio_path}")
    model = factory.from_pretrained(
        str(resolved),
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    return normalize_transcripts(
        model.transcribe(
            audio=str(audio_path),
            language=language,
            return_time_stamps=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="English")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    results = run_inference(
        model_dir=args.model_dir,
        audio=args.audio,
        language=args.language,
        device=args.device,
    )
    for index, item in enumerate(results):
        print(f"[{index}] language={item.language!r}")
        print(f"[{index}] text={item.text!r}")


if __name__ == "__main__":
    main()
