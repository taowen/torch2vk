"""OmniVoice official-source PyTorch example.

The official `omnivoice` source is vendored in `src/omnivoice`. The default CLI
generates a wav file that can be consumed by `models.qwen3_asr.pytorch.example`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch

from models.hf_cache import resolve_cached_model
from models.quiet import configure_quiet_runtime, suppress_output

REPO_ID = "k2-fsa/OmniVoice"
DEFAULT_OUTPUT_WAV = Path("/tmp/torch2vk_omnivoice.wav")
SAMPLE_RATE = 24_000


OmniVoiceAudio = torch.Tensor | np.ndarray


def normalize_audio(audio: OmniVoiceAudio | Sequence[OmniVoiceAudio]) -> torch.Tensor:
    if isinstance(audio, Sequence):
        if len(audio) == 0:
            raise ValueError("OmniVoice.generate returned an empty audio list")
        audio = audio[0]
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio)
    if isinstance(audio, torch.Tensor):
        return audio
    raise TypeError(f"OmniVoice.generate returned {type(audio).__name__}, expected audio array")


def run_official_generate(
    *,
    model_dir: str | Path | None = None,
    text: str,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> OmniVoiceAudio | Sequence[OmniVoiceAudio]:
    """Run the official OmniVoice API from vendored source."""
    configure_quiet_runtime()
    from omnivoice import OmniVoice

    resolved = resolve_cached_model(REPO_ID, model_dir)
    with suppress_output():
        model = OmniVoice.from_pretrained(resolved, device_map=device, dtype=dtype)
        audio = model.generate(
            text=text,
            ref_audio=None if ref_audio is None else str(ref_audio),
            ref_text=ref_text,
        )
    return audio


def save_audio_wav(audio: torch.Tensor, path: str | Path, *, sample_rate: int = SAMPLE_RATE) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    waveform = audio.detach().float().cpu()
    if waveform.ndim == 2 and waveform.shape[0] <= waveform.shape[1]:
        waveform = waveform.transpose(0, 1)
    sf.write(output, waveform.numpy(), sample_rate)
    return output


def run_inference_to_wav(
    *,
    model_dir: str | Path | None = None,
    text: str,
    output: str | Path = DEFAULT_OUTPUT_WAV,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> Path:
    audio = normalize_audio(run_official_generate(
        model_dir=model_dir,
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        device=device,
        dtype=dtype,
    ))
    return save_audio_wav(audio, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--text", default="hello world this is a speech recognition test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_WAV)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = run_inference_to_wav(
        model_dir=args.model_dir,
        text=args.text,
        output=args.output,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        device=args.device,
    )
    print(f"output_wav={output}")


if __name__ == "__main__":
    main()
