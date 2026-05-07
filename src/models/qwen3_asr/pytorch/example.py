"""Qwen3-ASR high-performance PyTorch inference example.

This path intentionally avoids the vendored `Qwen3ASRModel.transcribe()` helper
so the example controls the PyTorch execution knobs directly. The vendored
cache decode path is not correctness-safe for this model revision, so the
baseline keeps `use_cache=False`, enables SDPA by default, and compiles the
repeated text-model path.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from models.hf_cache import resolve_cached_model
from models.quiet import configure_quiet_runtime, suppress_output

_DEFAULT_TEST_WAV = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "qwen3_asr_asknot.wav"

REPO_ID = "Qwen/Qwen3-ASR-0.6B"


class _Qwen3AsrProcessorLike(Protocol):
    def __call__(
        self,
        *,
        text: Sequence[str],
        audio: Sequence[object],
        return_tensors: str,
        padding: bool,
    ) -> Any: ...

    def apply_chat_template(
        self,
        messages: object,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str: ...

    def batch_decode(
        self,
        sequences: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class Qwen3AsrTranscript:
    language: str
    text: str


def default_audio_path() -> Path:
    if not _DEFAULT_TEST_WAV.is_file():
        raise FileNotFoundError(
            f"Test wav is missing at {_DEFAULT_TEST_WAV}"
        )
    return _DEFAULT_TEST_WAV


def run_inference(
    *,
    model_dir: str | Path | None = None,
    audio: str | Path | None = None,
    language: str | None = "English",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    max_new_tokens: int = 256,
    attn_implementation: str = "sdpa",
    compile_mode: str = "reduce-overhead",
) -> list[Qwen3AsrTranscript]:
    """Run Qwen3-ASR with compiled PyTorch generation."""
    configure_quiet_runtime()
    _configure_torch_performance()

    from qwen_asr.core.transformers_backend import (
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from qwen_asr.inference.utils import (
        normalize_audios,
        normalize_language_name,
        parse_asr_output,
        validate_language,
    )

    resolved = resolve_cached_model(REPO_ID, model_dir)
    audio_path = default_audio_path() if audio is None else Path(audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input wav does not exist: {audio_path}")

    force_language = None
    if language is not None and language.strip():
        force_language = normalize_language_name(language)
        validate_language(force_language)

    with suppress_output():
        processor = cast(
            _Qwen3AsrProcessorLike,
            Qwen3ASRProcessor.from_pretrained(
                str(resolved),
                fix_mistral_regex=True,
            ),
        )
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            str(resolved),
            dtype=dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        model.eval()
        model.thinker.model.compile(
            mode=compile_mode,
            fullgraph=False,
        )

        prompt = _build_text_prompt(
            processor=processor,
            force_language=force_language,
        )
        waveform = normalize_audios(str(audio_path))[0]
        inputs = processor(
            text=[prompt],
            audio=[waveform],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        _normalize_generation_inputs(inputs)

        sequences = _generate_no_cache(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )
        decoded = processor.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    if len(decoded) != 1:
        raise ValueError(f"Expected one decoded transcript, got {len(decoded)}")
    parsed_language, text = parse_asr_output(decoded[0], user_language=force_language)
    return [Qwen3AsrTranscript(language=parsed_language, text=text)]


def _configure_torch_performance() -> None:
    torch.set_float32_matmul_precision("high")


def _normalize_generation_inputs(inputs: Any) -> None:
    for key in ("attention_mask", "feature_attention_mask"):
        value = inputs.get(key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.bool:
            inputs[key] = value.to(dtype=torch.int64)


def _generate_no_cache(
    *,
    model: Any,
    inputs: Any,
    max_new_tokens: int,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
    return generated.sequences[:, inputs["input_ids"].shape[1]:]


def _build_text_prompt(
    *,
    processor: _Qwen3AsrProcessorLike,
    force_language: str | None,
) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if force_language:
        prompt = f"{prompt}language {force_language}<asr_text>"
    return str(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="English")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    results = run_inference(
        model_dir=args.model_dir,
        audio=args.audio,
        language=args.language,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
    )
    for index, item in enumerate(results):
        print(f"[{index}] language={item.language!r}")
        print(f"[{index}] text={item.text!r}")


if __name__ == "__main__":
    main()
