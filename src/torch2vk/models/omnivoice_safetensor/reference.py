"""OmniVoice reference providers."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, cast

import numpy as np
import torch

from torch2vk.reference_trace import ReferenceTrace, TraceReferenceProvider


def omnivoice_official_reference_provider() -> TraceReferenceProvider:
    return TraceReferenceProvider(
        capture=capture_official_omnivoice_trace,
        provider_id="omnivoice_safetensor.official_generate.v2",
    )


def capture_official_omnivoice_trace(inputs: Mapping[str, Any]) -> ReferenceTrace:
    try:
        module: Any = import_module("omnivoice.models.omnivoice")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Official OmniVoice reference capture requires the `omnivoice` package"
        ) from exc
    omnivoice_cls = module.OmniVoice
    generation_config_cls = module.OmniVoiceGenerationConfig

    from .model_directory import resolve_omnivoice_model_dir

    text = str(inputs["text"])
    language = str(inputs.get("language", "English"))
    target_steps = int(inputs["target_steps"])
    num_steps = int(inputs.get("num_steps", target_steps))
    seed = int(inputs.get("seed", 20260501))
    position_temperature = float(inputs.get("position_temperature", 0.0))
    guidance_scale = float(inputs.get("guidance_scale", 2.0))
    layer_penalty_factor = float(inputs.get("layer_penalty_factor", 5.0))
    t_shift = float(inputs.get("t_shift", 0.1))
    class_temperature = float(inputs.get("class_temperature", 0.0))
    denoise = bool(inputs.get("denoise", False))

    _set_torch_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = resolve_omnivoice_model_dir(cast("str | None", inputs.get("model_dir")))
    model = omnivoice_cls.from_pretrained(
        str(model_dir),
        device_map=device_map,
        dtype=torch.float16,
    ).eval()
    gen_config = generation_config_cls(
        num_step=num_steps,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        layer_penalty_factor=layer_penalty_factor,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        denoise=denoise,
        preprocess_prompt=True,
        postprocess_output=True,
    )
    audio_tokenizer = model.audio_tokenizer
    duration = target_steps / float(audio_tokenizer.config.frame_rate)
    with torch.inference_mode():
        task: Any = model._preprocess_all(  # noqa: SLF001
            text=text,
            language=language,
            duration=duration,
            preprocess_prompt=gen_config.preprocess_prompt,
        )
        if int(task.batch_size) != 1:
            raise RuntimeError(f"OmniVoice debug expects batch_size=1, got {task.batch_size}")
        actual_target_steps = int(task.target_lens[0])
        if actual_target_steps != target_steps:
            raise RuntimeError(
                "Official target length mismatch: "
                f"expected {target_steps}, got {actual_target_steps}"
            )
        generated_tokens = model._generate_iterative(task, gen_config)[0]  # noqa: SLF001
        output: object = model._decode_and_post_process(  # noqa: SLF001
            generated_tokens,
            task.ref_rms[0],
            gen_config,
        )
    tokens = {
        "generate.final.audio_tokens": generated_tokens.detach().cpu().to(torch.int32).contiguous(),
    }
    wav = _official_output_to_tensor(output)
    tensors = {
        "output.wav": wav,
        "output.wav_pcm16": _wav_to_pcm16(wav),
    }
    return ReferenceTrace(
        tensors=tensors,
        tokens=tokens,
        timeline=(
            {
                "boundary": "output.wav",
                "kind": "tensor",
                "scope": "",
                "source": "official_omnivoice.generate",
            },
            {
                "boundary": "generate.final.audio_tokens",
                "kind": "token",
                "scope": "",
                "source": "official_omnivoice.generate_iterative",
            },
        ),
        metadata={
            "source": "official_omnivoice",
            "target_steps": target_steps,
            "num_steps": num_steps,
        },
    )


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _official_output_to_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.detach().cpu()
    if isinstance(output, list):
        if not output:
            raise TypeError("Official OmniVoice output list is empty")
        return torch.as_tensor(output[0]).detach().cpu()
    if isinstance(output, np.ndarray):
        return torch.as_tensor(output).detach().cpu()
    wav = getattr(output, "wav", None)
    if wav is not None:
        return torch.as_tensor(wav).detach().cpu()
    audio = getattr(output, "audio", None)
    if audio is not None:
        return torch.as_tensor(audio).detach().cpu()
    raise TypeError(f"Unsupported OmniVoice official output type: {type(output).__name__}")


def _wav_to_pcm16(wav: torch.Tensor) -> torch.Tensor:
    return (
        wav.detach()
        .cpu()
        .float()
        .clamp(min=-1.0, max=1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .contiguous()
    )
