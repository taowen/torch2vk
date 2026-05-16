"""Compare quantized FLUX.2 Klein 9B Vulkan denoiser against PyTorch ROCm."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from models.quantized_klein9b import reference
from models.quantized_klein9b.dispatch.flux import run_flux
from models.quantized_klein9b.export_gguf import DEFAULT_OUTPUT_DIR
from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.pytorch_reference import (
    FluxStreamingReference,
    configure_rocm_reference,
)
from models.quantized_klein9b.run import (
    _batched_prc_img,
    _batched_prc_txt,
    _ensure_ggufs,
    _get_schedule,
)
from models.quantized_klein9b.text_encoder import Qwen3TextEncoder
from models.quantized_klein9b.tensors.model import create_model_tensors, flux_output, model_tensors
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader


DEFAULT_PROMPT = "a clean watercolor illustration of a small red sailboat on a calm blue sea under morning light"
get_shader = make_shader_loader("models.quantized_klein9b.shaders")


@dataclass(frozen=True, slots=True)
class Klein9BCompareResult:
    width: int
    height: int
    steps: int
    max_abs: float
    mean_abs: float
    rms_abs: float


def compare_flux_steps(
    *,
    prompt: str = DEFAULT_PROMPT,
    model_dir: str | Path | None = None,
    text_encoder_dir: str | Path | None = None,
    ae_dir: str | Path | None = None,
    gguf_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ctx_cache: str | Path | None = None,
    width: int = 256,
    height: int = 256,
    seed: int = 7,
    num_steps: int = 1,
) -> Klein9BCompareResult:
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"width and height must be divisible by 16, got {width}x{height}")
    _log("resolving model dirs")
    configure_rocm_reference()
    model_dirs = resolve_model_dirs(
        model_dir=model_dir,
        text_encoder_dir=text_encoder_dir,
        ae_dir=ae_dir,
    )
    ggufs = _ensure_ggufs(
        model_dir=model_dirs.flux,
        text_encoder_dir=model_dirs.text_encoder,
        ae_dir=model_dirs.ae,
        output_dir=Path(gguf_dir).expanduser().resolve(),
    )
    _log("loading text context")
    ctx_np, ctx_ids_np = _load_or_build_context(
        prompt=prompt,
        text_encoder_dir=model_dirs.text_encoder,
        ctx_cache=Path(ctx_cache).expanduser().resolve() if ctx_cache is not None else None,
    )

    latent_shape = (1, 128, height // 16, width // 16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latent = torch.randn(
        latent_shape,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    img, img_ids = _batched_prc_img(latent)
    img_np = img.detach().cpu().float().numpy().astype(np.float16)
    img_ids_np = img_ids.detach().cpu().numpy().astype(np.int64)
    del latent, img, img_ids
    torch.cuda.empty_cache()

    _log("opening Vulkan runtime")
    create_model_tensors(image_seq_len=img_np.shape[1], text_seq_len=ctx_np.shape[1])
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=ggufs.flux.parent,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    ref = FluxStreamingReference(model_dirs.flux)
    timesteps = _get_schedule(num_steps, img_np.shape[1])
    max_abs = 0.0
    mean_abs = 0.0
    rms_abs = 0.0
    try:
        _log("materializing Vulkan weights")
        rt.materialize_model_weights()
        for step, (t_curr, t_prev) in enumerate(
            zip(timesteps[:-1], timesteps[1:], strict=True),
            start=1,
            ):
            _log(f"step {step}: running Vulkan denoiser")
            inputs = {
                _tensor_name(model_tensors().flux.x.name): img_np,
                _tensor_name(model_tensors().flux.x_ids.name): img_ids_np,
                _tensor_name(model_tensors().flux.timesteps.name): np.array(
                    [t_curr],
                    dtype=np.float16,
                ),
                _tensor_name(model_tensors().flux.ctx.name): ctx_np,
                _tensor_name(model_tensors().flux.ctx_ids.name): ctx_ids_np,
            }
            with rt.request(inputs=inputs):
                with rt.frame("klein9b.flux"):
                    run_flux(rt)
                expected = reference.run_flux(
                    rt,
                    ref,
                    x=img_np,
                    x_ids=img_ids_np,
                    timesteps=np.array([t_curr], dtype=np.float16),
                    ctx=ctx_np,
                    ctx_ids=ctx_ids_np,
                    guidance=np.array([0.0], dtype=np.float16),
                )
                pred = rt.read_request_state(flux_output()).astype(np.float32)
            _log(f"step {step}: compared Vulkan output against PyTorch reference")
            step_max, step_mean, step_rms = _error_metrics(pred, expected["linear_120"])
            max_abs = max(max_abs, step_max)
            mean_abs = step_mean
            rms_abs = step_rms
            print(
                json.dumps(
                    {
                        "step": step,
                        "max_abs": step_max,
                        "mean_abs": step_mean,
                        "rms_abs": step_rms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            img_np = (img_np.astype(np.float32) + (t_prev - t_curr) * pred).astype(np.float16)
    finally:
        ref.close()
        rt.close()

    result = Klein9BCompareResult(
        width=width,
        height=height,
        steps=num_steps,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms_abs=rms_abs,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2), flush=True)
    return result


def _load_or_build_context(
    *,
    prompt: str,
    text_encoder_dir: Path,
    ctx_cache: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if ctx_cache is not None and ctx_cache.is_file():
        cached = np.load(ctx_cache)
        return (
            np.ascontiguousarray(cached["ctx"]).astype(np.float16),
            np.ascontiguousarray(cached["ctx_ids"]).astype(np.int64),
        )
    text_encoder = Qwen3TextEncoder(str(text_encoder_dir), device="cuda")
    ctx = text_encoder([prompt]).to(torch.bfloat16)
    ctx, ctx_ids = _batched_prc_txt(ctx)
    ctx_np = ctx.detach().cpu().float().numpy().astype(np.float16)
    ctx_ids_np = ctx_ids.detach().cpu().numpy().astype(np.int64)
    if ctx_cache is not None:
        ctx_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ctx_cache, ctx=ctx_np, ctx_ids=ctx_ids_np)
    del text_encoder, ctx, ctx_ids
    torch.cuda.empty_cache()
    return ctx_np, ctx_ids_np


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _error_metrics(candidate: np.ndarray, expected: object) -> tuple[float, float, float]:
    expected_np = np.asarray(
        expected.detach().cpu().float().numpy() if isinstance(expected, torch.Tensor) else expected,
        dtype=np.float32,
    )
    diff = candidate.astype(np.float32) - expected_np
    return (
        float(np.max(np.abs(diff))),
        float(np.mean(np.abs(diff))),
        float(np.sqrt(np.mean(diff * diff))),
    )


def _tensor_name(name: str | None) -> str:
    if name is None:
        raise RuntimeError("Klein9B tensor name is missing")
    return name


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--text-encoder-dir", type=Path)
    parser.add_argument("--ae-dir", type=Path)
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ctx-cache", type=Path)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=1)
    args = parser.parse_args()
    compare_flux_steps(
        prompt=args.prompt,
        model_dir=args.model_dir,
        text_encoder_dir=args.text_encoder_dir,
        ae_dir=args.ae_dir,
        gguf_dir=args.gguf_dir,
        ctx_cache=args.ctx_cache,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    _main_cli()
