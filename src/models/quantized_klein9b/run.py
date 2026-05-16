"""Run the full FLUX.2 Klein 9B text-to-image pipeline."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from models.quantized_klein9b.autoencoder import AutoEncoder, AutoEncoderParams
from models.quantized_klein9b.dispatch.flux import run_flux
from models.quantized_klein9b.export_gguf import (
    DEFAULT_OUTPUT_DIR,
    Klein9BGGUFPaths,
    export_klein9b_q4_k_m_ggufs,
)
from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.text_encoder import Qwen3TextEncoder
from models.quantized_klein9b.tensors.model import (
    create_model_tensors,
    flux_output,
    model_tensors,
)
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader


get_shader = make_shader_loader("models.quantized_klein9b.shaders")


@dataclass(frozen=True, slots=True)
class Klein9BRunResult:
    image_path: str
    width: int
    height: int
    num_steps: int
    text_encoder_elapsed: float
    denoise_elapsed: float
    ae_elapsed: float


def _get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = _compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    shifted = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1))
    return [float(value) for value in shifted.tolist()]


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    return float(m_200 - 200.0 * a + a * num_steps)


def _prc_txt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = x.shape[0]
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(1),
        torch.arange(1),
        torch.arange(length),
    )
    return x, ids.to(x.device)


def _batched_prc_txt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    ids = []
    for item in x:
        token, token_ids = _prc_txt(item)
        tokens.append(token)
        ids.append(token_ids)
    return torch.stack(tokens), torch.stack(ids)


def _prc_img(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = x.shape
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(height),
        torch.arange(width),
        torch.arange(1),
    )
    return rearrange(x, "c h w -> (h w) c"), ids.to(x.device)


def _batched_prc_img(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    ids = []
    for item in x:
        token, token_ids = _prc_img(item)
        tokens.append(token)
        ids.append(token_ids)
    return torch.stack(tokens), torch.stack(ids)


def _scatter_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    images = []
    for data, pos in zip(x, x_ids, strict=True):
        channels = data.shape[1]
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        t_unique = torch.unique(t_ids, sorted=True)
        t_remap = torch.zeros((int(torch.max(t_ids)) + 1,), device=data.device, dtype=t_ids.dtype)
        t_remap[t_unique] = torch.arange(len(t_unique), device=data.device, dtype=t_ids.dtype)
        t_ids = t_remap[t_ids]
        t = int(torch.max(t_ids)) + 1
        h = int(torch.max(h_ids)) + 1
        w = int(torch.max(w_ids)) + 1
        flat_ids = t_ids * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, channels), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        images.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return torch.cat(images).squeeze(2)


def _load_autoencoder(ae_dir: Path) -> AutoEncoder:
    weight_path = ae_dir / "ae.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"AutoEncoder weights are missing: {weight_path}")
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    state_dict = load_safetensors(weight_path, device="cuda")
    ae.load_state_dict(state_dict, strict=True, assign=True)
    return ae.to("cuda").eval()


def _ensure_ggufs(
    *,
    model_dir: Path,
    text_encoder_dir: Path,
    ae_dir: Path,
    output_dir: Path,
) -> Klein9BGGUFPaths:
    paths = Klein9BGGUFPaths(
        flux=output_dir / "flux" / "model.gguf",
        text_encoder=output_dir / "text_encoder" / "model.gguf",
        ae=output_dir / "ae" / "model.gguf",
    )
    if paths.flux.is_file() and paths.text_encoder.is_file() and paths.ae.is_file():
        return paths
    return export_klein9b_q4_k_m_ggufs(
        model_dir=model_dir,
        text_encoder_dir=text_encoder_dir,
        ae_dir=ae_dir,
        output_dir=output_dir,
    )


def _tensor_name(name: str | None) -> str:
    if name is None:
        raise RuntimeError("Klein9B input tensors must be named before request execution")
    return name


def main(
    *,
    prompt: str,
    model_dir: str | Path | None = None,
    text_encoder_dir: str | Path | None = None,
    ae_dir: str | Path | None = None,
    output: str | Path = "klein9b.png",
    gguf_dir: str | Path = DEFAULT_OUTPUT_DIR,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    num_steps: int = 4,
    profile_dir: str | Path | None = None,
) -> Klein9BRunResult:
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"width and height must be divisible by 16, got {width}x{height}")
    if num_steps != 4:
        raise ValueError(f"FLUX.2 Klein 9B is distilled for 4 steps, got {num_steps}")

    model_dirs = resolve_model_dirs(
        model_dir=model_dir,
        text_encoder_dir=text_encoder_dir,
        ae_dir=ae_dir,
    )
    output_dir = Path(gguf_dir).expanduser().resolve()
    ggufs = _ensure_ggufs(
        model_dir=model_dirs.flux,
        text_encoder_dir=model_dirs.text_encoder,
        ae_dir=model_dirs.ae,
        output_dir=output_dir,
    )

    text_start = time.perf_counter()
    text_encoder = Qwen3TextEncoder(str(model_dirs.text_encoder), device="cuda")
    ctx = text_encoder([prompt]).to(torch.bfloat16)
    ctx, ctx_ids = _batched_prc_txt(ctx)
    text_elapsed = time.perf_counter() - text_start
    text_encoder = text_encoder.cpu()
    torch.cuda.empty_cache()

    latent_shape = (1, 128, height // 16, width // 16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latent = torch.randn(latent_shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    img, img_ids = _batched_prc_img(latent)

    create_model_tensors(image_seq_len=img.shape[1], text_seq_len=ctx.shape[1])
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=ggufs.flux.parent,
        profile_dir=profile_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    rt.materialize_model_weights()

    img_np = img.detach().cpu().float().numpy().astype(np.float16)
    img_ids_np = img_ids.detach().cpu().numpy().astype(np.int64)
    ctx_np = ctx.detach().cpu().float().numpy().astype(np.float16)
    ctx_ids_np = ctx_ids.detach().cpu().numpy().astype(np.int64)
    timesteps = _get_schedule(num_steps, img.shape[1])

    denoise_start = time.perf_counter()
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=True):
        inputs = {
            _tensor_name(model_tensors().flux.x.name): img_np,
            _tensor_name(model_tensors().flux.x_ids.name): img_ids_np,
            _tensor_name(model_tensors().flux.timesteps.name): np.array([t_curr], dtype=np.float16),
            _tensor_name(model_tensors().flux.ctx.name): ctx_np,
            _tensor_name(model_tensors().flux.ctx_ids.name): ctx_ids_np,
        }
        with rt.request(inputs=inputs):
            with rt.frame("klein9b.flux"):
                run_flux(rt)
            pred = rt.read_request_state(flux_output()).astype(np.float32)
        img_np = (img_np.astype(np.float32) + (t_prev - t_curr) * pred).astype(np.float16)
    rt.close()
    denoise_elapsed = time.perf_counter() - denoise_start

    ae_start = time.perf_counter()
    ae = _load_autoencoder(model_dirs.ae)
    img_tokens = torch.from_numpy(img_np).to("cuda", dtype=torch.bfloat16)
    img_ids_t = torch.from_numpy(img_ids_np).to("cuda")
    latent_grid = _scatter_ids(img_tokens, img_ids_t)
    decoded = ae.decode(latent_grid).float().clamp(-1, 1)
    image_array = rearrange(decoded[0], "c h w -> h w c")
    image = Image.fromarray((127.5 * (image_array + 1.0)).cpu().byte().numpy())
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95, subsampling=0)
    ae_elapsed = time.perf_counter() - ae_start

    result = Klein9BRunResult(
        image_path=str(output_path),
        width=width,
        height=height,
        num_steps=num_steps,
        text_encoder_elapsed=text_elapsed,
        denoise_elapsed=denoise_elapsed,
        ae_elapsed=ae_elapsed,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return result


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--text-encoder-dir", type=Path)
    parser.add_argument("--ae-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("klein9b.png"))
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    args = parser.parse_args()
    main(
        prompt=args.prompt,
        model_dir=args.model_dir,
        text_encoder_dir=args.text_encoder_dir,
        ae_dir=args.ae_dir,
        output=args.output,
        gguf_dir=args.gguf_dir,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    _main_cli()
