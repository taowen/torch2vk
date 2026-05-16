"""Run the FLUX.2 Klein 9B text-to-image pipeline on Vulkan."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from models.quantized_klein9b.custom_shaders import (
    KLEIN9B_CAPTURE_QWEN3_CTX_F16,
    KLEIN9B_EULER_UPDATE_F16,
)
from models.quantized_klein9b.dispatch.ae_decode import run_ae_decode
from models.quantized_klein9b.dispatch.flux import run_flux
from models.quantized_klein9b.dispatch.text_embed import run_text_embed
from models.quantized_klein9b.dispatch.text_layer import run_text_layer
from models.quantized_klein9b.export import DEFAULT_IMAGE_SEQ_LEN, DEFAULT_TEXT_SEQ_LEN
from models.quantized_klein9b.export_gguf import (
    DEFAULT_OUTPUT_DIR,
    Klein9BGGUFPaths,
    export_klein9b_q4_k_m_ggufs,
)
from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.tensors.model import (
    create_model_tensors,
    image_output,
    model_tensors,
)
from torch2vk.runtime.rope_table import ROPE_TABLE_F32
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
    timesteps = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
    shifted = np.zeros_like(timesteps)
    nonzero = timesteps > 0.0
    shifted[nonzero] = math.exp(mu) / (math.exp(mu) + (1.0 / timesteps[nonzero] - 1.0))
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


def _prepare_input_ids(*, prompt: str, text_encoder_dir: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=DEFAULT_TEXT_SEQ_LEN,
    )
    return np.ascontiguousarray(model_inputs["input_ids"], dtype=np.int64)


def _text_ids() -> np.ndarray:
    ids = np.zeros((1, DEFAULT_TEXT_SEQ_LEN, 4), dtype=np.int64)
    ids[0, :, 3] = np.arange(DEFAULT_TEXT_SEQ_LEN, dtype=np.int64)
    return ids


def _latent_tokens_and_ids(*, width: int, height: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    latent_h = height // 16
    latent_w = width // 16
    rng = np.random.default_rng(seed)
    tokens = rng.standard_normal((1, latent_h * latent_w, 128)).astype(np.float16)
    ids = np.zeros((1, latent_h * latent_w, 4), dtype=np.int64)
    offset = 0
    for h in range(latent_h):
        for w in range(latent_w):
            ids[0, offset, 1] = h
            ids[0, offset, 2] = w
            offset += 1
    return np.ascontiguousarray(tokens), ids


def _tensor_name(name: str | None) -> str:
    if name is None:
        raise RuntimeError("Klein9B input tensors must be named before request execution")
    return name


def _run_text_encoder(rt: RuntimeSession, *, rope_theta: float) -> float:
    tensors = model_tensors()
    start = time.perf_counter()
    with rt.frame("klein9b.text_encoder"):
        ROPE_TABLE_F32(
            rt,
            start_position=0,
            theta=rope_theta,
            cos=tensors.text_rope.cos,
            sin=tensors.text_rope.sin,
        )
        run_text_embed(rt)
        for layer_idx in range(len(tensors.text_layers)):
            run_text_layer(rt, layer_idx)
        KLEIN9B_CAPTURE_QWEN3_CTX_F16(
            rt,
            layer_9=tensors.text_layers[8].add_7,
            layer_18=tensors.text_layers[17].add_7,
            layer_27=tensors.text_layers[26].add_7,
            ctx=tensors.ctx,
        )
    return time.perf_counter() - start


def _run_denoise(rt: RuntimeSession, *, timesteps: list[float]) -> float:
    tensors = model_tensors()
    start = time.perf_counter()
    for step, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
        rt.register_host_inputs(
            {tensors.flux.timesteps: np.array([t_curr], dtype=np.float16)}
        )
        with rt.frame(f"klein9b.flux.{step:04d}"):
            run_flux(rt)
            KLEIN9B_EULER_UPDATE_F16(
                rt,
                x=tensors.latent_tokens,
                pred=tensors.flux.linear_120,
                dt=float(t_prev - t_curr),
            )
    return time.perf_counter() - start


def _run_ae_decode(rt: RuntimeSession) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    with rt.frame("klein9b.ae_decode"):
        run_ae_decode(rt)
    image = rt.read_request_state(image_output()).astype(np.float32)
    return time.perf_counter() - start, image


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
    if width != 512 or height != 512:
        raise ValueError("quantized_klein9b Vulkan AE decode currently supports 512x512 output")
    if num_steps != 4:
        raise ValueError(f"FLUX.2 Klein 9B is distilled for 4 steps, got {num_steps}")

    model_dirs = resolve_model_dirs(
        model_dir=model_dir,
        text_encoder_dir=text_encoder_dir,
        ae_dir=ae_dir,
    )
    output_dir = Path(gguf_dir).expanduser().resolve()
    _ensure_ggufs(
        model_dir=model_dirs.flux,
        text_encoder_dir=model_dirs.text_encoder,
        ae_dir=model_dirs.ae,
        output_dir=output_dir,
    )

    input_ids = _prepare_input_ids(prompt=prompt, text_encoder_dir=model_dirs.text_encoder)
    text_ids = _text_ids()
    latent_tokens, latent_ids = _latent_tokens_and_ids(width=width, height=height, seed=seed)
    create_model_tensors(image_seq_len=DEFAULT_IMAGE_SEQ_LEN, text_seq_len=DEFAULT_TEXT_SEQ_LEN)
    timesteps = _get_schedule(num_steps, DEFAULT_IMAGE_SEQ_LEN)

    rt = RuntimeSession.open(
        device_index=0,
        model_dir=output_dir,
        profile_dir=profile_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    try:
        tensors = model_tensors()
        request_inputs = {
            _tensor_name(tensors.input_ids.name): input_ids,
            _tensor_name(tensors.flux.x_ids.name): latent_ids,
            _tensor_name(tensors.flux.ctx_ids.name): text_ids,
        }
        request_state = {tensors.latent_tokens: latent_tokens}
        with rt.request(inputs=request_inputs, state=request_state):
            text_elapsed = _run_text_encoder(rt, rope_theta=1_000_000.0)
            denoise_elapsed = _run_denoise(rt, timesteps=timesteps)
            ae_elapsed, image_array = _run_ae_decode(rt)
    finally:
        rt.close()

    image_array = np.clip(image_array[0].transpose(1, 2, 0), -1.0, 1.0)
    image = Image.fromarray((127.5 * (image_array + 1.0)).astype(np.uint8))
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95, subsampling=0)

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
