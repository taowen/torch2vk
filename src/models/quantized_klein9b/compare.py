"""Compare quantized FLUX.2 Klein 9B Vulkan stages inside a PyTorch ROCm flow."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from models.quantized_klein9b.autoencoder import AutoEncoder, AutoEncoderParams
from models.quantized_klein9b.custom_shaders import KLEIN9B_CAPTURE_QWEN3_CTX_F16
from models.quantized_klein9b.dispatch.ae_decode import run_ae_decode
from models.quantized_klein9b.dispatch.flux_double_block import run_flux_double_block
from models.quantized_klein9b.dispatch.flux_final_layer import run_flux_final_layer
from models.quantized_klein9b.dispatch.flux_prologue import run_flux_prologue
from models.quantized_klein9b.dispatch.flux_single_block import run_flux_single_block
from models.quantized_klein9b.dispatch.text_embed import run_text_embed
from models.quantized_klein9b.dispatch.text_layer import run_text_layer
from models.quantized_klein9b.export_gguf import DEFAULT_OUTPUT_DIR
from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.pytorch_reference import (
    FluxStreamingReference,
    configure_rocm_reference,
)
from models.quantized_klein9b.run import (
    _ensure_ggufs,
    _get_schedule,
    _latent_tokens_and_ids,
    _prepare_input_ids,
    _text_ids,
)
from models.quantized_klein9b.text_encoder import Qwen3TextEncoder
from models.quantized_klein9b.tensors.model import (
    FLUX_DOUBLE_BLOCK_OUTPUTS,
    FLUX_FINAL_LAYER_OUTPUTS,
    FLUX_PROLOGUE_OUTPUTS,
    FLUX_SINGLE_BLOCK_OUTPUTS,
    create_model_tensors,
    image_output,
    model_tensors,
)
from torch2vk.runtime.rope_table import ROPE_TABLE_F32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader


DEFAULT_PROMPT = "a clean watercolor illustration of a small red sailboat on a calm blue sea under morning light"
DEFAULT_PYTORCH_IMAGE = Path(".cache/torch2vk/klein9b_pytorch_streaming.png")
DEFAULT_VULKAN_AE_IMAGE = Path(".cache/torch2vk/klein9b_vulkan_ae.png")
get_shader = make_shader_loader("models.quantized_klein9b.shaders")


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    rms_abs: float


@dataclass(frozen=True, slots=True)
class Klein9BCompareResult:
    width: int
    height: int
    steps: int
    pytorch_image_path: str
    max_abs: float
    mean_abs: float
    rms_abs: float
    text_ctx: ErrorMetrics | None
    flux: ErrorMetrics | None
    ae_decode: ErrorMetrics | None
    vulkan_ae_image_path: str | None


def compare_pytorch_main(
    *,
    prompt: str = DEFAULT_PROMPT,
    model_dir: str | Path | None = None,
    text_encoder_dir: str | Path | None = None,
    ae_dir: str | Path | None = None,
    gguf_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ctx_cache: str | Path | None = None,
    output: str | Path = DEFAULT_PYTORCH_IMAGE,
    vulkan_ae_output: str | Path = DEFAULT_VULKAN_AE_IMAGE,
    width: int = 256,
    height: int = 256,
    seed: int = 7,
    num_steps: int = 4,
    compare_text: bool = True,
    compare_flux: bool = True,
    vulkan_flux_steps: int = 1,
    compare_all_flux_blocks: bool = False,
    compare_ae: bool = False,
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
    _ensure_ggufs(
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
    text_metrics = None
    if compare_text:
        _log("comparing Vulkan text encoder against PyTorch context")
        text_metrics = _compare_text_context(
            prompt=prompt,
            expected_ctx=ctx_np,
            text_encoder_dir=model_dirs.text_encoder,
            gguf_dir=Path(gguf_dir).expanduser().resolve(),
        )
        print(json.dumps({"text_ctx": asdict(text_metrics)}, ensure_ascii=False), flush=True)

    img_np, img_ids_np = _latent_tokens_and_ids(width=width, height=height, seed=seed)
    latent_height = height // 16
    latent_width = width // 16
    timesteps = _get_schedule(num_steps, img_np.shape[1])
    flux_metrics = None
    if compare_flux and vulkan_flux_steps > 0:
        _log("opening Vulkan runtime for streaming FLUX compares")
        create_model_tensors(
            latent_height=latent_height,
            latent_width=latent_width,
            text_seq_len=ctx_np.shape[1],
            include_ae_decode=False,
        )
        rt = RuntimeSession.open(
            device_index=0,
            model_dir=Path(gguf_dir).expanduser().resolve(),
            model_tensors=model_tensors(),
            get_shader=get_shader,
        )
    else:
        rt = None

    flux_comparer = (
        _FluxStageComparer(rt, compare_all_blocks=compare_all_flux_blocks)
        if rt is not None
        else None
    )
    ref = FluxStreamingReference(model_dirs.flux)
    try:
        for step, (t_curr, t_prev) in enumerate(
            zip(timesteps[:-1], timesteps[1:], strict=True),
            start=1,
        ):
            _log(f"step {step}: running PyTorch streaming FLUX")
            pred = ref.step(
                img=_torch_floating(img_np),
                img_ids=torch.from_numpy(img_ids_np).cuda(),
                timesteps=torch.tensor([t_curr], device="cuda", dtype=torch.bfloat16),
                ctx=_torch_floating(ctx_np),
                ctx_ids=torch.from_numpy(ctx_ids_np).cuda(),
                stage_callback=flux_comparer.callback(step)
                if flux_comparer is not None and step <= vulkan_flux_steps
                else None,
            )
            pred_np = pred.detach().cpu().float().numpy().astype(np.float32)
            del pred
            _release_torch()
            img_np = (img_np.astype(np.float32) + (t_prev - t_curr) * pred_np).astype(np.float16)
    finally:
        ref.close()
        if rt is not None:
            rt.close()

    if flux_comparer is not None:
        flux_metrics = flux_comparer.metrics()

    _log("decoding PyTorch image")
    pytorch_image = _run_pytorch_ae_decode(
        model_dirs.ae,
        latent=img_np,
        width=width,
        height=height,
    )
    pytorch_image_path = _save_image(output, pytorch_image)
    ae_metrics = None
    vulkan_ae_path = None
    if compare_ae:
        _log("injecting Vulkan AE decode for local compare")
        create_model_tensors(
            latent_height=latent_height,
            latent_width=latent_width,
            text_seq_len=ctx_np.shape[1],
        )
        vulkan_image = _run_vulkan_ae_decode(
            gguf_dir=Path(gguf_dir).expanduser().resolve(),
            latent=img_np,
        )
        ae_metrics = _error_metrics(vulkan_image, pytorch_image)
        vulkan_ae_path = _save_image(vulkan_ae_output, vulkan_image)
        print(json.dumps({"ae_decode": asdict(ae_metrics)}, ensure_ascii=False), flush=True)

    max_abs = 0.0
    mean_abs = 0.0
    rms_abs = 0.0
    for metrics in (text_metrics, flux_metrics, ae_metrics):
        if metrics is None:
            continue
        max_abs = max(max_abs, metrics.max_abs)
        mean_abs = metrics.mean_abs
        rms_abs = metrics.rms_abs

    result = Klein9BCompareResult(
        width=width,
        height=height,
        steps=num_steps,
        pytorch_image_path=str(pytorch_image_path),
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms_abs=rms_abs,
        text_ctx=text_metrics,
        flux=flux_metrics,
        ae_decode=ae_metrics,
        vulkan_ae_image_path=None if vulkan_ae_path is None else str(vulkan_ae_path),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2), flush=True)
    return result


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
    num_steps: int = 4,
) -> Klein9BCompareResult:
    return compare_pytorch_main(
        prompt=prompt,
        model_dir=model_dir,
        text_encoder_dir=text_encoder_dir,
        ae_dir=ae_dir,
        gguf_dir=gguf_dir,
        ctx_cache=ctx_cache,
        width=width,
        height=height,
        seed=seed,
        num_steps=num_steps,
        vulkan_flux_steps=1,
        compare_text=False,
        compare_flux=True,
        compare_ae=False,
    )


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
    ctx_np = ctx.detach().cpu().float().numpy().astype(np.float16)
    ctx_ids_np = _text_ids()
    if ctx_cache is not None:
        ctx_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ctx_cache, ctx=ctx_np, ctx_ids=ctx_ids_np)
    del text_encoder, ctx
    torch.cuda.empty_cache()
    return ctx_np, ctx_ids_np


def _compare_text_context(
    *,
    prompt: str,
    expected_ctx: np.ndarray,
    text_encoder_dir: Path,
    gguf_dir: Path,
) -> ErrorMetrics:
    input_ids = _prepare_input_ids(prompt=prompt, text_encoder_dir=text_encoder_dir)
    create_model_tensors(
        text_seq_len=expected_ctx.shape[1],
        include_ae_decode=False,
    )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    try:
        tensors = model_tensors()
        with rt.request(inputs={_tensor_name(tensors.input_ids.name): input_ids}):
            with rt.frame("klein9b.text_encoder.compare"):
                ROPE_TABLE_F32(
                    rt,
                    start_position=0,
                    theta=1_000_000.0,
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
            actual = rt.read_request_state(tensors.ctx).astype(np.float32)
    finally:
        rt.close()
    return _error_metrics(actual, expected_ctx)


class _FluxStageComparer:
    def __init__(self, rt: RuntimeSession, *, compare_all_blocks: bool) -> None:
        self.rt = rt
        self.compare_all_blocks = compare_all_blocks
        self._max_abs = 0.0
        self._mean_abs = 0.0
        self._rms_abs = 0.0
        self._count = 0

    def callback(self, step: int):
        def _callback(
            stage: str,
            layer_idx: int | None,
            inputs: Mapping[str, torch.Tensor],
            expected: Mapping[str, torch.Tensor],
        ) -> None:
            self.compare(step, stage=stage, layer_idx=layer_idx, inputs=inputs, expected=expected)

        return _callback

    def metrics(self) -> ErrorMetrics | None:
        if self._count == 0:
            return None
        return ErrorMetrics(
            max_abs=self._max_abs,
            mean_abs=self._mean_abs,
            rms_abs=self._rms_abs,
        )

    def compare(
        self,
        step: int,
        *,
        stage: str,
        layer_idx: int | None,
        inputs: Mapping[str, torch.Tensor],
        expected: Mapping[str, torch.Tensor],
    ) -> None:
        if not self._should_compare(stage, layer_idx):
            return
        tensors = model_tensors()
        if stage == "prologue":
            tensor_group = tensors.flux_prologue
            output_bindings = FLUX_PROLOGUE_OUTPUTS
            request_inputs = self._host_inputs(
                tensor_group,
                inputs,
                exclude=frozenset(("x", "ctx")),
            )
            request_state = {
                tensor_group.x: _torch_to_numpy(inputs["x"]),
                tensor_group.ctx: _torch_to_numpy(inputs["ctx"]),
            }
            release_group = tensor_group
        elif stage == "double_block":
            if layer_idx is None:
                raise RuntimeError("double_block compare requires layer_idx")
            tensor_group = tensors.flux_double_blocks[layer_idx]
            output_bindings = FLUX_DOUBLE_BLOCK_OUTPUTS
            request_inputs = self._host_inputs(tensor_group, inputs)
            request_state = {}
            release_group = tensor_group
        elif stage == "single_block":
            if layer_idx is None:
                raise RuntimeError("single_block compare requires layer_idx")
            tensor_group = tensors.flux_single_blocks[layer_idx]
            output_bindings = FLUX_SINGLE_BLOCK_OUTPUTS
            request_inputs = self._host_inputs(tensor_group, inputs)
            request_state = {}
            release_group = tensor_group
        elif stage == "final_layer":
            tensor_group = tensors.flux_final_layer
            output_bindings = FLUX_FINAL_LAYER_OUTPUTS
            request_inputs = self._host_inputs(tensor_group, inputs)
            request_state = {}
            release_group = tensor_group
        else:
            return

        actual: dict[str, np.ndarray] = {}
        with self.rt.request(inputs=request_inputs, state=request_state):
            with self.rt.frame(self._frame_name(step, stage, layer_idx)):
                self._run_stage(stage, layer_idx)
                for semantic_name, field_name in output_bindings.items():
                    actual[semantic_name] = self.rt.readback(
                        getattr(tensor_group, field_name)
                    ).astype(np.float32)
        self.rt.release_model_weights(release_group)
        for semantic_name, candidate in actual.items():
            metric = _error_metrics(candidate, expected[semantic_name])
            self._record(metric)
            print(
                json.dumps(
                    {
                        "flux_step": step,
                        "stage": stage,
                        "layer_idx": layer_idx,
                        "output": semantic_name,
                        **asdict(metric),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    def _should_compare(self, stage: str, layer_idx: int | None) -> bool:
        if stage in ("prologue", "final_layer"):
            return True
        if self.compare_all_blocks:
            return True
        return layer_idx == 0

    def _run_stage(self, stage: str, layer_idx: int | None) -> None:
        if stage == "prologue":
            run_flux_prologue(self.rt)
        elif stage == "double_block":
            if layer_idx is None:
                raise RuntimeError("double_block compare requires layer_idx")
            run_flux_double_block(self.rt, layer_idx)
        elif stage == "single_block":
            if layer_idx is None:
                raise RuntimeError("single_block compare requires layer_idx")
            run_flux_single_block(self.rt, layer_idx)
        elif stage == "final_layer":
            run_flux_final_layer(self.rt)
        else:
            raise RuntimeError(f"unknown FLUX compare stage: {stage}")

    def _record(self, metric: ErrorMetrics) -> None:
        self._count += 1
        self._max_abs = _worse_metric(self._max_abs, metric.max_abs)
        self._mean_abs = _worse_metric(self._mean_abs, metric.mean_abs)
        self._rms_abs = _worse_metric(self._rms_abs, metric.rms_abs)

    def _host_inputs(
        self,
        tensor_group: object,
        inputs: Mapping[str, torch.Tensor],
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> dict[str, np.ndarray]:
        return {
            _tensor_name(getattr(tensor_group, name).name): _torch_to_numpy(value)
            for name, value in inputs.items()
            if name not in exclude
        }

    def _frame_name(self, step: int, stage: str, layer_idx: int | None) -> str:
        if layer_idx is None:
            return f"klein9b.flux.compare.{step:04d}.{stage}"
        return f"klein9b.flux.compare.{step:04d}.{stage}.{layer_idx}"


def _run_pytorch_ae_decode(
    ae_dir: Path,
    *,
    latent: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    with safe_open(ae_dir / "ae.safetensors", framework="pt", device="cpu") as handle:
        state: dict[str, torch.Tensor] = {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.cuda().to(torch.bfloat16)
            else:
                tensor = tensor.cuda()
            state[key] = tensor
    ae = AutoEncoder(AutoEncoderParams()).eval().cuda().to(torch.bfloat16)
    ae.load_state_dict(state, strict=True, assign=True)
    with torch.no_grad():
        tokens = _torch_floating(latent)
        z = tokens.reshape(1, height // 16, width // 16, 128).permute(0, 3, 1, 2).contiguous()
        image = ae.decode(z).detach().cpu().float().numpy()
    del ae, state, tokens, z
    _release_torch()
    return image


def _run_vulkan_ae_decode(
    *,
    gguf_dir: Path,
    latent: np.ndarray,
) -> np.ndarray:
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    try:
        with rt.request(state={model_tensors().latent_tokens: latent}):
            with rt.frame("klein9b.ae_decode.compare"):
                run_ae_decode(rt)
            return rt.read_request_state(image_output()).astype(np.float32)
    finally:
        rt.close()


def _save_image(path: str | Path, image: np.ndarray) -> Path:
    image_array = np.clip(image[0].transpose(1, 2, 0), -1.0, 1.0)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((127.5 * (image_array + 1.0)).astype(np.uint8)).save(
        output,
        quality=95,
        subsampling=0,
    )
    return output


def _torch_floating(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).cuda().to(torch.bfloat16)


def _torch_to_numpy(value: torch.Tensor) -> np.ndarray:
    cpu = value.detach().cpu()
    if cpu.is_floating_point():
        return np.ascontiguousarray(cpu.float().numpy().astype(np.float16))
    return np.ascontiguousarray(cpu.numpy())


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _error_metrics(candidate: np.ndarray, expected: object) -> ErrorMetrics:
    expected_np = np.asarray(
        expected.detach().cpu().float().numpy() if isinstance(expected, torch.Tensor) else expected,
        dtype=np.float32,
    )
    diff = candidate.astype(np.float32) - expected_np
    return ErrorMetrics(
        max_abs=float(np.max(np.abs(diff))),
        mean_abs=float(np.mean(np.abs(diff))),
        rms_abs=float(np.sqrt(np.mean(diff * diff))),
    )


def _worse_metric(current: float, candidate: float) -> float:
    if math.isnan(candidate):
        return candidate
    if math.isnan(current):
        return current
    return max(current, candidate)


def _release_torch() -> None:
    gc.collect()
    torch.cuda.empty_cache()


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
    parser.add_argument("--output", type=Path, default=DEFAULT_PYTORCH_IMAGE)
    parser.add_argument("--vulkan-ae-output", type=Path, default=DEFAULT_VULKAN_AE_IMAGE)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--vulkan-flux-steps", type=int, default=1)
    parser.add_argument("--compare-all-flux-blocks", action="store_true")
    parser.add_argument("--skip-text-compare", action="store_true")
    parser.add_argument("--skip-flux-compare", action="store_true")
    parser.add_argument("--compare-ae", action="store_true")
    args = parser.parse_args()
    compare_pytorch_main(
        prompt=args.prompt,
        model_dir=args.model_dir,
        text_encoder_dir=args.text_encoder_dir,
        ae_dir=args.ae_dir,
        gguf_dir=args.gguf_dir,
        ctx_cache=args.ctx_cache,
        output=args.output,
        vulkan_ae_output=args.vulkan_ae_output,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_steps=args.num_steps,
        compare_text=not args.skip_text_compare,
        compare_flux=not args.skip_flux_compare,
        vulkan_flux_steps=args.vulkan_flux_steps,
        compare_all_flux_blocks=args.compare_all_flux_blocks,
        compare_ae=args.compare_ae,
    )


if __name__ == "__main__":
    _main_cli()
