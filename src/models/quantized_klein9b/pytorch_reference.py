from __future__ import annotations

import gc
from pathlib import Path
from types import TracebackType
from collections.abc import Callable, Mapping
from typing import Protocol, cast

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

from models.quantized_klein9b.pytorch_modules import (
    DoubleStreamBlock,
    EmbedND,
    Klein9BParams,
    LastLayer,
    MLPEmbedder,
    Modulation,
    SingleStreamBlock,
    timestep_embedding,
)


FluxStageCallback = Callable[
    [str, int | None, Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    None,
]


class _SafeTensorHandle(Protocol):
    def keys(self) -> list[str]: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None: ...


class FluxWeights:
    def __init__(
        self,
        path: Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.path = path
        self.device = device
        self.dtype = dtype
        self._handle: _SafeTensorHandle = safe_open(path, framework="pt", device="cpu")
        self.keys = tuple(self._handle.keys())

    def close(self) -> None:
        self._handle.__exit__(None, None, None)

    def state(self, prefix: str) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for key in self.keys:
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = self._handle.get_tensor(key).to(
                    self.device,
                    dtype=self.dtype,
                )
        return state


class FluxStreamingReference:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.params = Klein9BParams()
        weight_path = Path(model_dir).expanduser().resolve() / "flux-2-klein-9b.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"FLUX weights are missing: {weight_path}")
        self.weights = FluxWeights(weight_path, device=self.device, dtype=self.dtype)

    def close(self) -> None:
        self.weights.close()

    @torch.no_grad()
    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        pred = self.step(
            img=_to_cuda(inputs["x"], device=self.device, dtype=self.dtype),
            img_ids=_to_cuda(inputs["x_ids"], device=self.device),
            timesteps=_to_cuda(inputs["timesteps"], device=self.device, dtype=self.dtype),
            ctx=_to_cuda(inputs["ctx"], device=self.device, dtype=self.dtype),
            ctx_ids=_to_cuda(inputs["ctx_ids"], device=self.device),
        )
        return {"linear_120": pred}

    @torch.no_grad()
    def step(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        stage_callback: FluxStageCallback | None = None,
    ) -> torch.Tensor:
        params = self.params
        hidden = params.hidden_size
        num_txt_tokens = int(ctx.shape[1])

        with torch.device("meta"):
            time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden, disable_bias=True)
            mod_img = Modulation(hidden, double=True, disable_bias=True)
            mod_txt = Modulation(hidden, double=True, disable_bias=True)
            mod_single = Modulation(hidden, double=False, disable_bias=True)
            img_in = nn.Linear(params.in_channels, hidden, bias=False)
            txt_in = nn.Linear(params.context_in_dim, hidden, bias=False)

        time_in = _load_module(time_in, self.weights, "time_in.")
        mod_img = _load_module(mod_img, self.weights, "double_stream_modulation_img.")
        mod_txt = _load_module(mod_txt, self.weights, "double_stream_modulation_txt.")
        mod_single = _load_module(mod_single, self.weights, "single_stream_modulation.")
        img_in = _load_module(img_in, self.weights, "img_in.")
        txt_in = _load_module(txt_in, self.weights, "txt_in.")

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = time_in(timestep_emb)
        double_block_mod_img = mod_img(vec)
        double_block_mod_txt = mod_txt(vec)
        single_block_mod, _ = mod_single(vec)
        img_hidden = img_in(img)
        txt_hidden = txt_in(ctx)

        del time_in, mod_img, mod_txt, mod_single, img_in, txt_in, timestep_emb
        _release_cuda_cache()

        pe_embedder = EmbedND(
            dim=hidden // params.num_heads,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )
        pe_x = pe_embedder(img_ids)
        pe_ctx = pe_embedder(ctx_ids)
        if stage_callback is not None:
            stage_callback(
                "prologue",
                None,
                {
                    "x": img,
                    "x_ids": img_ids,
                    "timesteps": timesteps,
                    "ctx": ctx,
                    "ctx_ids": ctx_ids,
                },
                {
                    "img": img_hidden,
                    "txt": txt_hidden,
                    "pe_x": pe_x,
                    "pe_ctx": pe_ctx,
                    "vec": vec,
                    "img_mod1_shift": double_block_mod_img[0][0],
                    "img_mod1_scale": double_block_mod_img[0][1],
                    "img_mod1_gate": double_block_mod_img[0][2],
                    "img_mod2_shift": double_block_mod_img[1][0],
                    "img_mod2_scale": double_block_mod_img[1][1],
                    "img_mod2_gate": double_block_mod_img[1][2],
                    "txt_mod1_shift": double_block_mod_txt[0][0],
                    "txt_mod1_scale": double_block_mod_txt[0][1],
                    "txt_mod1_gate": double_block_mod_txt[0][2],
                    "txt_mod2_shift": double_block_mod_txt[1][0],
                    "txt_mod2_scale": double_block_mod_txt[1][1],
                    "txt_mod2_gate": double_block_mod_txt[1][2],
                    "single_mod_shift": single_block_mod[0],
                    "single_mod_scale": single_block_mod[1],
                    "single_mod_gate": single_block_mod[2],
                },
            )

        for layer_idx in range(params.depth):
            with torch.device("meta"):
                block = DoubleStreamBlock(hidden, params.num_heads, params.mlp_ratio)
            block = cast(
                DoubleStreamBlock,
                _load_module(block, self.weights, f"double_blocks.{layer_idx}."),
            )
            block_inputs = {
                "img": img_hidden,
                "txt": txt_hidden,
                "pe": pe_x,
                "pe_ctx": pe_ctx,
                "img_mod1_shift": double_block_mod_img[0][0],
                "img_mod1_scale": double_block_mod_img[0][1],
                "img_mod1_gate": double_block_mod_img[0][2],
                "img_mod2_shift": double_block_mod_img[1][0],
                "img_mod2_scale": double_block_mod_img[1][1],
                "img_mod2_gate": double_block_mod_img[1][2],
                "txt_mod1_shift": double_block_mod_txt[0][0],
                "txt_mod1_scale": double_block_mod_txt[0][1],
                "txt_mod1_gate": double_block_mod_txt[0][2],
                "txt_mod2_shift": double_block_mod_txt[1][0],
                "txt_mod2_scale": double_block_mod_txt[1][1],
                "txt_mod2_gate": double_block_mod_txt[1][2],
            }
            block_outputs = block.forward_kv_extract_debug(
                img_hidden,
                txt_hidden,
                pe_x,
                pe_ctx,
                double_block_mod_img,
                double_block_mod_txt,
                num_ref_tokens=0,
            )
            img_hidden = block_outputs["img"]
            txt_hidden = block_outputs["txt"]
            if stage_callback is not None:
                stage_callback(
                    "double_block",
                    layer_idx,
                    block_inputs,
                    block_outputs,
                )
            del block
            _release_cuda_cache()

        img_hidden = torch.cat((txt_hidden, img_hidden), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        del txt_hidden, pe_ctx, pe_x, double_block_mod_img, double_block_mod_txt
        _release_cuda_cache()

        for layer_idx in range(params.depth_single_blocks):
            with torch.device("meta"):
                block = SingleStreamBlock(hidden, params.num_heads, params.mlp_ratio)
            block = cast(
                SingleStreamBlock,
                _load_module(block, self.weights, f"single_blocks.{layer_idx}."),
            )
            block_inputs = {
                "hidden_states": img_hidden,
                "pe": pe,
                "mod_shift": single_block_mod[0],
                "mod_scale": single_block_mod[1],
                "mod_gate": single_block_mod[2],
            }
            block_outputs = block.forward_kv_extract_debug(
                img_hidden,
                pe,
                single_block_mod,
                num_txt_tokens,
                num_ref_tokens=0,
            )
            img_hidden = block_outputs["hidden_states"]
            if stage_callback is not None:
                stage_callback(
                    "single_block",
                    layer_idx,
                    block_inputs,
                    block_outputs,
                )
            del block
            _release_cuda_cache()

        final_inputs = {"hidden_states": img_hidden, "vec": vec}
        img_hidden = img_hidden[:, num_txt_tokens:, ...]
        del pe, single_block_mod
        _release_cuda_cache()

        with torch.device("meta"):
            final_layer = LastLayer(hidden, params.in_channels)
        final_layer = _load_module(final_layer, self.weights, "final_layer.")
        pred = final_layer(img_hidden, vec)
        if stage_callback is not None:
            stage_callback("final_layer", None, final_inputs, {"pred": pred})
        del final_layer, img_hidden, vec
        _release_cuda_cache()
        return pred


def configure_rocm_reference() -> None:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _load_module(module: nn.Module, weights: FluxWeights, prefix: str) -> nn.Module:
    module = module.to_empty(device=weights.device).to(dtype=weights.dtype).eval()
    module.load_state_dict(weights.state(prefix), strict=True, assign=True)
    return module


def _to_cuda(
    value: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(value)).to(device)
    if dtype is not None and tensor.is_floating_point():
        return tensor.to(dtype=dtype)
    return tensor


def _release_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
