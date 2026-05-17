"""Export quantized FLUX.2 Klein 9B denoiser Vulkan modules.

Run from project root:
    uv run python -m models.quantized_klein9b.export
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import torch
from einops import rearrange
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from models.quantized_klein9b.autoencoder import AutoEncoder, AutoEncoderParams
from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.pytorch_modules import (
    DoubleStreamBlock,
    Flux2,
    Klein9BParams,
    SingleStreamBlock,
    timestep_embedding,
)
from models.quantized_klein9b.quantization import (
    ae_q4_k_m_config,
    klein9b_q4_k_m_config,
    qwen3_text_encoder_q8_config,
)
from torch2vk.export import (
    TensorSpecOverride,
    bind_dispatch_function_to_tensors,
    cast_floating_tensors,
    clear_python_modules,
    clear_shader_package,
    count_python_modules,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
    module_floating_dtype,
    rename_shader_variant,
    render_model_dispatch_module,
    write_shader_file,
)
from torch2vk.export.reference_codegen import (
    render_reference_module,
    render_streaming_compare_function,
)
from torch2vk.export.registry import DEFAULT_REGISTRY
from torch2vk.export.graph import graph_output_names
from torch2vk.export.tensor_codegen import layer_workspace_keep_fields, render_tensor_module
from torch2vk.runtime.shader import ShaderVariant


MODEL_PACKAGE = "models.quantized_klein9b"
DEFAULT_IMAGE_SEQ_LEN = 1024
DEFAULT_TEXT_SEQ_LEN = 512
DEFAULT_LATENT_HEIGHT = 32
DEFAULT_LATENT_WIDTH = 32


class AEDecodeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ae = AutoEncoder(AutoEncoderParams())
        self.decoder = ae.decoder
        self.bn = ae.bn
        self.bn_eps = ae.bn_eps
        self.ps = ae.ps

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = tokens
        running_var = self.bn.running_var
        running_mean = self.bn.running_mean
        if running_var is None or running_mean is None:
            raise RuntimeError("AutoEncoder BatchNorm running stats are required for decode")
        scale = torch.sqrt(running_var.view(1, -1, 1, 1) + self.bn_eps)
        mean = running_mean.view(1, -1, 1, 1)
        z = z * scale + mean
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)


class AEEntryModule(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.to(torch.float16).permute(0, 3, 1, 2).contiguous()


class TextContextCaptureModule(nn.Module):
    def forward(
        self,
        layer_9: torch.Tensor,
        layer_18: torch.Tensor,
        layer_27: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((layer_9.float(), layer_18.float(), layer_27.float()), dim=-1)


class FluxJoinModule(nn.Module):
    def forward(self, txt: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return torch.cat((txt, img), dim=1)


class FluxPeJoinModule(nn.Module):
    def forward(self, pe_ctx: torch.Tensor, pe_x: torch.Tensor) -> torch.Tensor:
        return torch.cat((pe_ctx, pe_x), dim=2)


class EulerUpdateModule(nn.Module):
    def forward(self, x: torch.Tensor, pred: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return x + dt.view(1, 1, 1) * pred


class FluxPrologueModule(nn.Module):
    def __init__(self, flux: Flux2) -> None:
        super().__init__()
        self.img_in = flux.img_in
        self.time_in = flux.time_in
        self.txt_in = flux.txt_in
        self.double_stream_modulation_img = flux.double_stream_modulation_img
        self.double_stream_modulation_txt = flux.double_stream_modulation_txt
        self.single_stream_modulation = flux.single_stream_modulation
        self.pe_embedder = flux.pe_embedder

    def forward(
        self,
        x: torch.Tensor,
        x_ids: torch.Tensor,
        timesteps: torch.Tensor,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        vec = self.time_in(timestep_embedding(timesteps, 256))
        img_mod1, img_mod2 = self.double_stream_modulation_img(vec)
        txt_mod1, txt_mod2 = self.double_stream_modulation_txt(vec)
        single_mod, _ = self.single_stream_modulation(vec)
        img = self.img_in(x)
        txt = self.txt_in(ctx)
        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)
        return (
            img,
            txt,
            pe_x,
            pe_ctx,
            vec,
            *img_mod1,
            *img_mod2,
            *txt_mod1,
            *txt_mod2,
            *single_mod,
        )


class FluxDoubleBlockModule(DoubleStreamBlock):
    def __init__(self, block: DoubleStreamBlock) -> None:
        nn.Module.__init__(self)
        self.num_heads = block.num_heads
        self.hidden_size = block.hidden_size
        self.mlp_mult_factor = block.mlp_mult_factor
        self.img_norm1 = block.img_norm1
        self.img_attn = block.img_attn
        self.img_norm2 = block.img_norm2
        self.img_mlp = block.img_mlp
        self.txt_norm1 = block.txt_norm1
        self.txt_attn = block.txt_attn
        self.txt_norm2 = block.txt_norm2
        self.txt_mlp = block.txt_mlp

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        pe: torch.Tensor,
        pe_ctx: torch.Tensor,
        img_mod1_shift: torch.Tensor,
        img_mod1_scale: torch.Tensor,
        img_mod1_gate: torch.Tensor,
        img_mod2_shift: torch.Tensor,
        img_mod2_scale: torch.Tensor,
        img_mod2_gate: torch.Tensor,
        txt_mod1_shift: torch.Tensor,
        txt_mod1_scale: torch.Tensor,
        txt_mod1_gate: torch.Tensor,
        txt_mod2_shift: torch.Tensor,
        txt_mod2_scale: torch.Tensor,
        txt_mod2_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        img_mod = (
            (img_mod1_shift, img_mod1_scale, img_mod1_gate),
            (img_mod2_shift, img_mod2_scale, img_mod2_gate),
        )
        txt_mod = (
            (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate),
            (txt_mod2_shift, txt_mod2_scale, txt_mod2_gate),
        )
        outputs = self.forward_kv_extract_debug(
            img,
            txt,
            pe,
            pe_ctx,
            img_mod,
            txt_mod,
            num_ref_tokens=0,
        )
        return (
            outputs["img_k_raw"],
            outputs["txt_k_raw"],
            outputs["img_k_rrms"],
            outputs["txt_k_rrms"],
            outputs["txt_q_unit"],
            outputs["img_q_unit"],
            outputs["txt_k_unit"],
            outputs["img_k_unit"],
            outputs["txt_q"],
            outputs["img_q"],
            outputs["txt_k"],
            outputs["img_k"],
            outputs["txt_v"],
            outputs["img_v"],
            outputs["q"],
            outputs["k"],
            outputs["v"],
            outputs["pe_full"],
            outputs["q_rope"],
            outputs["k_rope"],
            outputs["attn"],
            outputs["txt_attn"],
            outputs["img_attn"],
            outputs["img_attn_proj"],
            outputs["img_after_attn"],
            outputs["img_mlp_in"],
            outputs["img_mlp_hidden"],
            outputs["img_mlp_act"],
            outputs["img_mlp_out"],
            outputs["img"],
            outputs["txt_attn_proj"],
            outputs["txt_after_attn"],
            outputs["txt_mlp_in"],
            outputs["txt_mlp_hidden"],
            outputs["txt_mlp_act"],
            outputs["txt_mlp_out"],
            outputs["txt_mlp_gated"],
            outputs["txt"],
        )


class FluxSingleBlockModule(SingleStreamBlock):
    def __init__(self, block: SingleStreamBlock, *, text_seq_len: int) -> None:
        nn.Module.__init__(self)
        self.hidden_dim = block.hidden_dim
        self.num_heads = block.num_heads
        self.scale = block.scale
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.mlp_mult_factor = block.mlp_mult_factor
        self.linear1 = block.linear1
        self.linear2 = block.linear2
        self.norm = block.norm
        self.hidden_size = block.hidden_size
        self.pre_norm = block.pre_norm
        self.mlp_act = block.mlp_act
        self.text_seq_len = text_seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe: torch.Tensor,
        mod_shift: torch.Tensor,
        mod_scale: torch.Tensor,
        mod_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.forward_kv_extract_debug(
            hidden_states,
            pe,
            (mod_shift, mod_scale, mod_gate),
            self.text_seq_len,
            num_ref_tokens=0,
        )
        return (
            outputs["pre_norm"],
            outputs["x_mod"],
            outputs["linear1"],
            outputs["q_raw"],
            outputs["k_raw"],
            outputs["v"],
            outputs["q_unit"],
            outputs["k_unit"],
            outputs["q"],
            outputs["k"],
            outputs["q_rope"],
            outputs["k_rope"],
            outputs["attn"],
            outputs["mlp"],
            outputs["mlp_act"],
            outputs["out_input"],
            outputs["linear2"],
            outputs["gated"],
            outputs["hidden_states"],
        )


class FluxFinalLayerModule(nn.Module):
    def __init__(self, flux: Flux2, *, text_seq_len: int) -> None:
        super().__init__()
        self.final_layer = flux.final_layer
        self.text_seq_len = text_seq_len

    def forward(self, hidden_states: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.final_layer(hidden_states[:, self.text_seq_len :, ...], vec)


def _write_model_tensors_module(
    path: Path,
    *,
    text_seq_len: int,
    num_text_layers: int,
    text_hidden_size: int,
    text_head_dim: int,
    flux_prologue_outputs: dict[str, str],
    flux_double_block_outputs: dict[str, str],
    flux_single_block_outputs: dict[str, str],
    flux_final_layer_outputs: dict[str, str],
) -> None:
    path.write_text(
        f'''"""Generated model-level tensor wiring for FLUX.2 Klein 9B."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_klein9b.tensors.ae_decode import (
    AE_DECODE_OUTPUT,
    AeDecodeTensors,
    create_ae_decode,
)
from models.quantized_klein9b.tensors.ae_entry import AE_ENTRY_OUTPUT, AEEntryTensors, create_ae_entry
from models.quantized_klein9b.tensors.embed_tokens import EmbedTokensTensors, create_embed_tokens
from models.quantized_klein9b.tensors.euler_update import (
    EULER_UPDATE_OUTPUT,
    EulerUpdateTensors,
    create_euler_update,
)
from models.quantized_klein9b.tensors.flux import FLUX_OUTPUT, FluxTensors, create_flux
from models.quantized_klein9b.tensors.flux_double_block import (
    FluxDoubleBlockTensors,
    create_flux_double_block,
)
from models.quantized_klein9b.tensors.flux_final_layer import (
    FluxFinalLayerTensors,
    create_flux_final_layer,
)
from models.quantized_klein9b.tensors.flux_prologue import (
    FluxPrologueTensors,
    create_flux_prologue,
)
from models.quantized_klein9b.tensors.flux_single_block import (
    FluxSingleBlockTensors,
    create_flux_single_block,
)
from models.quantized_klein9b.tensors.flux_join import FLUX_JOIN_OUTPUT, FluxJoinTensors, create_flux_join
from models.quantized_klein9b.tensors.flux_pe_join import (
    FLUX_PE_JOIN_OUTPUT,
    FluxPeJoinTensors,
    create_flux_pe_join,
)
from models.quantized_klein9b.tensors.rope import RopeTableTensors, create_rope_table
from models.quantized_klein9b.tensors.text_context_capture import (
    TEXT_CONTEXT_CAPTURE_OUTPUT,
    TextContextCaptureTensors,
    create_text_context_capture,
)
from models.quantized_klein9b.tensors.text_layer import TextLayerTensors, create_text_layer
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.logical import (
    bind_logical_tensor_alias,
    bind_logical_tensor_names,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class QuantizedKlein9BTensors:
    input_ids: LogicalTensor
    text_rope: RopeTableTensors
    text_embed: EmbedTokensTensors
    text_layers: tuple[TextLayerTensors, ...]
    text_context_capture: TextContextCaptureTensors
    ctx: LogicalTensor
    latent_tokens: LogicalTensor
    ae_entry: AEEntryTensors | None
    flux_hidden_states: LogicalTensor
    flux_pe: LogicalTensor
    flux_join: FluxJoinTensors
    flux_pe_join: FluxPeJoinTensors
    flux: FluxTensors
    flux_prologue: FluxPrologueTensors
    flux_double_blocks: tuple[FluxDoubleBlockTensors, ...]
    flux_single_blocks: tuple[FluxSingleBlockTensors, ...]
    flux_final_layer: FluxFinalLayerTensors
    euler_update: EulerUpdateTensors
    ae_decode: AeDecodeTensors | None


_MODEL_TENSORS: QuantizedKlein9BTensors | None = None
FLUX_PROLOGUE_OUTPUTS: dict[str, str] = {flux_prologue_outputs!r}
FLUX_DOUBLE_BLOCK_OUTPUTS: dict[str, str] = {flux_double_block_outputs!r}
FLUX_SINGLE_BLOCK_OUTPUTS: dict[str, str] = {flux_single_block_outputs!r}
FLUX_FINAL_LAYER_OUTPUTS: dict[str, str] = {flux_final_layer_outputs!r}


def create_model_tensors(
    *,
    latent_height: int = {DEFAULT_LATENT_HEIGHT},
    latent_width: int = {DEFAULT_LATENT_WIDTH},
    text_seq_len: int = {text_seq_len},
    num_text_layers: int = {num_text_layers},
    text_hidden_size: int = {text_hidden_size},
    text_head_dim: int = {text_head_dim},
    include_ae_decode: bool = True,
) -> QuantizedKlein9BTensors:
    global _MODEL_TENSORS
    image_seq_len = latent_height * latent_width
    input_ids = _host_input_tensor("int64", (1, text_seq_len))
    text_rope = create_rope_table(
        "klein9b.text.rope",
        batch=1,
        sequence_length=text_seq_len,
        head_dim=text_head_dim,
    )
    text_embed = create_embed_tokens(
        "klein9b.text.embed",
        sequence_length=text_seq_len,
        input=input_ids,
    )
    text_layers_list: list[TextLayerTensors] = []
    text_hidden = text_embed.embedding
    for layer_idx in range(num_text_layers):
        layer_tensors = create_text_layer(
            f"klein9b.text.layer.{{layer_idx}}",
            layer_idx=layer_idx,
            sequence_length=text_seq_len,
            hidden_states=text_hidden,
            position_embeddings_0=text_rope.cos,
            position_embeddings_1=text_rope.sin,
        )
        text_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_layers = tuple(text_layers_list)
    ctx = _request_state_tensor("float32", (1, text_seq_len, text_hidden_size * 3))
    text_context_capture = create_text_context_capture(
        "klein9b.text.ctx",
        sequence_length=text_seq_len,
        layer_9=text_layers[8].add_7,
        layer_18=text_layers[17].add_7,
        layer_27=text_layers[26].add_7,
        cat=ctx,
        request_state_outputs=frozenset((TEXT_CONTEXT_CAPTURE_OUTPUT,)),
    )
    latent_tokens = _request_state_tensor("float32", (1, image_seq_len, 128))
    flux_hidden_states = _frame_workspace_tensor(
        "float32",
        (1, text_seq_len + image_seq_len, text_hidden_size),
    )
    flux_pe = _frame_workspace_tensor(
        "float32",
        (1, 1, text_seq_len + image_seq_len, 64, 2, 2),
    )
    flux = create_flux(
        "klein9b.flux",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        x=latent_tokens,
        ctx=ctx,
        request_state_outputs=frozenset((FLUX_OUTPUT,)),
    )
    flux_prologue = create_flux_prologue(
        "klein9b.flux.prologue",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        x=latent_tokens,
        ctx=ctx,
    )
    flux_img = getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img"])
    flux_txt = getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt"])
    flux_pe_x = getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["pe_x"])
    flux_pe_ctx = getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["pe_ctx"])
    flux_vec = getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["vec"])
    flux_double_blocks_list: list[FluxDoubleBlockTensors] = []
    for layer_idx in range(8):
        layer_tensors = create_flux_double_block(
            f"klein9b.flux.double_block.{{layer_idx}}",
            layer_idx=layer_idx,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            img=flux_img,
            txt=flux_txt,
            pe=flux_pe_x,
            pe_ctx=flux_pe_ctx,
            img_mod1_shift=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img_mod1_shift"]),
            img_mod1_scale=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img_mod1_scale"]),
            img_mod1_gate=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img_mod1_gate"]),
            img_mod2_shift=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img_mod2_shift"]),
            img_mod2_scale=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img_mod2_scale"]),
            img_mod2_gate=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["img_mod2_gate"]),
            txt_mod1_shift=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt_mod1_shift"]),
            txt_mod1_scale=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt_mod1_scale"]),
            txt_mod1_gate=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt_mod1_gate"]),
            txt_mod2_shift=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt_mod2_shift"]),
            txt_mod2_scale=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt_mod2_scale"]),
            txt_mod2_gate=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["txt_mod2_gate"]),
        )
        flux_double_blocks_list.append(layer_tensors)
        flux_img = getattr(layer_tensors, FLUX_DOUBLE_BLOCK_OUTPUTS["img"])
        flux_txt = getattr(layer_tensors, FLUX_DOUBLE_BLOCK_OUTPUTS["txt"])
    flux_double_blocks = tuple(flux_double_blocks_list)
    final_double = flux_double_blocks[-1]
    flux_join = create_flux_join(
        "klein9b.flux.join",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        txt=getattr(final_double, FLUX_DOUBLE_BLOCK_OUTPUTS["txt"]),
        img=getattr(final_double, FLUX_DOUBLE_BLOCK_OUTPUTS["img"]),
        cat=flux_hidden_states,
    )
    flux_pe_join = create_flux_pe_join(
        "klein9b.flux.pe_join",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        pe_ctx=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["pe_ctx"]),
        pe_x=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["pe_x"]),
        cat=flux_pe,
    )
    flux_single_hidden = flux_hidden_states
    flux_single_blocks_list: list[FluxSingleBlockTensors] = []
    for layer_idx in range(24):
        layer_tensors = create_flux_single_block(
            f"klein9b.flux.single_block.{{layer_idx}}",
            layer_idx=layer_idx,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            hidden_states=flux_single_hidden,
            pe=flux_pe,
            mod_shift=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["single_mod_shift"]),
            mod_scale=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["single_mod_scale"]),
            mod_gate=getattr(flux_prologue, FLUX_PROLOGUE_OUTPUTS["single_mod_gate"]),
        )
        flux_single_blocks_list.append(layer_tensors)
        flux_single_hidden = getattr(layer_tensors, FLUX_SINGLE_BLOCK_OUTPUTS["hidden_states"])
    flux_single_blocks = tuple(flux_single_blocks_list)
    flux_final_layer = create_flux_final_layer(
        "klein9b.flux.final_layer",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        hidden_states=flux_single_hidden,
        vec=flux_vec,
    )
    euler_update = create_euler_update(
        "klein9b.flux.euler_update",
        image_seq_len=image_seq_len,
        x=latent_tokens,
        pred=getattr(flux_final_layer, FLUX_FINAL_LAYER_OUTPUTS["pred"]),
        add=latent_tokens,
        request_state_outputs=frozenset((EULER_UPDATE_OUTPUT,)),
    )
    ae_entry = None
    ae_decode = (
        None
        if include_ae_decode
        else None
    )
    if include_ae_decode:
        ae_latent_tokens = _request_state_tensor("float32", (1, latent_height, latent_width, 128))
        ae_entry = create_ae_entry(
            "klein9b.ae_entry",
            latent_height=latent_height,
            latent_width=latent_width,
            tokens=ae_latent_tokens,
        )
        bind_logical_tensor_alias(latent_tokens, ae_latent_tokens)
        ae_decode = create_ae_decode(
            "klein9b.ae_decode",
            latent_height=latent_height,
            latent_width=latent_width,
            tokens=getattr(ae_entry, AE_ENTRY_OUTPUT),
            request_state_outputs=frozenset((AE_DECODE_OUTPUT,)),
        )
    _MODEL_TENSORS = QuantizedKlein9BTensors(
        input_ids=input_ids,
        text_rope=text_rope,
        text_embed=text_embed,
        text_layers=text_layers,
        text_context_capture=text_context_capture,
        ctx=ctx,
        latent_tokens=latent_tokens,
        ae_entry=ae_entry,
        flux_hidden_states=flux_hidden_states,
        flux_pe=flux_pe,
        flux_join=flux_join,
        flux_pe_join=flux_pe_join,
        flux=flux,
        flux_prologue=flux_prologue,
        flux_double_blocks=flux_double_blocks,
        flux_single_blocks=flux_single_blocks,
        flux_final_layer=flux_final_layer,
        euler_update=euler_update,
        ae_decode=ae_decode,
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    return _MODEL_TENSORS


def model_tensors() -> QuantizedKlein9BTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors() must be called before model_tensors()")
    return _MODEL_TENSORS


def flux_output(tensors: QuantizedKlein9BTensors | None = None) -> LogicalTensor:
    resolved = model_tensors() if tensors is None else tensors
    return getattr(resolved.flux, FLUX_OUTPUT)


def image_output(tensors: QuantizedKlein9BTensors | None = None) -> LogicalTensor:
    resolved = model_tensors() if tensors is None else tensors
    if resolved.ae_decode is None:
        raise RuntimeError("AE decode tensors were not created")
    return getattr(resolved.ae_decode, AE_DECODE_OUTPUT)


def _host_input_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _request_state_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _frame_workspace_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
    )
''',
        encoding="utf-8",
    )


def _write_init(path: Path) -> None:
    path.write_text('"""Generated package."""\n', encoding="utf-8")


def _shape_symbol_exprs(
    prog: torch.export.ExportedProgram,
    axes_by_placeholder: dict[str, dict[int, str]],
) -> dict[str, str]:
    symbol_exprs: dict[str, str] = {}
    for node in prog.graph_module.graph.nodes:
        axes = axes_by_placeholder.get(node.name)
        if axes is None:
            continue
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            raise ValueError(f"Placeholder {node.name!r} is missing tensor_meta")
        shape = tensor_meta.shape
        for axis, expression in axes.items():
            dim_source = str(shape[axis])
            if dim_source.lstrip("-").isdigit():
                raise ValueError(
                    f"Placeholder {node.name!r} axis {axis} is static; "
                    "expected a dynamic torch.export dimension"
                )
            symbol_exprs[dim_source] = expression
    return symbol_exprs


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    raise ValueError(f"Unsupported activation dtype for Klein9B export: {dtype}")


def _output_map(stage: str, names: tuple[str, ...], semantic_names: tuple[str, ...]) -> dict[str, str]:
    if len(names) != len(semantic_names):
        raise RuntimeError(
            f"{stage} exported {len(names)} outputs, expected {len(semantic_names)}: {names}"
        )
    return dict(zip(semantic_names, names, strict=True))


def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    tensors_dir = output_dir / "tensors"
    dispatch_dir = output_dir / "dispatch"
    shaders_dir.mkdir(exist_ok=True)
    tensors_dir.mkdir(exist_ok=True)
    dispatch_dir.mkdir(exist_ok=True)
    clear_shader_package(shaders_dir)
    clear_python_modules(tensors_dir)
    clear_python_modules(dispatch_dir)

    seen_shader_variants: dict[str, ShaderVariant] = {}
    shader_file_count = 0

    def write_generated_shader(variant: ShaderVariant) -> None:
        nonlocal shader_file_count
        existing = seen_shader_variants.get(variant.name)
        if existing is not None:
            if existing.contract != variant.contract:
                raise ValueError(f"shader name conflict after rename: {variant.name}")
            return
        seen_shader_variants[variant.name] = variant
        write_shader_file(shaders_dir, variant)
        shader_file_count += 1

    reference_functions: list[str] = []
    reference_dispatch_imports: list[str] = []

    def export_one(
        *,
        dispatch_name: str,
        tensor_file: str,
        tensor_class: str,
        create_function: str,
        tensor_expr: str,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, object] | None = None,
        checkpoint: str,
        weight_prefix: str = "",
        quantization_config=None,
        shape_exprs: dict[int, str] | None = None,
        shape_symbol_axes: dict[str, dict[int, str]] | None = None,
        parameters_source: str = "",
        arguments_source: str = "",
        extra_dispatch_functions: str = "",
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
        strict: bool = False,
        is_layered: bool | None = None,
        export_weight_dtype: torch.dtype = torch.float32,
        export_activation_dtype: torch.dtype | None = None,
        export_input_dtype: torch.dtype | None = None,
        tensor_spec_overrides: dict[str, TensorSpecOverride] | None = None,
    ) -> tuple[str, ...]:
        module = module.to(dtype=export_weight_dtype)
        export_dtype = module_floating_dtype(module)
        input_dtype = (
            export_input_dtype
            if export_input_dtype is not None
            else export_activation_dtype
            if export_activation_dtype is not None
            else export_dtype
        )
        registry = (
            DEFAULT_REGISTRY.with_activation_dtype(_torch_dtype_name(export_activation_dtype))
            if export_activation_dtype is not None
            else DEFAULT_REGISTRY
        )
        if input_dtype is not None:
            args = cast_floating_tensors(args, input_dtype)
            kwargs = cast_floating_tensors(kwargs, input_dtype)
        prog = export_submodule(
            module,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
        output_names = tuple(graph_output_names(prog.graph_module.graph))
        shape_symbol_exprs = _shape_symbol_exprs(prog, shape_symbol_axes or {})
        func_name = dispatch_name.removeprefix("run_")
        tensor_ctx = generate_tensor_class_source(
            prog,
            class_name=tensor_class,
            function_name=create_function,
            weight_prefix=weight_prefix,
            checkpoint=checkpoint,
            is_layered=is_layered,
            registry=registry,
            quantization_config=quantization_config,
            shape_exprs=shape_exprs,
            shape_symbol_exprs=shape_symbol_exprs,
            tensor_spec_overrides=tensor_spec_overrides,
        )
        tensor_source = render_tensor_module([tensor_ctx])
        (tensors_dir / f"{tensor_file}.py").write_text(
            tensor_source,
            encoding="utf-8",
        )

        func_src, shader_imports, used_variants = generate_dispatch_function_source(
            prog,
            class_name=tensor_class,
            function_name=dispatch_name,
            shader_package=f"{MODEL_PACKAGE}.shaders",
            registry=registry,
            weight_prefix=weight_prefix,
            quantization_config=quantization_config,
        )
        rename_map: dict[str, str] = {}
        for variant in used_variants.values():
            existing = seen_shader_variants.get(variant.name)
            if existing is None:
                write_generated_shader(variant)
                continue
            if existing.contract == variant.contract:
                continue
            renamed = rename_shader_variant(variant, f"{func_name}_{variant.name}")
            rename_map[variant.name] = renamed.name
            write_generated_shader(renamed)

        for old_name in sorted(rename_map, key=len, reverse=True):
            new_name = rename_map[old_name]
            func_src = re.sub(rf"\b{re.escape(old_name.upper())}\b", new_name.upper(), func_src)
            if old_name in shader_imports:
                shader_imports[new_name] = new_name.upper()
                del shader_imports[old_name]

        (dispatch_dir / f"{func_name}.py").write_text(
            render_model_dispatch_module(
                model_package=MODEL_PACKAGE,
                function_name=dispatch_name,
                tensor_file=tensor_file,
                tensor_class=tensor_class,
                tensor_expr=tensor_expr,
                shader_imports=shader_imports,
                function_source="\n\n\n".join(
                    part
                    for part in (
                        bind_dispatch_function_to_tensors(func_src),
                        extra_dispatch_functions.strip(),
                    )
                    if part
                ),
                parameters_source=parameters_source,
                arguments_source=arguments_source,
                uses_quantized_linear_dispatch="run_quantized_linear(" in func_src,
                workspace_keep_fields=layer_workspace_keep_fields(tensor_ctx),
            ),
            encoding="utf-8",
        )
        print(f"  {dispatch_name}: {len(used_variants)} shaders")
        return output_names

    print("Exporting FLUX.2 Klein 9B Vulkan modules...")
    flux_params = Klein9BParams()
    model_dirs = resolve_model_dirs()
    text_config = AutoConfig.from_pretrained(model_dirs.text_encoder)
    text_layers = int(text_config.num_hidden_layers)
    text_hidden_size = int(text_config.hidden_size)
    text_head_dim = int(text_config.head_dim)
    qwen3_config = qwen3_text_encoder_q8_config()
    flux_config = klein9b_q4_k_m_config()
    ae_config = ae_q4_k_m_config()

    with torch.device("meta"):
        qwen3_model = Qwen3ForCausalLM(text_config).eval().float()
        flux_model = Flux2(flux_params).eval().float()
        ae_decode = AEDecodeModule().eval().float()

    export_one(
        dispatch_name="run_text_embed",
        tensor_file="embed_tokens",
        tensor_class="EmbedTokensTensors",
        create_function="create_embed_tokens",
        tensor_expr="model_tensors().text_embed",
        module=qwen3_model.model.embed_tokens,
        args=(torch.zeros((1, DEFAULT_TEXT_SEQ_LEN), dtype=torch.long, device="meta"),),
        checkpoint="text_encoder/model.gguf",
        weight_prefix="model.embed_tokens.",
        quantization_config=qwen3_config,
        shape_exprs={DEFAULT_TEXT_SEQ_LEN: "sequence_length"},
    )
    export_one(
        dispatch_name="run_text_layer",
        tensor_file="text_layer",
        tensor_class="TextLayerTensors",
        create_function="create_text_layer",
        tensor_expr="model_tensors().text_layers[layer_idx]",
        module=qwen3_model.model.layers[0],
        args=(torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),),
        kwargs={
            "attention_mask": None,
            "position_embeddings": (
                torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_head_dim, device="meta"),
                torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_head_dim, device="meta"),
            ),
            "past_key_values": None,
        },
        checkpoint="text_encoder/model.gguf",
        weight_prefix="model.layers.0.",
        quantization_config=qwen3_config,
        shape_exprs={DEFAULT_TEXT_SEQ_LEN: "sequence_length"},
        parameters_source=", layer_idx: int",
    )
    export_one(
        dispatch_name="run_text_context_capture",
        tensor_file="text_context_capture",
        tensor_class="TextContextCaptureTensors",
        create_function="create_text_context_capture",
        tensor_expr="model_tensors().text_context_capture",
        module=TextContextCaptureModule(),
        args=(
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),
        ),
        checkpoint="text_encoder/model.gguf",
        export_activation_dtype=torch.float32,
        export_input_dtype=torch.float16,
        shape_exprs={DEFAULT_TEXT_SEQ_LEN: "sequence_length"},
        tensor_spec_overrides={
            "layer_9": TensorSpecOverride(
                dtype="float16",
                shape=(1, "sequence_length", text_hidden_size),
            ),
            "layer_18": TensorSpecOverride(
                dtype="float16",
                shape=(1, "sequence_length", text_hidden_size),
            ),
            "layer_27": TensorSpecOverride(
                dtype="float16",
                shape=(1, "sequence_length", text_hidden_size),
            ),
        },
    )
    flux_prologue_output_names = export_one(
        dispatch_name="run_flux_prologue",
        tensor_file="flux_prologue",
        tensor_class="FluxPrologueTensors",
        create_function="create_flux_prologue",
        tensor_expr="model_tensors().flux_prologue",
        module=FluxPrologueModule(flux_model),
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.context_in_dim, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, 4, dtype=torch.long, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        quantization_config=flux_config,
        export_weight_dtype=torch.float16,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    flux_double_block_output_names = export_one(
        dispatch_name="run_flux_double_block",
        tensor_file="flux_double_block",
        tensor_class="FluxDoubleBlockTensors",
        create_function="create_flux_double_block",
        tensor_expr="model_tensors().flux_double_blocks[layer_idx]",
        module=FluxDoubleBlockModule(cast(DoubleStreamBlock, flux_model.double_blocks[0])),
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.hidden_size, device="meta"),
            torch.zeros(
                1,
                1,
                DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            torch.zeros(
                1,
                1,
                DEFAULT_TEXT_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            *(
                torch.zeros(1, 1, flux_params.hidden_size, device="meta")
                for _ in range(12)
            ),
        ),
        checkpoint="flux/model.gguf",
        weight_prefix="double_blocks.0.",
        quantization_config=flux_config,
        export_weight_dtype=torch.float16,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
        parameters_source=", layer_idx: int",
        is_layered=True,
    )
    export_one(
        dispatch_name="run_flux_join",
        tensor_file="flux_join",
        tensor_class="FluxJoinTensors",
        create_function="create_flux_join",
        tensor_expr="model_tensors().flux_join",
        module=FluxJoinModule(),
        args=(
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.hidden_size, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    export_one(
        dispatch_name="run_flux_pe_join",
        tensor_file="flux_pe_join",
        tensor_class="FluxPeJoinTensors",
        create_function="create_flux_pe_join",
        tensor_expr="model_tensors().flux_pe_join",
        module=FluxPeJoinModule(),
        args=(
            torch.zeros(
                1,
                1,
                DEFAULT_TEXT_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            torch.zeros(
                1,
                1,
                DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
        ),
        checkpoint="flux/model.gguf",
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    flux_single_block_output_names = export_one(
        dispatch_name="run_flux_single_block",
        tensor_file="flux_single_block",
        tensor_class="FluxSingleBlockTensors",
        create_function="create_flux_single_block",
        tensor_expr="model_tensors().flux_single_blocks[layer_idx]",
        module=FluxSingleBlockModule(
            cast(SingleStreamBlock, flux_model.single_blocks[0]),
            text_seq_len=DEFAULT_TEXT_SEQ_LEN,
        ),
        args=(
            torch.zeros(
                1,
                DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size,
                device="meta",
            ),
            torch.zeros(
                1,
                1,
                DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            *(
                torch.zeros(1, 1, flux_params.hidden_size, device="meta")
                for _ in range(3)
            ),
        ),
        checkpoint="flux/model.gguf",
        weight_prefix="single_blocks.0.",
        quantization_config=flux_config,
        export_weight_dtype=torch.float16,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
        parameters_source=", layer_idx: int",
        is_layered=True,
    )
    flux_final_layer_output_names = export_one(
        dispatch_name="run_flux_final_layer",
        tensor_file="flux_final_layer",
        tensor_class="FluxFinalLayerTensors",
        create_function="create_flux_final_layer",
        tensor_expr="model_tensors().flux_final_layer",
        module=FluxFinalLayerModule(flux_model, text_seq_len=DEFAULT_TEXT_SEQ_LEN),
        args=(
            torch.zeros(
                1,
                DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size,
                device="meta",
            ),
            torch.zeros(1, flux_params.hidden_size, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        quantization_config=flux_config,
        export_weight_dtype=torch.float16,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    export_one(
        dispatch_name="run_euler_update",
        tensor_file="euler_update",
        tensor_class="EulerUpdateTensors",
        create_function="create_euler_update",
        tensor_expr="model_tensors().euler_update",
        module=EulerUpdateModule(),
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        export_activation_dtype=torch.float32,
        shape_exprs={DEFAULT_IMAGE_SEQ_LEN: "image_seq_len"},
    )
    flux_prologue_outputs = _output_map(
        "flux_prologue",
        flux_prologue_output_names,
        (
            "img",
            "txt",
            "pe_x",
            "pe_ctx",
            "vec",
            "img_mod1_shift",
            "img_mod1_scale",
            "img_mod1_gate",
            "img_mod2_shift",
            "img_mod2_scale",
            "img_mod2_gate",
            "txt_mod1_shift",
            "txt_mod1_scale",
            "txt_mod1_gate",
            "txt_mod2_shift",
            "txt_mod2_scale",
            "txt_mod2_gate",
            "single_mod_shift",
            "single_mod_scale",
            "single_mod_gate",
        ),
    )
    flux_double_block_outputs = _output_map(
        "flux_double_block",
        flux_double_block_output_names,
        (
            "img_k_raw",
            "txt_k_raw",
            "img_k_rrms",
            "txt_k_rrms",
            "txt_q_unit",
            "img_q_unit",
            "txt_k_unit",
            "img_k_unit",
            "txt_q",
            "img_q",
            "txt_k",
            "img_k",
            "txt_v",
            "img_v",
            "q",
            "k",
            "v",
            "pe_full",
            "q_rope",
            "k_rope",
            "attn",
            "txt_attn",
            "img_attn",
            "img_attn_proj",
            "img_after_attn",
            "img_mlp_in",
            "img_mlp_hidden",
            "img_mlp_act",
            "img_mlp_out",
            "img",
            "txt_attn_proj",
            "txt_after_attn",
            "txt_mlp_in",
            "txt_mlp_hidden",
            "txt_mlp_act",
            "txt_mlp_out",
            "txt_mlp_gated",
            "txt",
        ),
    )
    flux_single_block_outputs = _output_map(
        "flux_single_block",
        flux_single_block_output_names,
        (
            "pre_norm",
            "x_mod",
            "linear1",
            "q_raw",
            "k_raw",
            "v",
            "q_unit",
            "k_unit",
            "q",
            "k",
            "q_rope",
            "k_rope",
            "attn",
            "mlp",
            "mlp_act",
            "out_input",
            "linear2",
            "gated",
            "hidden_states",
        ),
    )
    flux_final_layer_outputs = _output_map(
        "flux_final_layer",
        flux_final_layer_output_names,
        ("pred",),
    )
    reference_dispatch_imports.extend(
        (
            "from models.quantized_klein9b.dispatch.flux_prologue import run_flux_prologue as _dispatch_flux_prologue",
            "from models.quantized_klein9b.dispatch.flux_double_block import run_flux_double_block as _dispatch_flux_double_block",
            "from models.quantized_klein9b.dispatch.flux_single_block import run_flux_single_block as _dispatch_flux_single_block",
            "from models.quantized_klein9b.dispatch.flux_final_layer import run_flux_final_layer as _dispatch_flux_final_layer",
        )
    )
    reference_functions.extend(
        (
            render_streaming_compare_function(
                name="flux_prologue",
                dispatch_source="_dispatch_flux_prologue",
                tensors="model_tensors().flux_prologue",
                frame_name="klein9b.flux.compare.{step:04d}.prologue",
                policy="tensor",
                input_bindings={
                    "x": "x",
                    "x_ids": "x_ids",
                    "timesteps": "timesteps",
                    "ctx": "ctx",
                    "ctx_ids": "ctx_ids",
                },
                output_bindings=flux_prologue_outputs,
            ),
            render_streaming_compare_function(
                name="flux_double_block",
                dispatch_source="_dispatch_flux_double_block",
                tensors="model_tensors().flux_double_blocks[layer_idx]",
                frame_name="klein9b.flux.compare.{step:04d}.double_block.{layer_idx}",
                policy="tensor",
                input_bindings={
                    "img": "img",
                    "txt": "txt",
                    "pe": "pe",
                    "pe_ctx": "pe_ctx",
                    "img_mod1_shift": "img_mod1_shift",
                    "img_mod1_scale": "img_mod1_scale",
                    "img_mod1_gate": "img_mod1_gate",
                    "img_mod2_shift": "img_mod2_shift",
                    "img_mod2_scale": "img_mod2_scale",
                    "img_mod2_gate": "img_mod2_gate",
                    "txt_mod1_shift": "txt_mod1_shift",
                    "txt_mod1_scale": "txt_mod1_scale",
                    "txt_mod1_gate": "txt_mod1_gate",
                    "txt_mod2_shift": "txt_mod2_shift",
                    "txt_mod2_scale": "txt_mod2_scale",
                    "txt_mod2_gate": "txt_mod2_gate",
                },
                output_bindings=flux_double_block_outputs,
                dispatch_args=("layer_idx",),
            ),
            render_streaming_compare_function(
                name="flux_single_block",
                dispatch_source="_dispatch_flux_single_block",
                tensors="model_tensors().flux_single_blocks[layer_idx]",
                frame_name="klein9b.flux.compare.{step:04d}.single_block.{layer_idx}",
                policy="tensor",
                input_bindings={
                    "hidden_states": "hidden_states",
                    "pe": "pe",
                    "mod_shift": "mod_shift",
                    "mod_scale": "mod_scale",
                    "mod_gate": "mod_gate",
                },
                output_bindings=flux_single_block_outputs,
                dispatch_args=("layer_idx",),
            ),
            render_streaming_compare_function(
                name="flux_final_layer",
                dispatch_source="_dispatch_flux_final_layer",
                tensors="model_tensors().flux_final_layer",
                frame_name="klein9b.flux.compare.{step:04d}.final_layer",
                policy="tensor",
                input_bindings={
                    "hidden_states": "hidden_states",
                    "vec": "vec",
                },
                output_bindings=flux_final_layer_outputs,
            ),
        )
    )
    export_one(
        dispatch_name="run_flux",
        tensor_file="flux",
        tensor_class="FluxTensors",
        create_function="create_flux",
        tensor_expr="model_tensors().flux",
        module=flux_model,
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.context_in_dim, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        quantization_config=flux_config,
        export_weight_dtype=torch.float16,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    export_one(
        dispatch_name="run_ae_entry",
        tensor_file="ae_entry",
        tensor_class="AEEntryTensors",
        create_function="create_ae_entry",
        tensor_expr="_require_ae_entry_tensors()",
        module=AEEntryModule(),
        args=(torch.zeros(1, DEFAULT_LATENT_HEIGHT, DEFAULT_LATENT_WIDTH, 128, device="meta"),),
        checkpoint="ae/model.gguf",
        export_activation_dtype=torch.float16,
        export_input_dtype=torch.float32,
        shape_symbol_axes={"tokens": {1: "latent_height", 2: "latent_width"}},
        tensor_spec_overrides={
            "tokens": TensorSpecOverride(
                dtype="float32",
                shape=(1, "latent_height", "latent_width", 128),
            ),
        },
        dynamic_shapes={
            "tokens": {
                1: torch.export.Dim("latent_height", min=4, max=256),
                2: torch.export.Dim("latent_width", min=4, max=256),
            }
        },
        strict=True,
        extra_dispatch_functions='''def _require_ae_entry_tensors() -> AEEntryTensors:
    tensors = model_tensors().ae_entry
    if tensors is None:
        raise RuntimeError("AE entry tensors were not created")
    return tensors''',
    )
    export_one(
        dispatch_name="run_ae_decode",
        tensor_file="ae_decode",
        tensor_class="AeDecodeTensors",
        create_function="create_ae_decode",
        tensor_expr="_require_ae_decode_tensors()",
        module=ae_decode,
        args=(torch.zeros(1, 128, DEFAULT_LATENT_HEIGHT, DEFAULT_LATENT_WIDTH, device="meta"),),
        checkpoint="ae/model.gguf",
        quantization_config=ae_config,
        export_activation_dtype=torch.float16,
        shape_symbol_axes={"tokens": {2: "latent_height", 3: "latent_width"}},
        dynamic_shapes={
            "tokens": {
                2: torch.export.Dim("latent_height", min=4, max=256),
                3: torch.export.Dim("latent_width", min=4, max=256),
            }
        },
        strict=True,
        extra_dispatch_functions='''def _require_ae_decode_tensors() -> AeDecodeTensors:
    tensors = model_tensors().ae_decode
    if tensors is None:
        raise RuntimeError("AE decode tensors were not created")
    return tensors''',
    )

    (tensors_dir / "rope.py").write_text(
        '''"""Generated RoPE tensor declarations."""

from __future__ import annotations

from torch2vk.runtime.rope_table import RopeTableTensors, declare_rope_table_tensors


def create_rope_table(
    prefix: str,
    *,
    batch: int,
    sequence_length: int,
    head_dim: int,
) -> RopeTableTensors:
    return declare_rope_table_tensors(
        prefix,
        batch=batch,
        sequence_length=sequence_length,
        head_dim=head_dim,
    )
''',
        encoding="utf-8",
    )
    _write_model_tensors_module(
        tensors_dir / "model.py",
        text_seq_len=DEFAULT_TEXT_SEQ_LEN,
        num_text_layers=text_layers,
        text_hidden_size=text_hidden_size,
        text_head_dim=text_head_dim,
        flux_prologue_outputs=flux_prologue_outputs,
        flux_double_block_outputs=flux_double_block_outputs,
        flux_single_block_outputs=flux_single_block_outputs,
        flux_final_layer_outputs=flux_final_layer_outputs,
    )
    _write_init(tensors_dir / "__init__.py")
    _write_init(dispatch_dir / "__init__.py")

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package=MODEL_PACKAGE,
            reference_functions=reference_functions,
            dispatch_imports=reference_dispatch_imports,
        ),
        encoding="utf-8",
    )

    print(f"  {shader_file_count} shader files written")
    print(f"  tensors/ written ({count_python_modules(tensors_dir)} files)")
    print(f"  dispatch/ written ({count_python_modules(dispatch_dir)} files)")
    print("  reference.py written")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
