"""Generated model-level tensor wiring for FLUX.2 Klein 9B."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_klein9b.tensors.ae_decode import (
    AE_DECODE_OUTPUT,
    AeDecodeTensors,
    create_ae_decode,
)
from models.quantized_klein9b.tensors.embed_tokens import EmbedTokensTensors, create_embed_tokens
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
from models.quantized_klein9b.tensors.rope import RopeTableTensors, create_rope_table
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
    ctx: LogicalTensor
    latent_tokens: LogicalTensor
    flux: FluxTensors
    flux_prologue: FluxPrologueTensors
    flux_double_blocks: tuple[FluxDoubleBlockTensors, ...]
    flux_single_blocks: tuple[FluxSingleBlockTensors, ...]
    flux_final_layer: FluxFinalLayerTensors
    ae_decode: AeDecodeTensors | None


_MODEL_TENSORS: QuantizedKlein9BTensors | None = None
FLUX_PROLOGUE_OUTPUTS: dict[str, str] = {'img': 'linear_5', 'txt': 'linear_6', 'pe_x': 'unsqueeze_5', 'pe_ctx': 'unsqueeze_6', 'vec': 'linear_1', 'img_mod1_shift': 'getitem', 'img_mod1_scale': 'getitem_1', 'img_mod1_gate': 'getitem_2', 'img_mod2_shift': 'getitem_3', 'img_mod2_scale': 'getitem_4', 'img_mod2_gate': 'getitem_5', 'txt_mod1_shift': 'getitem_6', 'txt_mod1_scale': 'getitem_7', 'txt_mod1_gate': 'getitem_8', 'txt_mod2_shift': 'getitem_9', 'txt_mod2_scale': 'getitem_10', 'txt_mod2_gate': 'getitem_11', 'single_mod_shift': 'getitem_12', 'single_mod_scale': 'getitem_13', 'single_mod_gate': 'getitem_14'}
FLUX_DOUBLE_BLOCK_OUTPUTS: dict[str, str] = {'img': 'add_13', 'txt': 'add_17'}
FLUX_SINGLE_BLOCK_OUTPUTS: dict[str, str] = {'hidden_states': 'add_6'}
FLUX_FINAL_LAYER_OUTPUTS: dict[str, str] = {'pred': 'linear_1'}


def create_model_tensors(
    *,
    latent_height: int = 32,
    latent_width: int = 32,
    text_seq_len: int = 512,
    num_text_layers: int = 36,
    text_hidden_size: int = 4096,
    text_head_dim: int = 128,
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
            f"klein9b.text.layer.{layer_idx}",
            layer_idx=layer_idx,
            sequence_length=text_seq_len,
            hidden_states=text_hidden,
            position_embeddings_0=text_rope.cos,
            position_embeddings_1=text_rope.sin,
        )
        text_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_layers = tuple(text_layers_list)
    ctx = _request_state_tensor("float16", (1, text_seq_len, text_hidden_size * 3))
    latent_tokens = _request_state_tensor("float16", (1, image_seq_len, 128))
    ae_tokens = _request_state_tensor("float16", (1, latent_height, latent_width, 128))
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
    flux_double_blocks = tuple(
        create_flux_double_block(
            f"klein9b.flux.double_block.{layer_idx}",
            layer_idx=layer_idx,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )
        for layer_idx in range(8)
    )
    flux_single_blocks = tuple(
        create_flux_single_block(
            f"klein9b.flux.single_block.{layer_idx}",
            layer_idx=layer_idx,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )
        for layer_idx in range(24)
    )
    flux_final_layer = create_flux_final_layer(
        "klein9b.flux.final_layer",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
    )
    ae_decode = (
        create_ae_decode(
            "klein9b.ae_decode",
            latent_height=latent_height,
            latent_width=latent_width,
            tokens=ae_tokens,
            request_state_outputs=frozenset((AE_DECODE_OUTPUT,)),
        )
        if include_ae_decode
        else None
    )
    _MODEL_TENSORS = QuantizedKlein9BTensors(
        input_ids=input_ids,
        text_rope=text_rope,
        text_embed=text_embed,
        text_layers=text_layers,
        ctx=ctx,
        latent_tokens=latent_tokens,
        flux=flux,
        flux_prologue=flux_prologue,
        flux_double_blocks=flux_double_blocks,
        flux_single_blocks=flux_single_blocks,
        flux_final_layer=flux_final_layer,
        ae_decode=ae_decode,
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    if ae_decode is not None:
        bind_logical_tensor_alias(latent_tokens, ae_decode.tokens)
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
