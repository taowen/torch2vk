"""Generated model-level tensor wiring for FLUX.2 Klein 9B."""

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
from models.quantized_klein9b.tensors.flux_join import FLUX_JOIN_OUTPUT, FluxJoinTensors, create_flux_join
from models.quantized_klein9b.tensors.flux_pe_join import (
    FLUX_PE_JOIN_OUTPUT,
    FluxPeJoinTensors,
    create_flux_pe_join,
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
from models.quantized_klein9b.tensors.text_context_capture import (
    TEXT_CONTEXT_CAPTURE_OUTPUT,
    TextContextCaptureTensors,
    create_text_context_capture,
)
from models.quantized_klein9b.tensors.text_layer import TextLayerTensors, create_text_layer
from torch2vk.runtime.logical import (
    bind_logical_tensor_alias,
    bind_logical_tensor_names,
    LogicalTensor,
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
            f"klein9b.flux.double_block.{layer_idx}",
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
            f"klein9b.flux.single_block.{layer_idx}",
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
    ae_decode = None
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
