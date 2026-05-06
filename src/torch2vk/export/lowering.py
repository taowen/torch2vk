"""Framework-level op->shader lowering registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch2vk.export.ir import TorchOpPattern


LoweringMatchFn = Callable[[TorchOpPattern], bool]


@dataclass(frozen=True, slots=True)
class OpShaderBinding:
    model: str
    frame: str
    target: str
    shader: str
    match: LoweringMatchFn | None = None
    note: str = ""

    def matches(self, op: TorchOpPattern) -> bool:
        if op.target != self.target:
            return False
        if self.match is None:
            return True
        return self.match(op)


class OpLoweringRegistry:
    def __init__(self, bindings: Iterable[OpShaderBinding] = ()) -> None:
        self._bindings = list(bindings)

    def register(self, binding: OpShaderBinding) -> None:
        self._bindings.append(binding)

    def resolve(self, *, model: str, frame: str, op: TorchOpPattern) -> OpShaderBinding | None:
        for binding in self._bindings:
            if binding.model != model or binding.frame != frame:
                continue
            if binding.matches(op):
                return binding
        return None


def _match_second_input(name: str) -> LoweringMatchFn:
    def _match(op: TorchOpPattern) -> bool:
        return len(op.inputs) > 1 and op.inputs[1] == name

    return _match


DEFAULT_LOWERING_REGISTRY = OpLoweringRegistry(
    (
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.select.int",
            shader="OMNIVOICE_ATEN_SELECT_INT_I64",
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.embedding.default",
            shader="OMNIVOICE_ATEN_EMBEDDING_F32",
            match=_match_second_input("text_token_ids"),
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.add.Tensor",
            shader="OMNIVOICE_ATEN_SHIFTED_IDS_I64",
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.embedding.default",
            shader="OMNIVOICE_ATEN_EMBEDDING_3D_F32",
            match=_match_second_input("shifted_ids"),
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.sum.dim_IntList",
            shader="OMNIVOICE_ATEN_SUM_DIM1_F32",
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.where.self",
            shader="OMNIVOICE_ATEN_WHERE_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="pad_feature",
            shader="QWEN3_ASR_PAD_FEATURE_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="conv2d_gelu",
            shader="QWEN3_ASR_CONV2D_GELU_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="conv_out",
            shader="QWEN3_ASR_CONV_OUT_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="add_position",
            shader="QWEN3_ASR_ADD_POSITION_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="compact_after_cnn",
            shader="QWEN3_ASR_COMPACT_AFTER_CNN_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="cu_seqlens",
            shader="QWEN3_ASR_CU_SEQLENS_U32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="layer_norm",
            shader="QWEN3_ASR_LAYER_NORM_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="linear_gelu",
            shader="QWEN3_ASR_LINEAR_GELU_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_tower",
            target="linear",
            shader="QWEN3_ASR_LINEAR_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_encoder_layer",
            target="layer_norm",
            shader="QWEN3_ASR_LAYER_NORM_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_encoder_layer",
            target="linear",
            shader="QWEN3_ASR_LINEAR_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_encoder_layer",
            target="attention",
            shader="QWEN3_ASR_ENCODER_ATTENTION_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_encoder_layer",
            target="residual_add",
            shader="QWEN3_ASR_ADD_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="audio_encoder_layer",
            target="linear_gelu",
            shader="QWEN3_ASR_LINEAR_GELU_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="text_prefill",
            target="prefill_inputs_embeds",
            shader="QWEN3_ASR_TEXT_PREFILL_INPUTS_EMBEDS_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="text_prefill",
            target="rms_norm",
            shader="QWEN3_ASR_TEXT_RMS_NORM_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="text_prefill",
            target="lm_head",
            shader="QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="text_decode",
            target="embed_lookup",
            shader="QWEN3_ASR_TEXT_EMBED_LOOKUP_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="text_decode",
            target="rms_norm",
            shader="QWEN3_ASR_TEXT_RMS_NORM_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="text_decode",
            target="lm_head_or_token_select",
            shader="QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="token_select",
            target="greedy_argmax",
            shader="QWEN3_ASR_TOKEN_SELECT_GREEDY_F32",
        ),
        OpShaderBinding(
            model="generated_qwen3_asr",
            frame="token_store",
            target="token_store",
            shader="QWEN3_ASR_TOKEN_STORE_F32",
        ),
    )
)


def resolve_shader_symbol(*, model: str, frame: str, op: TorchOpPattern) -> str | None:
    binding = DEFAULT_LOWERING_REGISTRY.resolve(model=model, frame=frame, op=op)
    return None if binding is None else binding.shader
