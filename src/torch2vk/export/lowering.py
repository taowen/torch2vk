"""Framework-level op->shader lowering registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch2vk.export.ir import TorchOpPattern


LoweringMatchFn = Callable[[TorchOpPattern], bool]


@dataclass(frozen=True, slots=True)
class OpShaderBinding:
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
        self._bindings: list[OpShaderBinding] = []
        for binding in bindings:
            self.register(binding)

    def register(self, binding: OpShaderBinding) -> None:
        _require_aten_target(binding.target, context="binding.target")
        self._bindings.append(binding)

    def resolve(self, *, op: TorchOpPattern) -> OpShaderBinding | None:
        _require_aten_target(op.target, context="op.target")
        for binding in self._bindings:
            if binding.matches(op):
                return binding
        return None


def _match_second_input(name: str) -> LoweringMatchFn:
    def _match(op: TorchOpPattern) -> bool:
        return len(op.inputs) > 1 and op.inputs[1] == name

    return _match


def _require_aten_target(target: str, *, context: str) -> None:
    if not target.startswith("aten."):
        raise ValueError(f"{context} must start with 'aten.', got {target!r}")


DEFAULT_LOWERING_REGISTRY = OpLoweringRegistry(
    (
        OpShaderBinding(
            target="aten.select.int",
            shader="OMNIVOICE_ATEN_SELECT_INT_I64",
        ),
        OpShaderBinding(
            target="aten.embedding.default",
            shader="OMNIVOICE_ATEN_EMBEDDING_F32",
            match=_match_second_input("text_token_ids"),
        ),
        OpShaderBinding(
            target="aten.add.Tensor",
            shader="OMNIVOICE_ATEN_SHIFTED_IDS_I64",
        ),
        OpShaderBinding(
            target="aten.embedding.default",
            shader="OMNIVOICE_ATEN_EMBEDDING_3D_F32",
            match=_match_second_input("shifted_ids"),
        ),
        OpShaderBinding(
            target="aten.sum.dim_IntList",
            shader="OMNIVOICE_ATEN_SUM_DIM1_F32",
        ),
        OpShaderBinding(
            target="aten.where.self",
            shader="OMNIVOICE_ATEN_WHERE_F32",
        ),
    )
)


def resolve_shader_symbol(*, op: TorchOpPattern) -> str | None:
    binding = DEFAULT_LOWERING_REGISTRY.resolve(op=op)
    return None if binding is None else binding.shader


FRAME_SHADER_REGISTRY: dict[tuple[str, str], dict[str, str]] = {
    ("generated_qwen3_asr", "audio_tower"): {
        "pad_feature": "QWEN3_ASR_PAD_FEATURE_F32",
        "conv2d_gelu": "QWEN3_ASR_CONV2D_GELU_F32",
        "conv_out": "QWEN3_ASR_CONV_OUT_F32",
        "add_position": "QWEN3_ASR_ADD_POSITION_F32",
        "compact_after_cnn": "QWEN3_ASR_COMPACT_AFTER_CNN_F32",
        "cu_seqlens": "QWEN3_ASR_CU_SEQLENS_U32",
        "layer_norm": "QWEN3_ASR_LAYER_NORM_F32",
        "linear_gelu": "QWEN3_ASR_LINEAR_GELU_F32",
        "linear": "QWEN3_ASR_LINEAR_F32",
    },
    ("generated_qwen3_asr", "audio_encoder_layer"): {
        "layer_norm": "QWEN3_ASR_LAYER_NORM_F32",
        "linear": "QWEN3_ASR_LINEAR_F32",
        "attention": "QWEN3_ASR_ENCODER_ATTENTION_F32",
        "residual_add": "QWEN3_ASR_ADD_F32",
        "linear_gelu": "QWEN3_ASR_LINEAR_GELU_F32",
    },
    ("generated_qwen3_asr", "text_prefill"): {
        "prefill_inputs_embeds": "QWEN3_ASR_TEXT_PREFILL_INPUTS_EMBEDS_F32",
        "rms_norm": "QWEN3_ASR_TEXT_RMS_NORM_F32",
        "lm_head": "QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32",
    },
    ("generated_qwen3_asr", "text_decode"): {
        "embed_lookup": "QWEN3_ASR_TEXT_EMBED_LOOKUP_F32",
        "rms_norm": "QWEN3_ASR_TEXT_RMS_NORM_F32",
        "lm_head_or_token_select": "QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32",
    },
    ("generated_qwen3_asr", "token_select"): {
        "greedy_argmax": "QWEN3_ASR_TOKEN_SELECT_GREEDY_F32",
    },
    ("generated_qwen3_asr", "token_store"): {
        "token_store": "QWEN3_ASR_TOKEN_STORE_F32",
    },
}


def resolve_frame_shader(*, model: str, frame: str, target: str) -> str:
    shader = FRAME_SHADER_REGISTRY.get((model, frame), {}).get(target)
    if shader is None:
        raise NotImplementedError(f"Unsupported generated {model}.{frame} op: {target}")
    return shader
