"""Framework-level op->shader lowering registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from torch2vk.export.protocols import ExportOpLike

LoweringMatchFn = Callable[[tuple[str, ...]], bool]


@dataclass(frozen=True, slots=True)
class OpShaderBinding:
    target: str
    shader: str
    match: LoweringMatchFn | None = None
    note: str = ""

    def matches(self, op: ExportOpLike) -> bool:
        return self.matches_target_inputs(target=op.target, inputs=op.inputs)

    def matches_target_inputs(self, *, target: str, inputs: Sequence[str]) -> bool:
        if target != self.target:
            return False
        if self.match is None:
            return True
        return self.match(tuple(inputs))


class OpLoweringRegistry:
    def __init__(self, bindings: Iterable[OpShaderBinding] = ()) -> None:
        self._bindings: list[OpShaderBinding] = []
        for binding in bindings:
            self.register(binding)

    def register(self, binding: OpShaderBinding) -> None:
        _require_aten_target(binding.target, context="binding.target")
        self._bindings.append(binding)

    def resolve(self, *, op: ExportOpLike) -> OpShaderBinding | None:
        _require_aten_target(op.target, context="op.target")
        for binding in self._bindings:
            if binding.matches(op):
                return binding
        return None

    def resolve_target_inputs(
        self,
        *,
        target: str,
        inputs: Sequence[str] = (),
    ) -> OpShaderBinding | None:
        _require_aten_target(target, context="op.target")
        for binding in self._bindings:
            if binding.matches_target_inputs(target=target, inputs=inputs):
                return binding
        return None


def _match_second_input(name: str) -> LoweringMatchFn:
    def _match(inputs: tuple[str, ...]) -> bool:
        return len(inputs) > 1 and inputs[1] == name

    return _match


def _require_aten_target(target: str, *, context: str) -> None:
    if not target.startswith("aten."):
        raise ValueError(f"{context} must start with 'aten.', got {target!r}")


DEFAULT_LOWERING_REGISTRY = OpLoweringRegistry(
    (
        OpShaderBinding(
            target="aten.select.int",
            shader="ATEN_SELECT_INT_I64",
        ),
        OpShaderBinding(
            target="aten.embedding.default",
            shader="ATEN_EMBEDDING_F32",
            match=_match_second_input("text_token_ids"),
        ),
        OpShaderBinding(
            target="aten.torch2vk.shifted_ids.default",
            shader="ATEN_SHIFTED_IDS_I64",
        ),
        OpShaderBinding(
            target="aten.embedding.default",
            shader="ATEN_EMBEDDING_3D_F32",
            match=_match_second_input("shifted_ids"),
        ),
        OpShaderBinding(
            target="aten.sum.dim_IntList",
            shader="ATEN_SUM_DIM1_F32",
        ),
        OpShaderBinding(
            target="aten.where.self",
            shader="ATEN_WHERE_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.pad_feature.default",
            shader="PAD_FEATURE_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.conv2d_gelu.default",
            shader="CONV2D_GELU_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.conv_out.default",
            shader="CONV_OUT_F32",
        ),
        OpShaderBinding(
            target="aten.add.Tensor",
            shader="ADD_POSITION_F32",
            match=lambda inputs: len(inputs) == 1 and inputs[0] == "conv_out",
        ),
        OpShaderBinding(
            target="aten.torch2vk.compact_after_cnn.default",
            shader="COMPACT_AFTER_CNN_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.cu_seqlens.default",
            shader="CU_SEQLENS_U32",
        ),
        OpShaderBinding(
            target="aten.native_layer_norm.default",
            shader="LAYER_NORM_F32",
        ),
        OpShaderBinding(
            target="aten.linear.default",
            shader="LINEAR_F32",
            match=lambda inputs: len(inputs) == 3,
        ),
        OpShaderBinding(
            target="aten.torch2vk.linear_gelu.default",
            shader="LINEAR_GELU_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.encoder_attention.default",
            shader="ENCODER_ATTENTION_F32",
        ),
        OpShaderBinding(
            target="aten.add.Tensor",
            shader="ADD_F32",
            match=lambda inputs: len(inputs) == 2
            and any(inputs[1].endswith(s) for s in ("out_proj", "fc2", ".out_proj", ".fc2")),
        ),
        OpShaderBinding(
            target="aten.torch2vk.prefill_inputs_embeds.default",
            shader="TEXT_PREFILL_INPUTS_EMBEDS_F32",
        ),
        OpShaderBinding(
            target="aten.rms_norm.default",
            shader="TEXT_RMS_NORM_F32",
        ),
        OpShaderBinding(
            target="aten.linear.default",
            shader="TEXT_LINEAR_NOBIAS_F32",
            match=lambda inputs: len(inputs) == 2,
        ),
        OpShaderBinding(
            target="aten.embedding.default",
            shader="TEXT_EMBED_LOOKUP_F32",
            match=lambda inputs: inputs == ("input_ids", "embed_tokens_weight"),
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_qk_norm.default",
            shader="TEXT_QK_NORM_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_rope.default",
            shader="TEXT_ROPE_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_kv_cache_write.default",
            shader="TEXT_KV_CACHE_WRITE_DECODE_F32",
            match=lambda inputs: "cache_position" in inputs,
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_kv_cache_write.default",
            shader="TEXT_KV_CACHE_WRITE_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_attention.default",
            shader="TEXT_ATTENTION_DECODE_F32",
            match=lambda inputs: "cache_position" in inputs,
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_attention.default",
            shader="TEXT_ATTENTION_PREFILL_F32",
        ),
        OpShaderBinding(
            target="aten.add.Tensor",
            shader="TEXT_ADD_3D_F32",
            match=lambda inputs: len(inputs) == 2
            and any(inputs[1].endswith(s) for s in ("o_proj", "down_proj", ".o_proj", ".down_proj")),
        ),
        OpShaderBinding(
            target="aten.torch2vk.text_swiglu.default",
            shader="TEXT_SWIGLU_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.greedy_argmax.default",
            shader="TOKEN_SELECT_GREEDY_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.token_store.default",
            shader="TOKEN_STORE_F32",
        ),
        OpShaderBinding(
            target="aten.torch2vk.rope_table.default",
            shader="ROPE_TABLE_F32",
        ),
    )
)


def resolve_shader_symbol(*, op: ExportOpLike) -> str | None:
    binding = DEFAULT_LOWERING_REGISTRY.resolve(op=op)
    return None if binding is None else binding.shader


def resolve_shader_symbol_from_target_inputs(
    *,
    target: str,
    inputs: Sequence[str] = (),
) -> str | None:
    binding = DEFAULT_LOWERING_REGISTRY.resolve_target_inputs(target=target, inputs=inputs)
    return None if binding is None else binding.shader
