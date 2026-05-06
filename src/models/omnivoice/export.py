"""Regenerate OmniVoice Vulkan adapter scaffold files.

OmniVoice-specific export knowledge lives here. The shared ``torch2vk.export``
package only provides generic template rendering, generated-file writes, and
PyTorch module reflection.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import importlib.util
import io
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from torch2vk.export.exported_program import export_torch_program, torch_ops_from_exported_program
from torch2vk.export.contract_codegen import (
    ParamsFieldDecl,
    TensorFieldDecl,
    render_shader_contract_variant_body,
)
from torch2vk.export.reflection import (
    TorchModuleReflection,
    instantiate_torch_module_on_meta,
    reflect_torch_module,
)
from torch2vk.export.torch_ops import TorchOpPattern, TensorFieldPattern
from torch2vk.export.writer import (
    ExportWriteResult,
    RenderedFile,
    TemplateRenderer,
    remove_stale_files,
    write_rendered_files,
)


JsonObject = Mapping[str, object]

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATE_ROOT = _PACKAGE_DIR / "export_templates"
_STALE_FILES = (
    Path("tensors/model.py"),
    Path("tensors/llm.py"),
    Path("tensors/pipeline.py"),
    Path("shaders/registry.py"),
    Path("shaders/_imported.py"),
    Path("shaders/audio_head_f32.py"),
    Path("shaders/token_select_f32.py"),
    Path("shaders/audio_codec_decoder_f32.py"),
)
_IMPORTED_GLSL_DIR = _PACKAGE_DIR / "imported_glsl"


@dataclass(frozen=True, slots=True)
class OmniVoiceExportConfig:
    audio_vocab_size: int
    audio_mask_id: int
    num_audio_codebook: int
    llm_hidden_size: int
    llm_intermediate_size: int
    llm_vocab_size: int
    llm_num_hidden_layers: int
    llm_num_attention_heads: int
    llm_num_key_value_heads: int
    llm_head_dim: int

    def to_template_defaults(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShaderPattern:
    constant: str
    shader_name: str
    class_name: str
    family: str
    source_file: str
    source: str
    variant_body: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class ShaderModulePattern:
    module: str
    doc: str
    variants: tuple[ShaderPattern, ...]


def export_omnivoice_package(
    *,
    output_dir: str | Path = _PACKAGE_DIR,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
    reflect_source: bool = True,
    check: bool = False,
    dry_run: bool = False,
) -> ExportWriteResult:
    config = load_omnivoice_export_config(model_dir=model_dir, config_json=config_json)
    reflection = reflect_omnivoice_source(config) if reflect_source else None
    context = build_omnivoice_pattern_context(config=config, reflection=reflection)
    renderer = TemplateRenderer(_TEMPLATE_ROOT)
    files = _render_omnivoice_files(renderer, context)
    result = write_rendered_files(files, output_dir, check=check, dry_run=dry_run)
    _remove_stale_files(output_dir, check=check, dry_run=dry_run)
    return result


def load_omnivoice_export_config(
    *,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
) -> OmniVoiceExportConfig:
    if config_json is not None:
        payload = _load_json(Path(config_json))
    elif model_dir is not None:
        payload = _load_json(Path(model_dir) / "config.json")
    else:
        payload = _default_config_payload()
    return omnivoice_export_config_from_payload(payload)


def omnivoice_export_config_from_payload(payload: JsonObject) -> OmniVoiceExportConfig:
    llm_config = _expect_mapping(payload["llm_config"], "llm_config")
    return OmniVoiceExportConfig(
        audio_vocab_size=_int_value(payload, "audio_vocab_size"),
        audio_mask_id=_int_value(payload, "audio_mask_id"),
        num_audio_codebook=_int_value(payload, "num_audio_codebook"),
        llm_hidden_size=_int_value(llm_config, "hidden_size"),
        llm_intermediate_size=_int_value(llm_config, "intermediate_size"),
        llm_vocab_size=_int_value(llm_config, "vocab_size"),
        llm_num_hidden_layers=_int_value(llm_config, "num_hidden_layers"),
        llm_num_attention_heads=_int_value(llm_config, "num_attention_heads"),
        llm_num_key_value_heads=_int_value(llm_config, "num_key_value_heads"),
        llm_head_dim=_int_value(llm_config, "head_dim"),
    )


def reflect_omnivoice_source(config: OmniVoiceExportConfig) -> TorchModuleReflection:
    stream = io.StringIO()
    with _suppress_native_output(), redirect_stdout(stream), redirect_stderr(stream):
        OmniVoice = _load_source_model_module().OmniVoice
        source_config = omnivoice_source_config_from_export_config(config)
        model = instantiate_torch_module_on_meta(lambda: OmniVoice(source_config))
    return reflect_torch_module(model)


def omnivoice_source_config_from_export_config(config: OmniVoiceExportConfig) -> Any:
    OmniVoiceConfig = _load_source_model_module().OmniVoiceConfig
    return OmniVoiceConfig(
        audio_vocab_size=config.audio_vocab_size,
        audio_mask_id=config.audio_mask_id,
        num_audio_codebook=config.num_audio_codebook,
        llm_config={
            "model_type": "qwen3",
            "hidden_size": config.llm_hidden_size,
            "intermediate_size": config.llm_intermediate_size,
            "vocab_size": config.llm_vocab_size,
            "num_hidden_layers": config.llm_num_hidden_layers,
            "num_attention_heads": config.llm_num_attention_heads,
            "num_key_value_heads": config.llm_num_key_value_heads,
            "head_dim": config.llm_head_dim,
        },
    )


def build_omnivoice_pattern_context(
    *,
    config: OmniVoiceExportConfig,
    reflection: TorchModuleReflection | None,
) -> dict[str, object]:
    if reflection is not None:
        _validate_reflected_source(reflection)
    return {
        "defaults": config.to_template_defaults(),
        "text_parameter_fields": _text_parameter_fields(),
        "text_layer_fields": _text_layer_fields(),
        "audio_codec_fields": _audio_codec_fields(),
        "audio_codec_layer_fields": _audio_codec_layer_fields(),
        "input_embeddings_ops": _input_embeddings_ops(config),
        "text_prefill_ops": _text_prefill_ops(),
        "iterative_decode_ops": _iterative_decode_ops(),
        "token_select_ops": _token_select_ops(),
        "audio_head_ops": _audio_head_ops(),
        "audio_codec_decode_ops": _audio_codec_decode_ops(),
        "reference_prompt_ops": _reference_prompt_ops(),
        "shader_modules": _shader_modules(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PACKAGE_DIR,
        help="Directory for the generated package files.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument(
        "--no-reflect",
        action="store_true",
        help="Skip PyTorch meta-device reflection and render from config/op declarations only.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = export_omnivoice_package(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        config_json=args.config_json,
        reflect_source=not args.no_reflect,
        check=args.check,
        dry_run=args.dry_run,
    )
    action = "checked" if args.check else "would write" if result.dry_run else "wrote"
    print(
        f"{action} {len(result.written)} files, "
        f"{len(result.unchanged)} unchanged under {args.output_dir}"
    )
    return 0


def _render_omnivoice_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    files = [
        renderer.render("omnivoice/__init__.py.j2", "__init__.py", **context),
    ]
    files.extend(_render_tensor_files(renderer, context))
    files.extend(_render_shader_files(renderer, context))
    files.extend(_render_frame_files(renderer, context))
    return files


def _render_tensor_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    return [
        renderer.render("omnivoice/tensors/__init__.py.j2", "tensors/__init__.py", **context),
        renderer.render("omnivoice/tensors/inference.py.j2", "tensors/inference.py", **context),
        renderer.render("omnivoice/tensors/text_layer.py.j2", "tensors/text_layer.py", **context),
        renderer.render("omnivoice/tensors/text.py.j2", "tensors/text.py", **context),
        renderer.render(
            "omnivoice/tensors/audio_codec_layer.py.j2",
            "tensors/audio_codec_layer.py",
            **context,
        ),
        renderer.render("omnivoice/tensors/audio_codec.py.j2", "tensors/audio_codec.py", **context),
    ]


def _render_shader_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    files = [
        renderer.render("omnivoice/shaders/__init__.py.j2", "shaders/__init__.py", **context),
    ]
    shader_modules = context["shader_modules"]
    if not isinstance(shader_modules, tuple):
        raise TypeError("shader_modules must be a tuple")
    for module in shader_modules:
        if not isinstance(module, ShaderModulePattern):
            raise TypeError(f"shader_modules entry must be ShaderModulePattern, got {type(module)}")
        files.append(
            renderer.render(
                "omnivoice/shaders/shader_module.py.j2",
                f"shaders/{module.module}.py",
                module=module,
                **context,
            )
        )
    return files


def _render_frame_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    return [
        renderer.render("omnivoice/_frame.py.j2", "_frame.py", **context),
        renderer.render("omnivoice/reference_prompt.py.j2", "reference_prompt.py", **context),
        renderer.render("omnivoice/input_embeddings.py.j2", "input_embeddings.py", **context),
        renderer.render("omnivoice/text_prefill.py.j2", "text_prefill.py", **context),
        renderer.render("omnivoice/audio_head.py.j2", "audio_head.py", **context),
        renderer.render("omnivoice/token_select.py.j2", "token_select.py", **context),
        renderer.render("omnivoice/iterative_decode.py.j2", "iterative_decode.py", **context),
        renderer.render("omnivoice/audio_codec_decode.py.j2", "audio_codec_decode.py", **context),
        renderer.render("omnivoice/execution.py.j2", "execution.py", **context),
    ]


def _validate_reflected_source(reflection: TorchModuleReflection) -> None:
    reflection.require_module_type("llm")
    reflection.require_parameter("audio_embeddings.weight")
    reflection.require_parameter("audio_heads.weight")
    reflection.require_parameter("llm.norm.weight")
    reflection.require_parameter("llm.layers.0.self_attn.q_norm.weight")
    reflection.require_parameter("llm.layers.0.self_attn.k_norm.weight")


def _text_parameter_fields() -> tuple[TensorFieldPattern, ...]:
    return (
        TensorFieldPattern("embed_tokens_weight", "llm.embed_tokens.weight"),
        TensorFieldPattern("audio_embeddings_weight", "audio_embeddings.weight"),
        TensorFieldPattern("codebook_layer_offsets", "codebook_layer_offsets"),
        TensorFieldPattern("norm_weight", "llm.norm.weight"),
        TensorFieldPattern("audio_heads_weight", "audio_heads.weight"),
    )


def _text_layer_fields() -> tuple[TensorFieldPattern, ...]:
    return (
        TensorFieldPattern("input_layernorm_weight", "input_layernorm.weight"),
        TensorFieldPattern("post_attention_layernorm_weight", "post_attention_layernorm.weight"),
        TensorFieldPattern("q_norm_weight", "self_attn.q_norm.weight"),
        TensorFieldPattern("k_norm_weight", "self_attn.k_norm.weight"),
        TensorFieldPattern("q_proj_weight", "self_attn.q_proj.weight"),
        TensorFieldPattern("k_proj_weight", "self_attn.k_proj.weight"),
        TensorFieldPattern("v_proj_weight", "self_attn.v_proj.weight"),
        TensorFieldPattern("o_proj_weight", "self_attn.o_proj.weight"),
        TensorFieldPattern("gate_proj_weight", "mlp.gate_proj.weight"),
        TensorFieldPattern("up_proj_weight", "mlp.up_proj.weight"),
        TensorFieldPattern("down_proj_weight", "mlp.down_proj.weight"),
        TensorFieldPattern("input_layernorm"),
        TensorFieldPattern("q_proj"),
        TensorFieldPattern("k_proj"),
        TensorFieldPattern("v_proj"),
        TensorFieldPattern("q_normed"),
        TensorFieldPattern("k_normed"),
        TensorFieldPattern("q_roped"),
        TensorFieldPattern("k_roped"),
        TensorFieldPattern("key_cache"),
        TensorFieldPattern("value_cache"),
        TensorFieldPattern("attention"),
        TensorFieldPattern("o_proj"),
        TensorFieldPattern("attn_residual"),
        TensorFieldPattern("post_attention_layernorm"),
        TensorFieldPattern("gate_proj"),
        TensorFieldPattern("up_proj"),
        TensorFieldPattern("swiglu"),
        TensorFieldPattern("down_proj"),
        TensorFieldPattern("output"),
    )


def _audio_codec_fields() -> tuple[TensorFieldPattern, ...]:
    return (
        TensorFieldPattern("audio_tokens"),
        TensorFieldPattern("quantizer_embeddings"),
        TensorFieldPattern("decoder_input"),
        TensorFieldPattern("decoder_layers"),
        TensorFieldPattern("decoder_hidden"),
        TensorFieldPattern("projected_waveform"),
        TensorFieldPattern("audio_waveform"),
    )


def _audio_codec_layer_fields() -> tuple[TensorFieldPattern, ...]:
    return (
        TensorFieldPattern("conv1_weight"),
        TensorFieldPattern("conv1_bias"),
        TensorFieldPattern("conv7_weight"),
        TensorFieldPattern("conv7_bias"),
        TensorFieldPattern("deconv_weight"),
        TensorFieldPattern("deconv_bias"),
        TensorFieldPattern("decoder_hidden"),
        TensorFieldPattern("snake_activation"),
        TensorFieldPattern("decoder_residual"),
        TensorFieldPattern("output"),
    )


def _input_embeddings_ops(config: OmniVoiceExportConfig) -> tuple[TorchOpPattern, ...]:
    exported_program = _export_input_embeddings_program(config)
    return torch_ops_from_exported_program(
        exported_program,
        tensor_name_map={
            "p_embed_tokens_weight": "embed_tokens_weight",
            "p_audio_embeddings_weight": "audio_embeddings_weight",
            "b_codebook_layer_offsets": "codebook_layer_offsets",
            "input_ids": "input_ids",
            "audio_mask": "audio_mask",
            "select": "text_token_ids",
            "embedding": "text_embeds",
            "unsqueeze": "audio_mask_for_shift",
            "mul": "masked_audio_ids",
            "view": "codebook_layer_offsets_view",
            "add": "shifted_ids",
            "embedding_1": "audio_embedding_values",
            "sum_1": "audio_embeds",
            "unsqueeze_1": "audio_mask_expanded",
            "where": "inputs_embeds",
        },
    )


def _export_input_embeddings_program(config: OmniVoiceExportConfig) -> object:
    import torch
    from torch import nn

    class InputEmbeddingsModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(
                config.llm_vocab_size,
                config.llm_hidden_size,
                device="meta",
            )
            self.audio_embeddings = nn.Embedding(
                config.num_audio_codebook * config.audio_vocab_size,
                config.llm_hidden_size,
                device="meta",
            )
            self.register_buffer(
                "codebook_layer_offsets",
                torch.arange(config.num_audio_codebook, device="meta") * config.audio_vocab_size,
            )

        def forward(self, input_ids: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
            text_embeds = self.embed_tokens(input_ids[:, 0, :])
            codebook_layer_offsets = torch.ops.aten.view.default(
                self.codebook_layer_offsets,
                [1, -1, 1],
            )
            shifted_ids = (
                input_ids * audio_mask.unsqueeze(1)
            ) + codebook_layer_offsets
            audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
            return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    example_input_ids = torch.zeros(
        (1, config.num_audio_codebook, 4),
        dtype=torch.long,
        device="meta",
    )
    example_audio_mask = torch.zeros((1, 4), dtype=torch.bool, device="meta")
    return export_torch_program(
        InputEmbeddingsModule().eval(),
        (example_input_ids, example_audio_mask),
        strict=False,
    )


def _text_prefill_ops() -> tuple[TorchOpPattern, ...]:
    return (
        TorchOpPattern("input_embeddings", ("input_ids", "audio_mask"), ("inputs_embeds",)),
        TorchOpPattern("llm_layer_loop", ("inputs_embeds", "attention_mask"), ("hidden_states",)),
        TorchOpPattern("final_norm", ("hidden_states",), ("hidden_states",)),
    )


def _audio_head_ops() -> tuple[TorchOpPattern, ...]:
    return (
        TorchOpPattern(
            "audio_head_projection", ("hidden_states", "audio_heads_weight"), ("audio_logits",)
        ),
        TorchOpPattern("reshape_codebooks", ("audio_logits",), ("audio_logits",)),
    )


def _token_select_ops() -> tuple[TorchOpPattern, ...]:
    return (
        TorchOpPattern(
            "classifier_free_guidance", ("cond_logits", "uncond_logits"), ("guided_logits",)
        ),
        TorchOpPattern("mask_forbidden_ids", ("guided_logits",), ("guided_logits",)),
        TorchOpPattern(
            "codebook_argmax", ("guided_logits",), ("pred_tokens", "confidence_scores")
        ),
        TorchOpPattern(
            "select_positions", ("confidence_scores", "current_tokens"), ("selected_positions",)
        ),
        TorchOpPattern(
            "apply_selected_tokens", ("pred_tokens", "selected_positions"), ("updated_tokens",)
        ),
    )


def _iterative_decode_ops() -> tuple[TorchOpPattern, ...]:
    return (
        TorchOpPattern("initialize_masked_tokens", ("target_lens",), ("tokens",)),
        TorchOpPattern(
            "for_each_decode_step", ("batch_input_ids", "batch_audio_mask"), ("audio_logits",)
        ),
        TorchOpPattern("token_select", ("audio_logits", "tokens"), ("tokens",)),
        TorchOpPattern("update_conditioned_inputs", ("tokens",), ("batch_input_ids",)),
    )


def _audio_codec_decode_ops() -> tuple[TorchOpPattern, ...]:
    return (
        TorchOpPattern(
            "quantizer_embed_sum", ("audio_tokens", "quantizer_embeddings"), ("decoder_hidden",)
        ),
        TorchOpPattern("decoder_conv_stack", ("decoder_hidden",), ("decoder_hidden",)),
        TorchOpPattern("project_out", ("decoder_hidden",), ("audio_waveform",)),
    )


def _reference_prompt_ops() -> tuple[TorchOpPattern, ...]:
    return (
        TorchOpPattern(
            "load_or_resample_reference_audio", ("ref_audio",), ("reference_waveform",)
        ),
        TorchOpPattern("audio_tokenizer_encode", ("reference_waveform",), ("reference_tokens",)),
        TorchOpPattern("normalize_reference_text", ("ref_text",), ("reference_text",)),
    )


def _shader_modules() -> tuple[ShaderModulePattern, ...]:
    return (
        _inline_shader_module(
            "OMNIVOICE_ATEN_SELECT_INT_I64",
            "aten_select_int_i64",
            family="aten",
            source=_aten_select_int_i64_source(),
            variant_body=_aten_select_int_i64_variant_body(),
        ),
        _inline_shader_module(
            "OMNIVOICE_ATEN_EMBEDDING_F32",
            "aten_embedding_f32",
            family="omnivoice.text",
            source=_aten_embedding_f32_source(),
            variant_body=_aten_embedding_f32_variant_body(),
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_EMBEDDING_SUM_F32",
            "audio_embedding_sum_f32.glsl",
            family="omnivoice.text",
            source=_audio_embedding_sum_source(),
            variant_body=_audio_embedding_sum_variant_body(),
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_HEAD_MATMUL_F16_F32_F16ACC_LARGE",
            "audio_head_matmul_f16_f32_f16acc_large.glsl",
            family="omnivoice.text",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_HEAD_MATMUL_F16_F32_F16ACC_SMALL",
            "audio_head_matmul_f16_f32_f16acc_small.glsl",
            family="omnivoice.text",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_HEAD_MAT_VEC_F16_F32_F32",
            "audio_head_mat_vec_f16_f32_f32.glsl",
            family="omnivoice.text",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_HEAD_SCALAR_F16_F32_F32",
            "audio_head_scalar_f16_f32_f32.glsl",
            family="omnivoice.text",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_HEAD_ROUND_F32_TO_F16_F32",
            "audio_head_round_f32_to_f16_f32.glsl",
            family="omnivoice.text",
        ),
        _single_shader_module(
            "OMNIVOICE_CODEBOOK_ARGMAX_F32",
            "codebook_argmax_f32.glsl",
            family="omnivoice.token_select",
            variant_body=_token_select_codebook_argmax_variant_body(),
        ),
        _single_shader_module(
            "OMNIVOICE_CODEBOOK_ARGMAX_SCORES_F32",
            "codebook_argmax_scores_f32.glsl",
            family="omnivoice.token_select",
            variant_body=_token_select_codebook_argmax_scores_variant_body(),
        ),
        _single_shader_module(
            "OMNIVOICE_ARGMAX_SELECT_APPLY_FUSED_L",
            "argmax_select_apply_fused_l.glsl",
            family="omnivoice.token_select",
            variant_body=_token_select_argmax_select_apply_variant_body(
                shader_name="omnivoice_argmax_select_apply_fused_l",
                class_name="OmniVoiceArgmaxSelectApplyFusedLProgram",
                local_size_x=256,
            ),
        ),
        _single_shader_module(
            "OMNIVOICE_ARGMAX_SELECT_APPLY_FUSED_S",
            "argmax_select_apply_fused_s.glsl",
            family="omnivoice.token_select",
            variant_body=_token_select_argmax_select_apply_variant_body(
                shader_name="omnivoice_argmax_select_apply_fused_s",
                class_name="OmniVoiceArgmaxSelectApplyFusedSProgram",
                local_size_x=128,
            ),
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_QUANTIZER_EMBED_SUM_F32",
            "audio_codec_decoder_quantizer_embed_sum_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_QUANTIZER_EMBED_PROJECT_OUT_SUM_F32",
            "audio_codec_decoder_quantizer_embed_project_out_sum_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_QUANTIZER_EMBED_PROJECT_OUT_ARGMAX_ALL_CODEBOOKS_F32",
            "audio_codec_decoder_quantizer_embed_project_out_argmax_all_codebooks_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_PROJECT_OUT_ARGMAX_ALL_CODEBOOKS_F32",
            "audio_codec_decoder_project_out_argmax_all_codebooks_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_CONV1D_K1_F32",
            "audio_codec_decoder_conv1d_k1_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_CONV1D_K7_F32",
            "audio_codec_decoder_conv1d_k7_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_DECONV1D_BLOCK0_F32",
            "audio_codec_decoder_deconv1d_block0_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_F32",
            "audio_codec_decoder_snake_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_CONV1D_K1_RESIDUAL_ADD_F32",
            "audio_codec_decoder_snake_conv1d_k1_residual_add_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_CONV1D_K7_D1_F32",
            "audio_codec_decoder_snake_conv1d_k7_d1_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_CONV1D_K7_D3_F32",
            "audio_codec_decoder_snake_conv1d_k7_d3_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_CONV1D_K7_D9_F32",
            "audio_codec_decoder_snake_conv1d_k7_d9_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_DECONV1D_BLOCK0_F32",
            "audio_codec_decoder_snake_deconv1d_block0_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_DECONV1D_BLOCK1_F32",
            "audio_codec_decoder_snake_deconv1d_block1_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_DECONV1D_BLOCK2_F32",
            "audio_codec_decoder_snake_deconv1d_block2_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_DECONV1D_BLOCK3_F32",
            "audio_codec_decoder_snake_deconv1d_block3_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_SNAKE_DECONV1D_BLOCK4_F32",
            "audio_codec_decoder_snake_deconv1d_block4_f32.glsl",
            family="omnivoice.audio_codec",
        ),
        _single_shader_module(
            "OMNIVOICE_AUDIO_CODEC_DECODER_RESIDUAL_ADD_F32",
            "audio_codec_decoder_residual_add_f32.glsl",
            family="omnivoice.audio_codec",
        ),
    )


def _single_shader_module(
    constant: str,
    source_file: str,
    *,
    family: str,
    source: str | None = None,
    variant_body: str = "",
) -> ShaderModulePattern:
    module = source_file.removesuffix(".glsl")
    return ShaderModulePattern(
        module,
        module.replace("_", " "),
        (
            _shader_variant(
                constant,
                source_file,
                family=family,
                source=source,
                variant_body=variant_body,
            ),
        ),
    )


def _inline_shader_module(
    constant: str,
    module: str,
    *,
    family: str,
    source: str,
    variant_body: str,
) -> ShaderModulePattern:
    return ShaderModulePattern(
        module,
        module.replace("_", " "),
        (
            _shader_variant(
                constant,
                f"{module}.glsl",
                family=family,
                source=source,
                variant_body=variant_body,
            ),
        ),
    )


def _shader_variant(
    constant: str,
    source_file: str,
    *,
    family: str,
    source: str | None = None,
    variant_body: str = "",
) -> ShaderPattern:
    shader_name = f"omnivoice_{source_file.removesuffix('.glsl')}"
    return ShaderPattern(
        constant=constant,
        shader_name=shader_name,
        class_name=_shader_class_name(shader_name),
        family=family,
        source_file=source_file,
        source=_load_imported_glsl_source(source_file) if source is None else source,
        variant_body=variant_body,
    )


def _aten_select_int_i64_variant_body() -> str:
    return """
contract=ShaderContract(
    class_name="OmniVoiceAtenSelectIntI64Program",
    shader_name="omnivoice_aten_select_int_i64",
    fields=(
        TensorFieldSpec(
            name="x",
            io_kind=IOKind.INPUT,
            role="x",
            contract=TensorContract(dtype="int64", shape=("B", "C", "T")),
        ),
        TensorFieldSpec(
            name="output",
            io_kind=IOKind.OUTPUT,
            role="output",
            contract=TensorContract(dtype="int64", shape=("B", "T")),
        ),
    ),
    push_constants=PushConstantSpec(
        size=8,
        fields=(
            PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
            PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
        ),
    ),
    dispatch=(ceil_div("T", 256), "B", 1),
),
execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
""".lstrip()


def _aten_embedding_f32_variant_body() -> str:
    return """
contract=ShaderContract(
    class_name="OmniVoiceAtenEmbeddingF32Program",
    shader_name="omnivoice_aten_embedding_f32",
    fields=(
        TensorFieldSpec(
            name="weight",
            io_kind=IOKind.INPUT,
            role="weight",
            contract=TensorContract(dtype="float32", shape=("V", "H")),
        ),
        TensorFieldSpec(
            name="indices",
            io_kind=IOKind.INPUT,
            role="indices",
            contract=TensorContract(dtype="int64", shape=("B", "T")),
        ),
        TensorFieldSpec(
            name="output",
            io_kind=IOKind.OUTPUT,
            role="output",
            contract=TensorContract(dtype="float32", shape=("B", "T", "H")),
        ),
    ),
    push_constants=PushConstantSpec(
        size=12,
        fields=(
            PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
            PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            PushConstantFieldSpec("V", PushConstantType.UINT32, 8, "V"),
        ),
    ),
    dispatch=(ceil_div("H", 256), "T", "B"),
),
execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
""".lstrip()


def _audio_embedding_sum_variant_body() -> str:
    return """
contract=ShaderContract(
    class_name="OmniVoiceAudioEmbeddingSumF32Program",
    shader_name="omnivoice_audio_embedding_sum_f32",
    fields=(
        TensorFieldSpec(
            name="input_ids",
            io_kind=IOKind.INPUT,
            role="input_ids",
            contract=TensorContract(dtype="int64", shape=("B", "C", "T")),
        ),
        TensorFieldSpec(
            name="audio_mask",
            io_kind=IOKind.INPUT,
            role="audio_mask",
            contract=TensorContract(dtype="bool", shape=("B", "T")),
        ),
        TensorFieldSpec(
            name="codebook_layer_offsets",
            io_kind=IOKind.INPUT,
            role="codebook_layer_offsets",
            contract=TensorContract(dtype="int64", shape=("C",)),
        ),
        TensorFieldSpec(
            name="audio_embeddings_weight",
            io_kind=IOKind.INPUT,
            role="audio_embeddings_weight",
            contract=TensorContract(dtype="float32", shape=("E", "H")),
        ),
        TensorFieldSpec(
            name="output",
            io_kind=IOKind.OUTPUT,
            role="output",
            contract=TensorContract(dtype="float32", shape=("B", "T", "H")),
        ),
    ),
    push_constants=PushConstantSpec(
        size=16,
        fields=(
            PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
            PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
            PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "H"),
            PushConstantFieldSpec("E", PushConstantType.UINT32, 12, "E"),
        ),
    ),
    dispatch=(ceil_div("H", 256), "T", "B"),
),
execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
""".lstrip()


def _token_select_codebook_argmax_variant_body() -> str:
    return render_shader_contract_variant_body(
        class_name="OmniVoiceCodebookArgmaxF32Program",
        shader_name="omnivoice_codebook_argmax_f32",
        tensor_fields=(
            TensorFieldDecl("output_ids", "OUTPUT", "output_ids", "int32", ("B", "C", "S")),
            TensorFieldDecl("logits", "INPUT", "logits", "float32", ("B", "C", "S", "V")),
            TensorFieldDecl("codebook_offsets", "INPUT", "codebook_offsets", "int32", ("C",)),
        ),
        dispatch=("C", "S", "B"),
        params_fields=(
            ParamsFieldDecl("steps", "UINT32", 0, "S"),
            ParamsFieldDecl("batches", "UINT32", 4, "B"),
            ParamsFieldDecl("vocab", "UINT32", 8, "V"),
            ParamsFieldDecl("codebooks", "UINT32", 12, "C"),
        ),
        params_size=16,
        params_binding_index=3,
    )


def _token_select_codebook_argmax_scores_variant_body() -> str:
    return render_shader_contract_variant_body(
        class_name="OmniVoiceCodebookArgmaxScoresF32Program",
        shader_name="omnivoice_codebook_argmax_scores_f32",
        tensor_fields=(
            TensorFieldDecl("output_ids", "OUTPUT", "output_ids", "int32", ("B", "C", "S")),
            TensorFieldDecl("output_scores", "OUTPUT", "output_scores", "float32", ("B", "C", "S")),
            TensorFieldDecl("logits", "INPUT", "logits", "float32", ("B", "C", "S", "V")),
            TensorFieldDecl("codebook_offsets", "INPUT", "codebook_offsets", "int32", ("C",)),
        ),
        dispatch=("C", "S", "B"),
        params_fields=(
            ParamsFieldDecl("steps", "UINT32", 0, "S"),
            ParamsFieldDecl("batches", "UINT32", 4, "B"),
            ParamsFieldDecl("vocab", "UINT32", 8, "V"),
            ParamsFieldDecl("codebooks", "UINT32", 12, "C"),
        ),
        params_size=16,
        params_binding_index=4,
    )


def _token_select_argmax_select_apply_variant_body(
    *,
    shader_name: str,
    class_name: str,
    local_size_x: int,
) -> str:
    del local_size_x
    return render_shader_contract_variant_body(
        class_name=class_name,
        shader_name=shader_name,
        tensor_fields=(
            TensorFieldDecl("output_updated_ids", "OUTPUT", "output_updated_ids", "int32", ("B", "C", "S")),
            TensorFieldDecl("output_selected_flat_index", "OUTPUT", "output_selected_flat_index", "int32", ("B",)),
            TensorFieldDecl("output_selected_score", "OUTPUT", "output_selected_score", "float32", ("B",)),
            TensorFieldDecl(
                "output_selected_candidate_id", "OUTPUT", "output_selected_candidate_id", "int32", ("B",)
            ),
            TensorFieldDecl("logits", "INPUT", "logits", "float32", ("B", "C", "S", "V")),
            TensorFieldDecl("codebook_offsets", "INPUT", "codebook_offsets", "int32", ("C",)),
            TensorFieldDecl("penalty", "INPUT", "penalty", "float32", ("B", "C")),
            TensorFieldDecl("current_ids", "INPUT", "current_ids", "int32", ("B", "C", "S")),
        ),
        dispatch=(1, 1, "B"),
        params_fields=(
            ParamsFieldDecl("steps", "UINT32", 0, "S"),
            ParamsFieldDecl("codebooks", "UINT32", 4, "C"),
            ParamsFieldDecl("batches", "UINT32", 8, "B"),
            ParamsFieldDecl("vocab", "UINT32", 12, "V"),
        ),
        params_size=16,
        params_binding_index=8,
    )


def _aten_select_int_i64_source() -> str:
    return """
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly InputBuffer {
    int64_t x[];
};

layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer {
    int64_t output_values[];
};

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint token = gl_GlobalInvocationID.x;
    const uint batch = gl_GlobalInvocationID.y;
    if (token >= pc.T) {
        return;
    }
    output_values[batch * pc.T + token] = x[(batch * pc.C) * pc.T + token];
}
""".lstrip()


def _aten_embedding_f32_source() -> str:
    return """
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer {
    float weight[];
};

layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer {
    int64_t indices[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint T;
    uint H;
    uint V;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint batch = gl_GlobalInvocationID.z;
    if (h >= pc.H || token >= pc.T) {
        return;
    }

    const int64_t token_id = indices[batch * pc.T + token];
    const uint out_index = (batch * pc.T + token) * pc.H + h;
    if (token_id >= int64_t(0) && token_id < int64_t(pc.V)) {
        output_values[out_index] = weight[uint(token_id) * pc.H + h];
    } else {
        output_values[out_index] = 0.0;
    }
}
""".lstrip()


def _audio_embedding_sum_source() -> str:
    return """
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly InputIdsBuffer {
    int64_t input_ids[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioMaskBuffer {
    bool audio_mask[];
};

layout(set = 0, binding = 2) buffer restrict readonly CodebookOffsetsBuffer {
    int64_t codebook_layer_offsets[];
};

layout(set = 0, binding = 3) buffer restrict readonly AudioEmbeddingsBuffer {
    float audio_embeddings_weight[];
};

layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
    uint H;
    uint E;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint batch = gl_GlobalInvocationID.z;
    if (h >= pc.H || token >= pc.T) {
        return;
    }

    const int64_t mask = audio_mask[batch * pc.T + token] ? int64_t(1) : int64_t(0);
    float acc = 0.0;
    for (uint codebook = 0u; codebook < pc.C; ++codebook) {
        const uint id_index = (batch * pc.C + codebook) * pc.T + token;
        const int64_t shifted_id = input_ids[id_index] * mask + codebook_layer_offsets[codebook];
        if (shifted_id >= int64_t(0) && shifted_id < int64_t(pc.E)) {
            acc += audio_embeddings_weight[uint(shifted_id) * pc.H + h];
        }
    }
    output_values[(batch * pc.T + token) * pc.H + h] = acc;
}
""".lstrip()


def _load_imported_glsl_source(source_file: str) -> str:
    return (_IMPORTED_GLSL_DIR / source_file).read_text(encoding="utf-8").rstrip() + "\n"


def _shader_class_name(shader_name: str) -> str:
    return "".join(_shader_class_name_part(part) for part in shader_name.split("_")) + "Program"


def _shader_class_name_part(part: str) -> str:
    if part == "omnivoice":
        return "OmniVoice"
    if part.startswith("f") and part[1:].isdigit():
        return part.upper()
    if part.startswith("f") and part.endswith("acc") and part[1:-3].isdigit():
        return f"F{part[1:-3]}Acc"
    if part.startswith("k") and part[1:].isdigit():
        return part.upper()
    return part.capitalize()


def _default_config_payload() -> JsonObject:
    cached = _cached_omnivoice_config()
    if cached is not None:
        return cached
    return {
        "model_type": "omnivoice",
        "audio_vocab_size": 1025,
        "audio_mask_id": 1024,
        "num_audio_codebook": 8,
        "llm_config": {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "vocab_size": 151676,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
    }


def _cached_omnivoice_config() -> JsonObject | None:
    candidates = sorted(
        Path.home().glob(".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/*/config.json")
    )
    if not candidates:
        return None
    return _load_json(candidates[-1])


def _load_source_model_module() -> ModuleType:
    src_root = Path(__file__).resolve().parents[2]
    path = src_root / "omnivoice/models/omnivoice.py"
    spec = importlib.util.spec_from_file_location("_torch2vk_omnivoice_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OmniVoice model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> JsonObject:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object in {path}, got {type(value).__name__}")
    return value


def _expect_mapping(value: object, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _int_value(config: JsonObject, key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _remove_stale_files(
    output_dir: str | Path,
    *,
    check: bool,
    dry_run: bool,
) -> None:
    remove_stale_files(output_dir, _STALE_FILES, check=check, dry_run=dry_run)


@contextmanager
def _suppress_native_output() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


if __name__ == "__main__":
    raise SystemExit(main())
