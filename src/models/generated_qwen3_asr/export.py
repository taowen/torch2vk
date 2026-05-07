"""Regenerate the generated Qwen3-ASR scaffold package.

This module owns the Qwen3-ASR-specific export recipe. The shared
``torch2vk.export`` package intentionally stays generic: template rendering,
file writes, and PyTorch module reflection only.
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

import torch2vk.export.shaders as export_shaders
from torch2vk.export.reflection import (
    TorchModuleReflection,
    instantiate_torch_module_on_meta,
    reflect_torch_module,
)
from torch2vk.export.torch_ops import TensorFieldPattern
from torch2vk.export.logical_tensor_codegen import render_logical_tensor_helpers_file
from torch2vk.export.package_codegen import PythonImportDecl, render_python_init_file
from torch2vk.export.tensor_scaffold_codegen import (
    LoweredOpContract,
    TensorDataclassDecl,
    TensorDataclassFieldDecl,
    logical_tensor_dataclass_from_patterns,
    render_parameter_fields_constant,
    render_tensor_dataclass,
    render_tensor_dataclasses,
    tensor_scaffold_fields_from_lowered_ops,
)
from torch2vk.export.frame_dispatch_codegen import FrameSpec, render_frame_module
from torch2vk.export.writer import (
    ExportWriteResult,
    RenderedFile,
    TemplateRenderer,
    format_python_source,
    remove_stale_files,
    write_rendered_files,
)
from torch2vk.runtime.shader import ShaderVariant


JsonObject = Mapping[str, object]

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATE_ROOT = _PACKAGE_DIR / "export_templates"
_STALE_FILES = (
    Path("frames.py"),
    Path("tensors.py"),
    Path("shaders/registry.py"),
    Path("tensors/single_tensor_preview.py"),
    Path("tensors/exported_logical_tensors.py"),
    Path("tensors/exported/__init__.py"),
    Path("tensors/exported/audio_attention.py"),
    Path("tensors/exported/audio_encoder_layer.py"),
    Path("tensors/exported/text_attention_decode.py"),
    Path("tensors/exported/text_attention_prefill.py"),
    Path("tensors/exported/text_decoder_layer_decode.py"),
    Path("tensors/exported/text_decoder_layer_prefill.py"),
    Path("tensors/exported/text_mlp.py"),
    Path("tensors/exported/text_rms_norm.py"),
)
_SOURCE_SHADER_DIR = _PACKAGE_DIR.parent / "qwen3_asr" / "shaders"


@dataclass(frozen=True, slots=True)
class Qwen3AsrExportConfig:
    audio_num_mel_bins: int
    audio_hidden_size: int
    audio_output_size: int
    audio_downsample_hidden_size: int
    audio_encoder_layers: int
    audio_encoder_attention_heads: int
    audio_encoder_ffn_dim: int
    audio_n_window: int
    audio_n_window_infer: int
    text_hidden_size: int
    text_intermediate_size: int
    text_vocab_size: int
    text_decoder_layers: int
    text_num_attention_heads: int
    text_num_key_value_heads: int
    text_head_dim: int
    text_rope_theta: float
    text_mrope_section: tuple[int, ...]
    audio_token_id: int
    eos_token_ids: tuple[int, ...] = (151645, 151643)

    def to_template_defaults(self) -> dict[str, object]:
        value = asdict(self)
        value["text_mrope_section"] = tuple(self.text_mrope_section)
        value["eos_token_ids"] = tuple(self.eos_token_ids)
        return value


@dataclass(frozen=True, slots=True)
class ShaderModulePattern:
    module: str
    constants: tuple[str, ...]
    source: str


def _op(
    target: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    note: str = "",
    **extra: object,
) -> object:
    return {
        "target": target,
        "inputs": inputs,
        "outputs": outputs,
        "note": note,
        **extra,
    }


def _lowered_op_contracts(ops: Sequence[object]) -> tuple[LoweredOpContract, ...]:
    contracts: list[LoweredOpContract] = []
    for op in ops:
        if not isinstance(op, Mapping):
            raise TypeError(f"lowered op must be a mapping, got {type(op).__name__}")
        contracts.append(
            (
                str(op["target"]),
                tuple(str(item) for item in _expect_tuple(op["inputs"], "op.inputs")),
                tuple(str(item) for item in _expect_tuple(op["outputs"], "op.outputs")),
            )
        )
    return tuple(contracts)


def export_qwen3_asr_package(
    *,
    output_dir: str | Path = _PACKAGE_DIR,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
    reflect_source: bool = True,
    check: bool = False,
    dry_run: bool = False,
) -> ExportWriteResult:
    config = load_qwen3_asr_export_config(
        model_dir=model_dir,
        config_json=config_json,
    )
    reflection = reflect_qwen3_asr_source(config) if reflect_source else None
    context = build_qwen3_asr_pattern_context(config=config, reflection=reflection)
    renderer = TemplateRenderer(_TEMPLATE_ROOT)
    files = _render_qwen3_asr_files(renderer, context)
    result = write_rendered_files(files, output_dir, check=check, dry_run=dry_run)
    _remove_stale_files(output_dir, check=check, dry_run=dry_run)
    return result


def load_qwen3_asr_export_config(
    *,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
) -> Qwen3AsrExportConfig:
    if config_json is not None:
        payload = _load_json(Path(config_json))
    elif model_dir is not None:
        payload = _load_json(Path(model_dir) / "config.json")
    else:
        payload = _source_default_config_payload()
    return qwen3_asr_export_config_from_payload(payload)


def qwen3_asr_export_config_from_payload(payload: JsonObject) -> Qwen3AsrExportConfig:
    thinker_config = _expect_mapping(payload.get("thinker_config", payload), "thinker_config")
    audio_config = _expect_mapping(thinker_config["audio_config"], "audio_config")
    text_config = _expect_mapping(thinker_config["text_config"], "text_config")
    rope_scaling = text_config.get("rope_scaling")
    mrope_section: tuple[int, ...] = (24, 20, 20)
    if isinstance(rope_scaling, Mapping):
        mrope_section = _int_tuple(
            rope_scaling.get("mrope_section", mrope_section),
            "rope_scaling.mrope_section",
        )
    return Qwen3AsrExportConfig(
        audio_num_mel_bins=_int_value(audio_config, "num_mel_bins"),
        audio_hidden_size=_int_value(audio_config, "d_model"),
        audio_output_size=_int_value(audio_config, "output_dim"),
        audio_downsample_hidden_size=_int_value(audio_config, "downsample_hidden_size"),
        audio_encoder_layers=_int_value(audio_config, "encoder_layers"),
        audio_encoder_attention_heads=_int_value(audio_config, "encoder_attention_heads"),
        audio_encoder_ffn_dim=_int_value(audio_config, "encoder_ffn_dim"),
        audio_n_window=_int_value(audio_config, "n_window"),
        audio_n_window_infer=_int_value(audio_config, "n_window_infer"),
        text_hidden_size=_int_value(text_config, "hidden_size"),
        text_intermediate_size=_int_value(text_config, "intermediate_size"),
        text_vocab_size=_int_value(text_config, "vocab_size"),
        text_decoder_layers=_int_value(text_config, "num_hidden_layers"),
        text_num_attention_heads=_int_value(text_config, "num_attention_heads"),
        text_num_key_value_heads=_int_value(text_config, "num_key_value_heads"),
        text_head_dim=_int_value(text_config, "head_dim"),
        text_rope_theta=_float_value(text_config, "rope_theta", 5_000_000.0),
        text_mrope_section=mrope_section,
        audio_token_id=_int_value(thinker_config, "audio_token_id"),
    )


def reflect_qwen3_asr_source(config: Qwen3AsrExportConfig) -> TorchModuleReflection:
    from transformers.utils import logging as hf_logging

    stream = io.StringIO()
    previous_verbosity = hf_logging.get_verbosity()
    try:
        hf_logging.set_verbosity(50)
        with _suppress_native_output(), redirect_stdout(stream), redirect_stderr(stream):
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )

            source_config = qwen3_asr_source_config_from_export_config(config)
            model = instantiate_torch_module_on_meta(
                lambda: Qwen3ASRForConditionalGeneration(source_config)
            )
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    return reflect_torch_module(model)


def qwen3_asr_source_config_from_export_config(config: Qwen3AsrExportConfig) -> Any:
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    return Qwen3ASRConfig(
        thinker_config={
            "audio_config": {
                "num_mel_bins": config.audio_num_mel_bins,
                "encoder_layers": config.audio_encoder_layers,
                "encoder_attention_heads": config.audio_encoder_attention_heads,
                "encoder_ffn_dim": config.audio_encoder_ffn_dim,
                "d_model": config.audio_hidden_size,
                "n_window": config.audio_n_window,
                "output_dim": config.audio_output_size,
                "n_window_infer": config.audio_n_window_infer,
                "downsample_hidden_size": config.audio_downsample_hidden_size,
            },
            "text_config": {
                "vocab_size": config.text_vocab_size,
                "hidden_size": config.text_hidden_size,
                "intermediate_size": config.text_intermediate_size,
                "num_hidden_layers": config.text_decoder_layers,
                "num_attention_heads": config.text_num_attention_heads,
                "num_key_value_heads": config.text_num_key_value_heads,
                "head_dim": config.text_head_dim,
                "rope_theta": config.text_rope_theta,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_section": list(config.text_mrope_section),
                },
                "pad_token_id": None,
            },
            "audio_token_id": config.audio_token_id,
        }
    )


def build_qwen3_asr_pattern_context(
    *,
    config: Qwen3AsrExportConfig,
    reflection: TorchModuleReflection | None,
) -> dict[str, object]:
    if reflection is not None:
        _validate_reflected_source(config, reflection)

    shader_variants = _export_shader_variants()
    audio_layer_ops = _audio_layer_ops()
    text_layer_ops = _text_layer_ops()
    audio_tower_ops = _audio_tower_ops()
    text_prefill_ops = _text_prefill_ops()
    text_decode_ops = _text_decode_ops()
    token_select_ops = _token_select_ops()
    token_store_ops = _token_store_ops()
    audio_layer_fields = tensor_scaffold_fields_from_lowered_ops(
        ops=_lowered_op_contracts(audio_layer_ops),
        shader_variants=shader_variants,
        parameter_sources=_audio_layer_parameter_sources(),
        external_fields=("hidden_states", "cu_seqlens"),
    )
    text_layer_fields = tensor_scaffold_fields_from_lowered_ops(
        ops=_lowered_op_contracts(text_layer_ops),
        shader_variants=shader_variants,
        parameter_sources=_text_layer_parameter_sources(),
        external_fields=("hidden", "rope_cos", "rope_sin", "cache_position"),
        role_overrides={"key_cache": "state", "value_cache": "state"},
    )
    audio_tower_fields = tensor_scaffold_fields_from_lowered_ops(
        ops=_lowered_op_contracts(audio_tower_ops),
        shader_variants=shader_variants,
        parameter_sources=_audio_tower_parameter_sources(),
        extra_fields=(TensorFieldPattern("layers"),),
        role_overrides={"last_hidden_state": "output"},
    )
    text_prefill_fields = _text_prefill_fields(
        ops=_lowered_op_contracts(text_prefill_ops),
        shader_variants=shader_variants,
    )
    text_decode_fields = _text_decode_fields(
        ops=_lowered_op_contracts(text_decode_ops),
        shader_variants=shader_variants,
    )
    token_select_fields = tensor_scaffold_fields_from_lowered_ops(
        ops=_lowered_op_contracts(token_select_ops),
        shader_variants=shader_variants,
        external_fields=("logits",),
        role_overrides={"next_token": "output", "done": "output"},
    )
    token_store_fields = tensor_scaffold_fields_from_lowered_ops(
        ops=_lowered_op_contracts(token_store_ops),
        shader_variants=shader_variants,
        role_overrides={
            "generated_tokens": "state",
            "generated_length": "state",
            "stopped": "state",
        },
    )
    return {
        "config": config,
        "defaults": config.to_template_defaults(),
        "audio_layer_fields": audio_layer_fields,
        "text_layer_fields": text_layer_fields,
        "audio_tower_fields": audio_tower_fields,
        "audio_layer_parameter_fields_source": render_parameter_fields_constant(
            "AUDIO_ENCODER_LAYER_PARAMETER_FIELDS",
            audio_layer_fields,
        ),
        "audio_layer_dataclass_source": render_tensor_dataclass(
            logical_tensor_dataclass_from_patterns(
                class_name="GeneratedQwen3AsrAudioEncoderLayerTensors",
                fields=audio_layer_fields,
            )
        ),
        "text_layer_parameter_fields_source": render_parameter_fields_constant(
            "TEXT_DECODER_LAYER_PARAMETER_FIELDS",
            text_layer_fields,
        ),
        "text_layer_dataclass_source": render_tensor_dataclass(
            logical_tensor_dataclass_from_patterns(
                class_name="GeneratedQwen3AsrTextLayerTensors",
                fields=text_layer_fields,
            )
        ),
        "audio_tower_parameter_fields_source": render_parameter_fields_constant(
            "AUDIO_TOWER_PARAMETER_FIELDS",
            audio_tower_fields,
        ),
        "audio_tower_dataclass_source": render_tensor_dataclass(
            logical_tensor_dataclass_from_patterns(
                class_name="GeneratedQwen3AsrAudioTowerTensors",
                fields=audio_tower_fields,
                annotation_overrides={
                    "layers": "tuple[GeneratedQwen3AsrAudioEncoderLayerTensors, ...]"
                },
            )
        ),
        "text_dataclasses_source": render_tensor_dataclasses(
            _text_tensor_dataclasses(
                prefill_fields=text_prefill_fields,
                decode_fields=text_decode_fields,
                token_select_fields=token_select_fields,
                token_store_fields=token_store_fields,
            )
        ),
        "text_frame_fields": _text_frame_fields(),
        "audio_tower_ops": audio_tower_ops,
        "audio_layer_ops": audio_layer_ops,
        "text_layer_ops": text_layer_ops,
        "text_layer_ops_decode": _text_layer_ops_decode(),
        "text_prefill_ops": text_prefill_ops,
        "text_decode_ops": text_decode_ops,
        "token_select_ops": token_select_ops,
        "token_store_ops": token_store_ops,
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

    result = export_qwen3_asr_package(
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


def _render_qwen3_asr_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    files = [
        renderer.render("qwen3_asr/__init__.py.j2", "__init__.py", **context),
    ]
    files.extend(_render_tensor_files(renderer, context))
    files.extend(_render_shader_files(renderer, context))
    files.extend(_render_frame_files(renderer, context))
    return files


def _render_tensor_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    files = [
        _render_tensor_package_init_file(),
        render_logical_tensor_helpers_file(relative_path="tensors/_logical.py"),
        renderer.render(
            "qwen3_asr/tensors/audio_tower_layer.py.j2",
            "tensors/audio_tower_layer.py",
            **context,
        ),
        renderer.render(
            "qwen3_asr/tensors/audio_tower.py.j2",
            "tensors/audio_tower.py",
            **context,
        ),
        renderer.render(
            "qwen3_asr/tensors/text_layer.py.j2",
            "tensors/text_layer.py",
            **context,
        ),
        renderer.render("qwen3_asr/tensors/text.py.j2", "tensors/text.py", **context),
    ]
    return files


def _render_tensor_package_init_file() -> RenderedFile:
    return render_python_init_file(
        relative_path="tensors/__init__.py",
        docstring="Generated Qwen3-ASR tensor scaffolds.",
        imports=(
            PythonImportDecl(
                "models.generated_qwen3_asr.tensors.audio_tower",
                (
                    "AUDIO_TOWER_PARAMETER_FIELDS",
                    "GeneratedQwen3AsrAudioTowerTensors",
                    "declare_generated_qwen3_asr_audio_tower_tensors",
                ),
            ),
            PythonImportDecl(
                "models.generated_qwen3_asr.tensors.audio_tower_layer",
                (
                    "AUDIO_ENCODER_LAYER_PARAMETER_FIELDS",
                    "GeneratedQwen3AsrAudioEncoderLayerTensors",
                    "declare_generated_qwen3_asr_audio_encoder_layer_tensors",
                ),
            ),
            PythonImportDecl(
                "models.generated_qwen3_asr.tensors.text",
                (
                    "GeneratedQwen3AsrTextDecodeTensors",
                    "GeneratedQwen3AsrTextPrefillTensors",
                    "GeneratedQwen3AsrTextTensors",
                    "GeneratedQwen3AsrTokenSelectTensors",
                    "GeneratedQwen3AsrTokenStoreTensors",
                    "declare_generated_qwen3_asr_text_tensors",
                ),
            ),
            PythonImportDecl(
                "models.generated_qwen3_asr.tensors.text_layer",
                (
                    "TEXT_DECODER_LAYER_PARAMETER_FIELDS",
                    "GeneratedQwen3AsrTextLayerTensors",
                    "declare_generated_qwen3_asr_text_layer_tensors",
                ),
            ),
        ),
    )


def _render_shader_files(
    renderer: TemplateRenderer,
    context: dict[str, object],
) -> list[RenderedFile]:
    files = [
        renderer.render("qwen3_asr/shaders/__init__.py.j2", "shaders/__init__.py", **context),
    ]
    shader_modules = context["shader_modules"]
    if not isinstance(shader_modules, tuple):
        raise TypeError("shader_modules must be a tuple")
    for module in shader_modules:
        if not isinstance(module, ShaderModulePattern):
            raise TypeError(f"shader_modules entry must be ShaderModulePattern, got {type(module)}")
        files.append(
            renderer.render(
                "qwen3_asr/shaders/shader_module.py.j2",
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
    shader_variants = _export_shader_variants()
    return [
        renderer.render("qwen3_asr/_frame.py.j2", "_frame.py", **context),
        render_frame_module(_audio_tower_frame_spec(), shader_variants),
        render_frame_module(_text_prefill_frame_spec(), shader_variants),
        render_frame_module(_text_decode_frame_spec(), shader_variants),
        renderer.render("qwen3_asr/token_select.py.j2", "token_select.py", **context),
        renderer.render("qwen3_asr/token_store.py.j2", "token_store.py", **context),
        renderer.render("qwen3_asr/execution.py.j2", "execution.py", **context),
        renderer.render("qwen3_asr/transcribe.py.j2", "transcribe.py", **context),
    ]


def _audio_tower_frame_spec() -> FrameSpec:
    return FrameSpec(
        function_name="run_generated_qwen3_asr_audio_tower",
        frame_name="generated_qwen3_asr.audio_tower",
        tensors_class="GeneratedQwen3AsrAudioTowerTensors",
        tensors_import="models.generated_qwen3_asr.tensors.audio_tower",
        ops=_audio_tower_ops(),
        return_expr="tensors.last_hidden_state",
        layer_loop_target="aten.torch2vk.audio_encoder_layer_loop.default",
        layer_ops=_audio_layer_ops(),
        layer_state_field="hidden_states",
        layer_state_init="tensors.hidden_states",
        layer_external_fields={"cu_seqlens": "tensors.cu_seqlens"},
        extra_params="    **_kw,",
    )


def _text_prefill_frame_spec() -> FrameSpec:
    return FrameSpec(
        function_name="run_generated_qwen3_asr_text_prefill",
        frame_name="generated_qwen3_asr.text_prefill",
        tensors_class="GeneratedQwen3AsrTextPrefillTensors",
        tensors_import="models.generated_qwen3_asr.tensors.text",
        ops=_text_prefill_ops(),
        layer_loop_target="aten.torch2vk.text_decoder_layer_loop.default",
        layer_ops=_text_layer_ops(),
        layer_state_field="hidden",
        layer_state_init="tensors.inputs_embeds",
        layer_external_fields={"rope_cos": "tensors.rope_cos", "rope_sin": "tensors.rope_sin"},
        extra_params="    **_kw,",
    )


def _text_decode_frame_spec() -> FrameSpec:
    return FrameSpec(
        function_name="run_generated_qwen3_asr_text_decode",
        frame_name="generated_qwen3_asr.text_decode",
        tensors_class="GeneratedQwen3AsrTextDecodeTensors",
        tensors_import="models.generated_qwen3_asr.tensors.text",
        ops=_text_decode_ops(),
        layer_loop_target="aten.torch2vk.text_decoder_layer_loop.default",
        layer_ops=_text_layer_ops_decode(),
        layer_state_field="hidden",
        layer_state_init="tensors.inputs_embeds",
        layer_external_fields={
            "rope_cos": "tensors.rope_cos",
            "rope_sin": "tensors.rope_sin",
            "cache_position": "tensors.cache_position",
        },
        extra_params="    **_kw,",
    )


def _validate_reflected_source(
    config: Qwen3AsrExportConfig,
    reflection: TorchModuleReflection,
) -> None:
    reflection.require_module_type("thinker.audio_tower")
    reflection.require_module_type("thinker.model")
    if config.audio_encoder_layers:
        reflection.require_parameter("thinker.audio_tower.layers.0.self_attn.q_proj.weight")
        reflection.require_parameter("thinker.audio_tower.layers.0.self_attn.q_proj.bias")
        reflection.require_parameter("thinker.audio_tower.layers.0.fc1.weight")
    if config.text_decoder_layers:
        reflection.require_parameter("thinker.model.layers.0.self_attn.q_proj.weight")
        reflection.require_parameter("thinker.model.layers.0.self_attn.q_norm.weight")
        reflection.require_parameter("thinker.model.layers.0.self_attn.k_norm.weight")
        reflection.require_parameter("thinker.model.layers.0.mlp.down_proj.weight")
    reflection.require_parameter("thinker.audio_tower.conv2d1.weight")
    reflection.require_parameter("thinker.audio_tower.conv_out.weight")
    reflection.require_parameter("thinker.audio_tower.proj1.weight")
    reflection.require_parameter("thinker.audio_tower.proj2.weight")
    reflection.require_parameter("thinker.model.embed_tokens.weight")
    reflection.require_parameter("thinker.model.norm.weight")
    reflection.require_parameter("thinker.lm_head.weight")


def _audio_layer_parameter_sources() -> dict[str, str]:
    return {
        "self_attn_layer_norm_weight": "self_attn_layer_norm.weight",
        "self_attn_layer_norm_bias": "self_attn_layer_norm.bias",
        "q_proj_weight": "self_attn.q_proj.weight",
        "q_proj_bias": "self_attn.q_proj.bias",
        "k_proj_weight": "self_attn.k_proj.weight",
        "k_proj_bias": "self_attn.k_proj.bias",
        "v_proj_weight": "self_attn.v_proj.weight",
        "v_proj_bias": "self_attn.v_proj.bias",
        "out_proj_weight": "self_attn.out_proj.weight",
        "out_proj_bias": "self_attn.out_proj.bias",
        "final_layer_norm_weight": "final_layer_norm.weight",
        "final_layer_norm_bias": "final_layer_norm.bias",
        "fc1_weight": "fc1.weight",
        "fc1_bias": "fc1.bias",
        "fc2_weight": "fc2.weight",
        "fc2_bias": "fc2.bias",
    }


def _text_layer_parameter_sources() -> dict[str, str]:
    return {
        "input_layernorm_weight": "input_layernorm.weight",
        "post_attention_layernorm_weight": "post_attention_layernorm.weight",
        "q_norm_weight": "self_attn.q_norm.weight",
        "k_norm_weight": "self_attn.k_norm.weight",
        "q_proj_weight": "self_attn.q_proj.weight",
        "k_proj_weight": "self_attn.k_proj.weight",
        "v_proj_weight": "self_attn.v_proj.weight",
        "o_proj_weight": "self_attn.o_proj.weight",
        "gate_proj_weight": "mlp.gate_proj.weight",
        "up_proj_weight": "mlp.up_proj.weight",
        "down_proj_weight": "mlp.down_proj.weight",
    }


def _audio_tower_parameter_sources() -> dict[str, str]:
    return {
        "conv2d1_weight": "conv2d1.weight",
        "conv2d1_bias": "conv2d1.bias",
        "conv2d2_weight": "conv2d2.weight",
        "conv2d2_bias": "conv2d2.bias",
        "conv2d3_weight": "conv2d3.weight",
        "conv2d3_bias": "conv2d3.bias",
        "conv_out_weight": "conv_out.weight",
        "ln_post_weight": "ln_post.weight",
        "ln_post_bias": "ln_post.bias",
        "proj1_weight": "proj1.weight",
        "proj1_bias": "proj1.bias",
        "proj2_weight": "proj2.weight",
        "proj2_bias": "proj2.bias",
    }


def _text_parameter_sources() -> dict[str, str]:
    return {
        "embed_tokens_weight": "thinker.model.embed_tokens.weight",
        "norm_weight": "thinker.model.norm.weight",
        "lm_head_weight": "thinker.lm_head.weight",
    }


def _text_frame_fields() -> tuple[TensorFieldPattern, ...]:
    return (
        TensorFieldPattern("prefill"),
        TensorFieldPattern("decode"),
        TensorFieldPattern("token_select"),
        TensorFieldPattern("token_store"),
    )


def _text_prefill_fields(
    *,
    ops: Sequence[LoweredOpContract],
    shader_variants: Mapping[str, ShaderVariant],
) -> tuple[TensorFieldPattern, ...]:
    fields = tensor_scaffold_fields_from_lowered_ops(
        ops=ops,
        shader_variants=shader_variants,
        parameter_sources=_text_parameter_sources(),
        extra_fields=(
            TensorFieldPattern("attention_mask", dtype="int64", role="input"),
            TensorFieldPattern("input_features", dtype="float32", role="input"),
            TensorFieldPattern("feature_attention_mask", dtype="int64", role="input"),
            TensorFieldPattern("position_ids", dtype="int64", role="input"),
            TensorFieldPattern("rope_cos", dtype="float32", role="state"),
            TensorFieldPattern("rope_sin", dtype="float32", role="state"),
            TensorFieldPattern("layers"),
        ),
        external_fields=("hidden",),
        role_overrides={
            "audio_features": "activation",
            "audio_scatter_mask": "activation",
            "inputs_embeds": "activation",
            "logits": "output",
        },
        include_unresolved_ops=False,
    )
    return _ordered_tensor_fields(
        fields,
        (
            "input_ids",
            "attention_mask",
            "input_features",
            "feature_attention_mask",
            "position_ids",
            "rope_cos",
            "rope_sin",
            "audio_features",
            "audio_scatter_mask",
            "embed_tokens_weight",
            "inputs_embeds",
            "layers",
            "norm_weight",
            "final_norm",
            "lm_head_weight",
            "logits",
        ),
    )


def _text_decode_fields(
    *,
    ops: Sequence[LoweredOpContract],
    shader_variants: Mapping[str, ShaderVariant],
) -> tuple[TensorFieldPattern, ...]:
    fields = tensor_scaffold_fields_from_lowered_ops(
        ops=ops,
        shader_variants=shader_variants,
        parameter_sources=_text_parameter_sources(),
        extra_fields=(
            TensorFieldPattern("attention_mask", dtype="int64", role="input"),
            TensorFieldPattern("position_ids", dtype="int64", role="input"),
            TensorFieldPattern("rope_cos", dtype="float32", role="state"),
            TensorFieldPattern("rope_sin", dtype="float32", role="state"),
            TensorFieldPattern("cache_position", dtype="int64", role="input"),
            TensorFieldPattern("layers"),
            TensorFieldPattern("lm_head_select_scratch", dtype="float32", role="scratch"),
        ),
        external_fields=("hidden",),
        role_overrides={
            "inputs_embeds": "activation",
            "logits": "output",
        },
        include_unresolved_ops=False,
    )
    return _ordered_tensor_fields(
        fields,
        (
            "input_ids",
            "attention_mask",
            "position_ids",
            "rope_cos",
            "rope_sin",
            "cache_position",
            "embed_tokens_weight",
            "inputs_embeds",
            "layers",
            "norm_weight",
            "final_norm",
            "lm_head_weight",
            "lm_head_select_scratch",
            "logits",
        ),
    )


def _ordered_tensor_fields(
    fields: Sequence[TensorFieldPattern],
    order: Sequence[str],
) -> tuple[TensorFieldPattern, ...]:
    by_name = {field.field: field for field in fields}
    ordered = [by_name.pop(name) for name in order if name in by_name]
    ordered.extend(by_name.values())
    return tuple(ordered)


def _text_tensor_dataclasses(
    *,
    prefill_fields: Sequence[TensorFieldPattern],
    decode_fields: Sequence[TensorFieldPattern],
    token_select_fields: Sequence[TensorFieldPattern],
    token_store_fields: Sequence[TensorFieldPattern],
) -> tuple[TensorDataclassDecl, ...]:
    return (
        logical_tensor_dataclass_from_patterns(
            class_name="GeneratedQwen3AsrTextPrefillTensors",
            fields=prefill_fields,
            annotation_overrides={
                "input_features": "LogicalTensor | None",
                "feature_attention_mask": "LogicalTensor | None",
                "layers": "tuple[GeneratedQwen3AsrTextLayerTensors, ...]",
            },
        ),
        logical_tensor_dataclass_from_patterns(
            class_name="GeneratedQwen3AsrTextDecodeTensors",
            fields=decode_fields,
            annotation_overrides={
                "layers": "tuple[GeneratedQwen3AsrTextLayerTensors, ...]",
            },
        ),
        logical_tensor_dataclass_from_patterns(
            class_name="GeneratedQwen3AsrTokenSelectTensors",
            fields=token_select_fields,
        ),
        logical_tensor_dataclass_from_patterns(
            class_name="GeneratedQwen3AsrTokenStoreTensors",
            fields=token_store_fields,
        ),
        TensorDataclassDecl(
            "GeneratedQwen3AsrTextTensors",
            (
                TensorDataclassFieldDecl("prefill", "GeneratedQwen3AsrTextPrefillTensors"),
                TensorDataclassFieldDecl("decode", "GeneratedQwen3AsrTextDecodeTensors"),
                TensorDataclassFieldDecl("token_select", "GeneratedQwen3AsrTokenSelectTensors"),
                TensorDataclassFieldDecl("token_store", "GeneratedQwen3AsrTokenStoreTensors"),
            ),
        ),
    )


def _audio_tower_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.torch2vk.pad_feature.default",
            ("input_features", "feature_lens"),
            ("padded_feature",),
        ),
        _op(
            "aten.torch2vk.conv2d_gelu.default",
            ("padded_feature", "conv2d1_weight", "conv2d1_bias"),
            ("conv2d1_gelu",),
        ),
        _op(
            "aten.torch2vk.conv2d_gelu.default",
            ("conv2d1_gelu", "conv2d2_weight", "conv2d2_bias"),
            ("conv2d2_gelu",),
        ),
        _op(
            "aten.torch2vk.conv2d_gelu.default",
            ("conv2d2_gelu", "conv2d3_weight", "conv2d3_bias"),
            ("conv2d3_gelu",),
        ),
        _op("aten.torch2vk.conv_out.default", ("conv2d3_gelu", "conv_out_weight"), ("conv_out",)),
        _op("aten.add.Tensor", ("conv_out",), ("conv_out_add_position",), name="add_position"),
        _op(
            "aten.torch2vk.compact_after_cnn.default",
            ("conv_out_add_position", "feature_lens"),
            ("hidden_states",),
        ),
        _op("aten.torch2vk.cu_seqlens.default", ("feature_lens",), ("cu_seqlens",)),
        _op(
            "aten.torch2vk.audio_encoder_layer_loop.default",
            ("hidden_states", "cu_seqlens", "layers"),
            ("hidden_states",),
        ),
        _op(
            "aten.native_layer_norm.default",
            ("hidden_states", "ln_post_weight", "ln_post_bias"),
            ("ln_post",),
        ),
        _op(
            "aten.torch2vk.linear_gelu.default",
            ("ln_post", "proj1_weight", "proj1_bias"),
            ("proj1_gelu",),
        ),
        _op(
            "aten.linear.default",
            ("proj1_gelu", "proj2_weight", "proj2_bias"),
            ("last_hidden_state",),
        ),
    )


def _audio_layer_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.native_layer_norm.default",
            ("hidden_states", "self_attn_layer_norm_weight", "self_attn_layer_norm_bias"),
            ("self_attn_layer_norm",),
        ),
        _op(
            "aten.linear.default",
            ("self_attn_layer_norm", "q_proj_weight", "q_proj_bias"),
            ("q_proj",),
        ),
        _op(
            "aten.linear.default",
            ("self_attn_layer_norm", "k_proj_weight", "k_proj_bias"),
            ("k_proj",),
        ),
        _op(
            "aten.linear.default",
            ("self_attn_layer_norm", "v_proj_weight", "v_proj_bias"),
            ("v_proj",),
        ),
        _op(
            "aten.torch2vk.encoder_attention.default",
            ("q_proj", "k_proj", "v_proj", "cu_seqlens"),
            ("self_attn",),
        ),
        _op(
            "aten.linear.default", ("self_attn", "out_proj_weight", "out_proj_bias"), ("out_proj",)
        ),
        _op(
            "aten.add.Tensor",
            ("hidden_states", "out_proj"),
            ("self_attn_residual",),
            name="residual_add",
        ),
        _op(
            "aten.native_layer_norm.default",
            ("self_attn_residual", "final_layer_norm_weight", "final_layer_norm_bias"),
            ("final_layer_norm",),
        ),
        _op(
            "aten.torch2vk.linear_gelu.default",
            ("final_layer_norm", "fc1_weight", "fc1_bias"),
            ("fc1_gelu",),
        ),
        _op("aten.linear.default", ("fc1_gelu", "fc2_weight", "fc2_bias"), ("fc2",)),
        _op(
            "aten.add.Tensor",
            ("self_attn_residual", "fc2"),
            ("output",),
            name="residual_add",
        ),
    )


def _text_layer_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.rms_norm.default",
            ("hidden", "input_layernorm_weight"),
            ("input_layernorm",),
        ),
        _op(
            "aten.linear.default",
            ("input_layernorm", "q_proj_weight"),
            ("q_proj",),
        ),
        _op(
            "aten.linear.default",
            ("input_layernorm", "k_proj_weight"),
            ("k_proj",),
        ),
        _op(
            "aten.linear.default",
            ("input_layernorm", "v_proj_weight"),
            ("v_proj",),
        ),
        _op(
            "aten.torch2vk.text_qk_norm.default",
            ("q_proj", "q_norm_weight"),
            ("q_normed",),
        ),
        _op(
            "aten.torch2vk.text_qk_norm.default",
            ("k_proj", "k_norm_weight"),
            ("k_normed",),
        ),
        _op(
            "aten.torch2vk.text_rope.default",
            ("q_normed", "rope_cos", "rope_sin"),
            ("q_roped",),
        ),
        _op(
            "aten.torch2vk.text_rope.default",
            ("k_normed", "rope_cos", "rope_sin"),
            ("k_roped",),
        ),
        _op(
            "aten.torch2vk.text_kv_cache_write.default",
            ("k_roped", "v_proj", "key_cache", "value_cache"),
            ("key_cache", "value_cache"),
        ),
        _op(
            "aten.torch2vk.text_attention.default",
            ("q_roped", "key_cache", "value_cache"),
            ("attention",),
        ),
        _op(
            "aten.linear.default",
            ("attention", "o_proj_weight"),
            ("o_proj",),
        ),
        _op(
            "aten.add.Tensor",
            ("hidden", "o_proj"),
            ("attn_residual",),
        ),
        _op(
            "aten.rms_norm.default",
            ("attn_residual", "post_attention_layernorm_weight"),
            ("post_attention_layernorm",),
        ),
        _op(
            "aten.linear.default",
            ("post_attention_layernorm", "gate_proj_weight"),
            ("gate_proj",),
        ),
        _op(
            "aten.linear.default",
            ("post_attention_layernorm", "up_proj_weight"),
            ("up_proj",),
        ),
        _op(
            "aten.torch2vk.text_swiglu.default",
            ("gate_proj", "up_proj"),
            ("swiglu",),
        ),
        _op(
            "aten.linear.default",
            ("swiglu", "down_proj_weight"),
            ("down_proj",),
        ),
        _op(
            "aten.add.Tensor",
            ("attn_residual", "down_proj"),
            ("output",),
        ),
    )


def _text_layer_ops_decode() -> tuple[object, ...]:
    """Text decoder layer ops for decode phase (includes cache_position)."""
    return (
        _op("aten.rms_norm.default", ("hidden", "input_layernorm_weight"), ("input_layernorm",)),
        _op("aten.linear.default", ("input_layernorm", "q_proj_weight"), ("q_proj",)),
        _op("aten.linear.default", ("input_layernorm", "k_proj_weight"), ("k_proj",)),
        _op("aten.linear.default", ("input_layernorm", "v_proj_weight"), ("v_proj",)),
        _op("aten.torch2vk.text_qk_norm.default", ("q_proj", "q_norm_weight"), ("q_normed",)),
        _op("aten.torch2vk.text_qk_norm.default", ("k_proj", "k_norm_weight"), ("k_normed",)),
        _op("aten.torch2vk.text_rope.default", ("q_normed", "rope_cos", "rope_sin"), ("q_roped",)),
        _op("aten.torch2vk.text_rope.default", ("k_normed", "rope_cos", "rope_sin"), ("k_roped",)),
        _op(
            "aten.torch2vk.text_kv_cache_write.default",
            ("k_roped", "v_proj", "cache_position", "key_cache", "value_cache"),
            ("key_cache", "value_cache"),
        ),
        _op(
            "aten.torch2vk.text_attention.default",
            ("q_roped", "key_cache", "value_cache", "cache_position"),
            ("attention",),
        ),
        _op("aten.linear.default", ("attention", "o_proj_weight"), ("o_proj",)),
        _op("aten.add.Tensor", ("hidden", "o_proj"), ("attn_residual",)),
        _op(
            "aten.rms_norm.default",
            ("attn_residual", "post_attention_layernorm_weight"),
            ("post_attention_layernorm",),
        ),
        _op("aten.linear.default", ("post_attention_layernorm", "gate_proj_weight"), ("gate_proj",)),
        _op("aten.linear.default", ("post_attention_layernorm", "up_proj_weight"), ("up_proj",)),
        _op("aten.torch2vk.text_swiglu.default", ("gate_proj", "up_proj"), ("swiglu",)),
        _op("aten.linear.default", ("swiglu", "down_proj_weight"), ("down_proj",)),
        _op("aten.add.Tensor", ("attn_residual", "down_proj"), ("output",)),
    )


def _text_prefill_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.torch2vk.prefill_inputs_embeds.default",
            ("input_ids", "embed_tokens_weight", "audio_features"),
            ("audio_scatter_mask", "inputs_embeds"),
        ),
        _op(
            "aten.torch2vk.text_decoder_layer_loop.default",
            ("inputs_embeds", "rope_cos", "rope_sin", "layers"),
            ("hidden",),
        ),
        _op("aten.rms_norm.default", ("hidden", "norm_weight"), ("final_norm",)),
        _op("aten.linear.default", ("final_norm", "lm_head_weight"), ("logits",)),
    )


def _text_decode_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.embedding.default",
            ("input_ids", "embed_tokens_weight"),
            ("inputs_embeds",),
        ),
        _op(
            "aten.torch2vk.text_decoder_layer_loop.default",
            ("inputs_embeds", "cache_position", "rope_cos", "rope_sin", "layers"),
            ("hidden",),
        ),
        _op("aten.rms_norm.default", ("hidden", "norm_weight"), ("final_norm",)),
        _op("aten.linear.default", ("final_norm", "lm_head_weight"), ("logits",)),
    )


def _token_select_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.torch2vk.greedy_argmax.default",
            ("logits", "eos_token_ids"),
            ("next_token", "done"),
        ),
    )


def _token_store_ops() -> tuple[object, ...]:
    return (
        _op(
            "aten.torch2vk.token_store.default",
            ("next_token", "token_index", "done"),
            ("generated_tokens", "generated_length", "stopped"),
        ),
    )


def _export_shader_variants() -> dict[str, ShaderVariant]:
    return {
        name: value
        for name in export_shaders.__all__
        if isinstance(value := getattr(export_shaders, name), ShaderVariant)
    }


def _shader_modules() -> tuple[ShaderModulePattern, ...]:
    modules: list[ShaderModulePattern] = []
    for path in sorted(_SOURCE_SHADER_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        modules.append(
            ShaderModulePattern(
                module=path.stem,
                constants=_source_shader_constants(path),
                source=format_python_source(
                    path.read_text(encoding="utf-8").rstrip() + "\n",
                    filename=_PACKAGE_DIR / "shaders" / path.name,
                ),
            )
        )
    return tuple(modules)


def _source_shader_constants(path: Path) -> tuple[str, ...]:
    module = _load_source_shader_module(path)
    constants = tuple(
        name
        for name, value in vars(module).items()
        if name.startswith("QWEN3_ASR_") and isinstance(value, ShaderVariant)
    )
    if not constants:
        raise ValueError(f"{path} does not define any QWEN3_ASR_* ShaderVariant constants")
    return constants


def _load_source_shader_module(path: Path) -> ModuleType:
    module_name = f"_torch2vk_qwen3_asr_shader_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Qwen3-ASR shader module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_default_config_payload() -> JsonObject:
    Qwen3ASRConfig = _load_source_configuration_module().Qwen3ASRConfig
    payload = Qwen3ASRConfig().to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError(f"Qwen3ASRConfig.to_dict returned {type(payload).__name__}")
    return payload


def _load_source_configuration_module() -> ModuleType:
    src_root = Path(__file__).resolve().parents[2]
    path = src_root / "qwen_asr/core/transformers_backend/configuration_qwen3_asr.py"
    spec = importlib.util.spec_from_file_location("_torch2vk_qwen3_asr_configuration", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Qwen3-ASR configuration from {path}")
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


def _expect_tuple(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    return value


def _int_value(config: JsonObject, key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _float_value(config: JsonObject, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} values must be int, got {type(item).__name__}")
        result.append(item)
    return tuple(result)


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
