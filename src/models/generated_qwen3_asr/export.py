"""Regenerate the generated Qwen3-ASR scaffold package.

This module owns the Qwen3-ASR-specific export recipe. PyTorch module
reflection is the source of truth for parameter names, layer counts, and model
dimensions; the recipe only names the frame boundaries and lowered op patterns.
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

import torch2vk.exportv2.shaders as export_shaders
from torch2vk.exportv2 import (
    ExportWriteResult,
    FrameSpec,
    PythonImportDecl,
    RenderedFile,
    TemplateRenderer,
    TensorDataclassDecl,
    TensorDataclassFieldDecl,
    TensorFieldPattern,
    TorchModuleReflection,
    instantiate_torch_module_on_meta,
    logical_tensor_dataclass_from_patterns,
    reflect_torch_module,
    render_frame_module,
    render_logical_tensor_helpers_file,
    render_parameter_fields_constant,
    render_python_init_file,
    render_tensor_dataclass,
    render_tensor_dataclasses,
    remove_stale_files,
    shader_variants_from_module,
    tensor_fields_from_reflected_static_nodes,
    tensor_scaffold_fields_from_static_nodes,
    write_rendered_files,
)
from torch2vk.runtime.shader import ShaderVariant


JsonObject = Mapping[str, object]
Qwen3AsrNode = tuple[str, tuple[str, ...], tuple[str, ...]]

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATE_ROOT = _PACKAGE_DIR / "export_templates"
_STALE_SHADER_FILES = tuple(
    Path("shaders") / f"{module}.py"
    for module in (
        "__init__",
        "add_f32",
        "attention_f32",
        "compact_after_cnn_f32",
        "conv2d_gelu_f32",
        "conv_out_add_position_f32",
        "cu_seqlens_u32",
        "layer_norm_f32",
        "linear_f32",
        "pad_feature_f32",
        "rope_table_f32",
        "text_add_3d_f32",
        "text_attention_decode_f32",
        "text_attention_prefill_f32",
        "text_embed_lookup_f32",
        "text_gate_up_swiglu_t1_f32",
        "text_kv_cache_write_f32",
        "text_linear_nobias_f32",
        "text_linear_nobias_t1_f32",
        "text_linear_nobias_t1_splitk4_f32",
        "text_lm_head_select_t1_f32",
        "text_prefill_inputs_embeds_f32",
        "text_qk_norm_f32",
        "text_qkv_proj_t1_f32",
        "text_rms_norm_f32",
        "text_rope_f32",
        "text_swiglu_f32",
        "token_select_f32",
        "token_store_f32",
    )
)
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
    *_STALE_SHADER_FILES,
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrModelLayout:
    audio_hidden_size: int
    audio_output_size: int
    audio_downsample_hidden_size: int
    audio_encoder_layers: int
    audio_encoder_ffn_dim: int
    text_hidden_size: int
    text_intermediate_size: int
    text_vocab_size: int
    text_decoder_layers: int
    text_num_attention_heads: int
    text_num_key_value_heads: int
    text_head_dim: int


@dataclass(frozen=True, slots=True)
class Qwen3AsrRuntimeConfig:
    text_rope_theta: float
    text_mrope_section: tuple[int, ...]
    eos_token_ids: tuple[int, ...] = (151645, 151643)


@dataclass(frozen=True, slots=True)
class Qwen3AsrExportSpec:
    layout: Qwen3AsrModelLayout
    runtime: Qwen3AsrRuntimeConfig

    def to_template_defaults(self) -> dict[str, object]:
        value = asdict(self.layout)
        value.update(asdict(self.runtime))
        value["text_mrope_section"] = tuple(self.runtime.text_mrope_section)
        value["eos_token_ids"] = tuple(self.runtime.eos_token_ids)
        return value


@dataclass(frozen=True, slots=True)
class Qwen3AsrInspectedModel:
    spec: Qwen3AsrExportSpec
    reflection: TorchModuleReflection


def _op(
    target: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> Qwen3AsrNode:
    return target, inputs, outputs


def export_qwen3_asr_package(
    *,
    output_dir: str | Path = _PACKAGE_DIR,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
    check: bool = False,
    dry_run: bool = False,
) -> ExportWriteResult:
    inspected = inspect_qwen3_asr_export_source(
        model_dir=model_dir,
        config_json=config_json,
    )
    context = build_qwen3_asr_pattern_context(inspected=inspected)
    renderer = TemplateRenderer(_TEMPLATE_ROOT)
    files = _render_qwen3_asr_files(renderer, context)
    result = write_rendered_files(files, output_dir, check=check, dry_run=dry_run)
    _remove_stale_files(output_dir, check=check, dry_run=dry_run)
    return result


def load_qwen3_asr_export_spec(
    *,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
) -> Qwen3AsrExportSpec:
    return inspect_qwen3_asr_export_source(
        model_dir=model_dir,
        config_json=config_json,
    ).spec


def inspect_qwen3_asr_export_source(
    *,
    model_dir: str | Path | None = None,
    config_json: str | Path | None = None,
) -> Qwen3AsrInspectedModel:
    if config_json is not None:
        payload = _load_json(Path(config_json))
    elif model_dir is not None:
        payload = _load_json(Path(model_dir) / "config.json")
    else:
        payload = _source_default_config_payload()
    source_config = qwen3_asr_source_config_from_payload(payload)
    reflection = reflect_qwen3_asr_source_config(source_config)
    spec = Qwen3AsrExportSpec(
        layout=qwen3_asr_model_layout_from_reflection(reflection),
        runtime=qwen3_asr_runtime_config_from_payload(payload),
    )
    return Qwen3AsrInspectedModel(spec=spec, reflection=reflection)


def qwen3_asr_runtime_config_from_payload(payload: JsonObject) -> Qwen3AsrRuntimeConfig:
    thinker_config = _expect_mapping(payload.get("thinker_config", payload), "thinker_config")
    text_config = _expect_mapping(thinker_config["text_config"], "text_config")
    rope_scaling = text_config.get("rope_scaling")
    mrope_section: tuple[int, ...] = (24, 20, 20)
    if isinstance(rope_scaling, Mapping):
        mrope_section = _int_tuple(
            rope_scaling.get("mrope_section", mrope_section),
            "rope_scaling.mrope_section",
        )
    return Qwen3AsrRuntimeConfig(
        text_rope_theta=_float_value(text_config, "rope_theta", 5_000_000.0),
        text_mrope_section=mrope_section,
    )


def qwen3_asr_source_config_from_payload(payload: JsonObject) -> Any:
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    payload_dict = dict(payload)
    thinker_config = dict(
        _expect_mapping(payload_dict.get("thinker_config", payload_dict), "thinker_config")
    )
    text_config = dict(_expect_mapping(thinker_config["text_config"], "text_config"))
    text_config["pad_token_id"] = text_config.get("pad_token_id")
    thinker_config["text_config"] = text_config
    payload_dict["thinker_config"] = thinker_config
    return Qwen3ASRConfig(**payload_dict)


def reflect_qwen3_asr_source_config(source_config: Any) -> TorchModuleReflection:
    from transformers.utils import logging as hf_logging

    stream = io.StringIO()
    previous_verbosity = hf_logging.get_verbosity()
    try:
        hf_logging.set_verbosity(50)
        with _suppress_native_output(), redirect_stdout(stream), redirect_stderr(stream):
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )

            model = instantiate_torch_module_on_meta(
                lambda: Qwen3ASRForConditionalGeneration(source_config)
            )
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    return reflect_torch_module(model)


def qwen3_asr_model_layout_from_reflection(
    reflection: TorchModuleReflection,
) -> Qwen3AsrModelLayout:
    audio_encoder_layers = _indexed_parameter_count(
        reflection,
        "thinker.audio_tower.layers",
    )
    text_decoder_layers = _indexed_parameter_count(
        reflection,
        "thinker.model.layers",
    )
    conv2d1_weight = _parameter_shape(
        reflection,
        "thinker.audio_tower.conv2d1.weight",
        rank=4,
    )
    conv_out_weight = _parameter_shape(
        reflection,
        "thinker.audio_tower.conv_out.weight",
        rank=2,
    )
    proj2_weight = _parameter_shape(
        reflection,
        "thinker.audio_tower.proj2.weight",
        rank=2,
    )
    audio_fc1_weight = _parameter_shape(
        reflection,
        "thinker.audio_tower.layers.0.fc1.weight",
        rank=2,
    )
    embed_tokens_weight = _parameter_shape(
        reflection,
        "thinker.model.embed_tokens.weight",
        rank=2,
    )
    q_proj_weight = _parameter_shape(
        reflection,
        "thinker.model.layers.0.self_attn.q_proj.weight",
        rank=2,
    )
    k_proj_weight = _parameter_shape(
        reflection,
        "thinker.model.layers.0.self_attn.k_proj.weight",
        rank=2,
    )
    q_norm_weight = _parameter_shape(
        reflection,
        "thinker.model.layers.0.self_attn.q_norm.weight",
        rank=1,
    )
    gate_proj_weight = _parameter_shape(
        reflection,
        "thinker.model.layers.0.mlp.gate_proj.weight",
        rank=2,
    )
    lm_head_weight = _parameter_shape(
        reflection,
        "thinker.lm_head.weight",
        rank=2,
    )

    text_vocab_size, text_hidden_size = embed_tokens_weight
    if q_proj_weight[1] != text_hidden_size:
        raise RuntimeError("text q_proj input dim does not match embed_tokens hidden dim")
    if k_proj_weight[1] != text_hidden_size:
        raise RuntimeError("text k_proj input dim does not match embed_tokens hidden dim")
    if gate_proj_weight[1] != text_hidden_size:
        raise RuntimeError("text gate_proj input dim does not match embed_tokens hidden dim")
    if lm_head_weight != embed_tokens_weight:
        raise RuntimeError("lm_head.weight shape must match embed_tokens.weight shape")

    text_head_dim = q_norm_weight[0]
    text_num_attention_heads = _checked_div(
        q_proj_weight[0],
        text_head_dim,
        "q_proj output dim",
    )
    text_num_key_value_heads = _checked_div(
        k_proj_weight[0],
        text_head_dim,
        "k_proj output dim",
    )

    return Qwen3AsrModelLayout(
        audio_hidden_size=conv_out_weight[0],
        audio_output_size=proj2_weight[0],
        audio_downsample_hidden_size=conv2d1_weight[0],
        audio_encoder_layers=audio_encoder_layers,
        audio_encoder_ffn_dim=audio_fc1_weight[0],
        text_hidden_size=text_hidden_size,
        text_intermediate_size=gate_proj_weight[0],
        text_vocab_size=text_vocab_size,
        text_decoder_layers=text_decoder_layers,
        text_num_attention_heads=text_num_attention_heads,
        text_num_key_value_heads=text_num_key_value_heads,
        text_head_dim=text_head_dim,
    )


def _parameter_shape(
    reflection: TorchModuleReflection,
    name: str,
    *,
    rank: int,
) -> tuple[int, ...]:
    shape = reflection.require_parameter(name)
    if len(shape) != rank:
        raise RuntimeError(f"{name} must have rank {rank}, got shape {shape}")
    return shape


def _indexed_parameter_count(
    reflection: TorchModuleReflection,
    prefix: str,
) -> int:
    marker = f"{prefix}."
    indices: set[int] = set()
    for parameter_name in reflection.parameter_shapes:
        if not parameter_name.startswith(marker):
            continue
        index_part = parameter_name[len(marker) :].split(".", 1)[0]
        if not index_part.isdecimal():
            raise RuntimeError(f"{prefix} child index must be decimal, got {index_part!r}")
        indices.add(int(index_part))
    if not indices:
        raise RuntimeError(f"reflected model is missing indexed parameters under {prefix!r}")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise RuntimeError(f"{prefix} layer indices are not contiguous: {sorted(indices)}")
    return len(indices)


def _checked_div(value: int, divisor: int, label: str) -> int:
    if divisor <= 0:
        raise RuntimeError(f"{label} divisor must be positive, got {divisor}")
    quotient, remainder = divmod(value, divisor)
    if remainder:
        raise RuntimeError(f"{label}={value} is not divisible by {divisor}")
    return quotient


def build_qwen3_asr_pattern_context(
    *,
    inspected: Qwen3AsrInspectedModel,
) -> dict[str, object]:
    spec = inspected.spec
    reflection = inspected.reflection
    shader_variants = _export_shader_variants()
    audio_layer_ops = _audio_layer_ops()
    text_layer_ops = _text_layer_ops()
    audio_tower_ops = _audio_tower_ops()
    text_prefill_ops = _text_prefill_ops()
    text_decode_ops = _text_decode_ops()
    token_select_ops = _token_select_ops()
    token_store_ops = _token_store_ops()
    audio_layer_fields = tensor_fields_from_reflected_static_nodes(
        reflection=reflection,
        module_prefix="thinker.audio_tower.layers.0",
        nodes=audio_layer_ops,
        shader_variants=shader_variants,
        relative_parameter_sources=True,
        external_fields=("hidden_states", "cu_seqlens"),
    )
    text_layer_fields = tensor_fields_from_reflected_static_nodes(
        reflection=reflection,
        module_prefix="thinker.model.layers.0",
        nodes=text_layer_ops,
        shader_variants=shader_variants,
        relative_parameter_sources=True,
        external_fields=("hidden", "rope_cos", "rope_sin", "cache_position"),
        role_overrides={"key_cache": "state", "value_cache": "state"},
    )
    audio_tower_fields = tensor_fields_from_reflected_static_nodes(
        reflection=reflection,
        module_prefix="thinker.audio_tower",
        nodes=audio_tower_ops,
        shader_variants=shader_variants,
        relative_parameter_sources=True,
        extra_fields=(TensorFieldPattern("layers"),),
        role_overrides={"last_hidden_state": "output"},
    )
    text_prefill_fields = _text_prefill_fields(
        nodes=text_prefill_ops,
        shader_variants=shader_variants,
        reflection=reflection,
    )
    text_decode_fields = _text_decode_fields(
        nodes=text_decode_ops,
        shader_variants=shader_variants,
        reflection=reflection,
    )
    token_select_fields = tensor_scaffold_fields_from_static_nodes(
        token_select_ops,
        shader_variants=shader_variants,
        external_fields=("logits",),
        role_overrides={"next_token": "output", "done": "output"},
    )
    token_store_fields = tensor_scaffold_fields_from_static_nodes(
        token_store_ops,
        shader_variants=shader_variants,
        role_overrides={
            "generated_tokens": "state",
            "generated_length": "state",
            "stopped": "state",
        },
    )
    return {
        "spec": spec,
        "defaults": spec.to_template_defaults(),
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
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = export_qwen3_asr_package(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        config_json=args.config_json,
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
        shader_import="torch2vk.exportv2.shaders",
        nodes=_audio_tower_ops(),
        return_expr="tensors.last_hidden_state",
        layer_loop_target="aten.torch2vk.audio_encoder_layer_loop.default",
        layer_nodes=_audio_layer_ops(),
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
        shader_import="torch2vk.exportv2.shaders",
        nodes=_text_prefill_ops(),
        layer_loop_target="aten.torch2vk.text_decoder_layer_loop.default",
        layer_nodes=_text_layer_ops(),
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
        shader_import="torch2vk.exportv2.shaders",
        nodes=_text_decode_ops(),
        layer_loop_target="aten.torch2vk.text_decoder_layer_loop.default",
        layer_nodes=_text_layer_ops_decode(),
        layer_state_field="hidden",
        layer_state_init="tensors.inputs_embeds",
        layer_external_fields={
            "rope_cos": "tensors.rope_cos",
            "rope_sin": "tensors.rope_sin",
            "cache_position": "tensors.cache_position",
        },
        extra_params="    **_kw,",
    )


def _text_frame_fields() -> tuple[TensorFieldPattern, ...]:
    return (
        TensorFieldPattern("prefill"),
        TensorFieldPattern("decode"),
        TensorFieldPattern("token_select"),
        TensorFieldPattern("token_store"),
    )


def _text_prefill_fields(
    *,
    nodes: Sequence[Qwen3AsrNode],
    shader_variants: Mapping[str, ShaderVariant],
    reflection: TorchModuleReflection,
) -> tuple[TensorFieldPattern, ...]:
    return tensor_fields_from_reflected_static_nodes(
        reflection=reflection,
        module_prefix="thinker",
        nodes=nodes,
        shader_variants=shader_variants,
        relative_parameter_sources=False,
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
        field_order=(
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
    nodes: Sequence[Qwen3AsrNode],
    shader_variants: Mapping[str, ShaderVariant],
    reflection: TorchModuleReflection,
) -> tuple[TensorFieldPattern, ...]:
    return tensor_fields_from_reflected_static_nodes(
        reflection=reflection,
        module_prefix="thinker",
        nodes=nodes,
        shader_variants=shader_variants,
        relative_parameter_sources=False,
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
        field_order=(
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


def _audio_tower_ops() -> tuple[Qwen3AsrNode, ...]:
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
        _op("aten.add.Tensor", ("conv_out",), ("conv_out_add_position",)),
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


def _audio_layer_ops() -> tuple[Qwen3AsrNode, ...]:
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
        _op("aten.add.Tensor", ("hidden_states", "out_proj"), ("self_attn_residual",)),
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
        _op("aten.add.Tensor", ("self_attn_residual", "fc2"), ("output",)),
    )


def _text_layer_ops() -> tuple[Qwen3AsrNode, ...]:
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


def _text_layer_ops_decode() -> tuple[Qwen3AsrNode, ...]:
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
        _op(
            "aten.linear.default", ("post_attention_layernorm", "gate_proj_weight"), ("gate_proj",)
        ),
        _op("aten.linear.default", ("post_attention_layernorm", "up_proj_weight"), ("up_proj",)),
        _op("aten.torch2vk.text_swiglu.default", ("gate_proj", "up_proj"), ("swiglu",)),
        _op("aten.linear.default", ("swiglu", "down_proj_weight"), ("down_proj",)),
        _op("aten.add.Tensor", ("attn_residual", "down_proj"), ("output",)),
    )


def _text_prefill_ops() -> tuple[Qwen3AsrNode, ...]:
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


def _text_decode_ops() -> tuple[Qwen3AsrNode, ...]:
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


def _token_select_ops() -> tuple[Qwen3AsrNode, ...]:
    return (
        _op(
            "aten.torch2vk.greedy_argmax.default",
            ("logits", "eos_token_ids"),
            ("next_token", "done"),
        ),
    )


def _token_store_ops() -> tuple[Qwen3AsrNode, ...]:
    return (
        _op(
            "aten.torch2vk.token_store.default",
            ("next_token", "token_index", "done"),
            ("generated_tokens", "generated_length", "stopped"),
        ),
    )


def _export_shader_variants() -> dict[str, ShaderVariant]:
    return shader_variants_from_module(export_shaders)


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
