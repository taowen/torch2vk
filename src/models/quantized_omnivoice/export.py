"""Export quantized OmniVoice submodules to generated Vulkan code.

The generation boundary mirrors exported_omnivoice: the Python loop remains in
run.py, while LLM forward and audio_head are generated as reusable dispatch
libraries. Weight declarations match export_gguf.py: large linears use Q4_K_M
unless they are explicitly Q8_0.

Run from project root:
    uv run python -m models.quantized_omnivoice.export
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import torch
import transformers.integrations.sdpa_attention as sdpa_attention_mod
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoTokenizer

from models.exported_omnivoice.pytorch_modules import LlmForwardModule
from models.exported_omnivoice.custom_shaders import (
    OMNIVOICE_CFG_SCORE_F32,
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
)
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from models.quantized_omnivoice.custom_shaders import (
    OMNIVOICE_INPUT_EMBED_Q8_0_F32,
)
from models.quantized_omnivoice.input_prep import DEFAULT_TEXT, prepare_omnivoice_inputs
from models.quantized_omnivoice.quantization import (
    Q8_TENSOR_NAMES,
    omnivoice_q4_k_m_q6_tensor_names,
)
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.export import (
    LayerLoopHint,
    Q4KMWeightQuantization,
    Q4_K_M_REGISTRY,
    Q8_0_REGISTRY,
    ReferencePolicy,
    bind_dispatch_function_to_tensors,
    cast_floating_tensors,
    clear_python_modules,
    clear_shader_package,
    count_python_modules,
    export_submodule,
    generate_dispatch_function_source,
    generate_looped_dispatch_function_source,
    generate_looped_tensor_class_sources,
    generate_tensor_class_source,
    module_floating_dtype,
    rename_shader_variant,
    render_exported_reference_function,
    render_model_dispatch_module,
    render_reference_function,
    render_reference_loader,
    render_reference_module,
    write_shader_file,
)
from torch2vk.export.registry import ShaderRegistry
from torch2vk.export.tensor_codegen import render_tensor_module
from torch2vk.runtime.shader import ShaderVariant


MODEL_PACKAGE = "models.quantized_omnivoice"
_TEMPLATE_DIR = Path(__file__).with_name("templates")

_JINJA = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    loader=FileSystemLoader(_TEMPLATE_DIR),
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _render_template(template_name: str, **context: object) -> str:
    return _JINJA.get_template(template_name).render(**context)


def _to_class_name(plan_name: str) -> str:
    base = plan_name.removeprefix("run_")
    return "".join(part.capitalize() for part in base.split("_")) + "Tensors"


def _tensor_file_name(class_name: str) -> str:
    if not class_name.endswith("Tensors"):
        raise ValueError(f"tensor class name must end with Tensors: {class_name}")
    stem = class_name.removesuffix("Tensors")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _load_model_and_shapes() -> tuple[OmniVoice, OmniVoiceConfig, dict[str, int]]:
    model_dir = resolve_cached_model(REPO_ID)
    config = OmniVoiceConfig(**json.loads((model_dir / "config.json").read_text()))
    model = cast(OmniVoice, OmniVoice(config).float().cuda())  # pyright: ignore[reportCallIssue]

    llm_config = config.llm_config
    if llm_config is None:
        raise ValueError("OmniVoice config requires llm_config")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prepared = prepare_omnivoice_inputs(
        text=DEFAULT_TEXT,
        tokenizer=tokenizer,
        config=config,
    )

    return model, config, {
        "batch_size": 2,
        "seq_len": prepared.seq_len,
        "target_len": prepared.target_len,
        "hidden_size": llm_config.hidden_size,
        "head_dim": llm_config.head_dim,
        "num_attention_heads": llm_config.num_attention_heads,
        "num_key_value_heads": llm_config.num_key_value_heads,
        "num_hidden_layers": llm_config.num_hidden_layers,
        "text_vocab_size": cast(torch.nn.Embedding, model.get_input_embeddings()).weight.shape[0],
        "num_audio_codebook": config.num_audio_codebook,
        "audio_vocab_size": config.audio_vocab_size,
    }


def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    tensors_dir = output_dir / "tensors"
    dispatch_dir = output_dir / "dispatch"
    shaders_dir.mkdir(exist_ok=True)
    clear_shader_package(shaders_dir)
    tensors_dir.mkdir(exist_ok=True)
    dispatch_dir.mkdir(exist_ok=True)
    clear_python_modules(tensors_dir)
    clear_python_modules(dispatch_dir)

    print("Loading model and computing shapes...")
    model, config, shapes = _load_model_and_shapes()

    batch = shapes["batch_size"]
    seq_len = shapes["seq_len"]
    hidden_size = shapes["hidden_size"]
    head_dim = shapes["head_dim"]
    num_layers = shapes["num_hidden_layers"]
    quantized_weights = Q4KMWeightQuantization(
        q6_tensor_names=frozenset(omnivoice_q4_k_m_q6_tensor_names(num_layers)),
        q8_tensor_names=frozenset(Q8_TENSOR_NAMES),
    )

    custom_variants = (
        OMNIVOICE_INPUT_EMBED_Q8_0_F32,
        OMNIVOICE_CFG_SCORE_F32,
        OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
    )
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

    for variant in custom_variants:
        write_generated_shader(variant)

    reference_functions: list[str] = []
    loader_fields: list[str] = []
    loader_sources: list[str] = []

    reference_functions.append(render_reference_function(
        name="input_embed",
        reference_source="reference",
        tensors="model_tensors()",
        frame_name="omnivoice.step.{step:04d}.input_embed",
        policy="q8_tensor",
        input_bindings={
            "input_ids": "batch_input_ids",
            "audio_mask": "batch_audio_mask",
        },
        output_bindings={"hidden_states": "llm_forward.hidden_states"},
        needs_reference=True,
    ))
    reference_functions.append(render_reference_function(
        name="token_score",
        reference_source="reference",
        tensors="model_tensors()",
        frame_name="omnivoice.step.{step:04d}.token_score",
        policy={"candidate_tokens": "token", "candidate_scores": "tensor"},
        input_bindings={
            "logits": "audio_head.linear",
            "tokens": "tokens",
            "audio_mask_id": "audio_mask_id",
            "rng_seed": "rng_seed",
            "step_index": "step_index",
        },
        output_bindings={
            "candidate_scores": "candidate_scores",
        },
        needs_reference=True,
    ))
    reference_functions.append(render_reference_function(
        name="token_update",
        reference_source="reference",
        tensors="model_tensors()",
        frame_name="omnivoice.step.{step:04d}.token_update",
        policy="token",
        input_bindings={
            "tokens": "tokens",
            "batch_input_ids": "batch_input_ids",
            "candidate_tokens": "candidate_tokens",
            "candidate_scores": "candidate_scores",
            "unmask_count": "unmask_count",
        },
        output_bindings={
            "tokens": "tokens",
            "batch_input_ids": "batch_input_ids",
        },
        needs_reference=True,
    ))

    def export_one(
        name: str,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        *,
        kwargs: dict[str, object] | None = None,
        weight_prefix: str = "",
        layer_loop: LayerLoopHint | None = None,
        reference_input_bindings: dict[str, str] | None = None,
        reference_output_bindings: dict[str, str] | None = None,
        reference_tensors: str | None = None,
        reference_name: str | None = None,
        reference_policy: ReferencePolicy = "tensor",
        reference_module: str | None = None,
        export_registry: ShaderRegistry,
    ) -> None:
        module = module.float()
        export_dtype = module_floating_dtype(module)
        if export_dtype is not None:
            args = cast_floating_tensors(args, export_dtype)
            kwargs = cast_floating_tensors(kwargs, export_dtype)
        prog = export_submodule(module, args=args, kwargs=kwargs)
        class_name = _to_class_name(name)
        function_name = name.removeprefix("run_")
        tensor_file = _tensor_file_name(class_name)
        reference_source = "reference"
        if reference_module is not None:
            loader_fields.append(function_name)
            loader_sources.append(
                render_reference_loader(
                    field=function_name,
                    module_path=reference_module,
                )
            )
            reference_source = f"_load_{function_name}()"
        reference_functions.append(render_exported_reference_function(
            prog,
            name=function_name,
            reference_source=reference_source,
            tensors=reference_tensors if reference_tensors is not None else f"model_tensors().{function_name}",
            frame_name=reference_name if reference_name is not None else function_name,
            policy=reference_policy,
            input_bindings=reference_input_bindings,
            output_bindings=reference_output_bindings,
        ))

        if layer_loop is None:
            tensor_source = generate_tensor_class_source(
                prog,
                class_name=class_name,
                function_name=f"create_{function_name}",
                weight_prefix=weight_prefix,
                registry=export_registry,
                weight_quantization=quantized_weights,
            )
            (tensors_dir / f"{tensor_file}.py").write_text(render_tensor_module([tensor_source]))
            function_source, shader_imports, used_variants = generate_dispatch_function_source(
                prog,
                class_name=class_name,
                function_name=name,
                shader_package=f"{MODEL_PACKAGE}.shaders",
                registry=export_registry,
                weight_quantization=quantized_weights,
            )
        else:
            parent_source, layer_source = generate_looped_tensor_class_sources(
                prog,
                parent_class_name=class_name,
                layer_class_name="LlmLayerTensors",
                parent_function_name=f"create_{function_name}",
                layer_function_name="create_llm_layer",
                weight_prefix=weight_prefix,
                hint=layer_loop,
                registry=export_registry,
                weight_quantization=quantized_weights,
            )
            (tensors_dir / f"{tensor_file}.py").write_text(
                render_tensor_module([layer_source, parent_source])
            )
            function_source, shader_imports, used_variants = generate_looped_dispatch_function_source(
                prog,
                parent_class_name=class_name,
                layer_class_name="LlmLayerTensors",
                function_name=name,
                weight_prefix=weight_prefix,
                hint=layer_loop,
                registry=export_registry,
                weight_quantization=quantized_weights,
            )

        rename_map: dict[str, str] = {}
        for variant in used_variants.values():
            existing = seen_shader_variants.get(variant.name)
            if existing is None:
                write_generated_shader(variant)
                continue
            if existing.contract == variant.contract:
                continue
            renamed = rename_shader_variant(variant, f"{function_name}_{variant.name}")
            rename_map[variant.name] = renamed.name
            write_generated_shader(renamed)

        for old_name in sorted(rename_map, key=len, reverse=True):
            new_name = rename_map[old_name]
            function_source = re.sub(
                rf"\b{re.escape(old_name.upper())}\b",
                new_name.upper(),
                function_source,
            )
            if old_name in shader_imports:
                shader_imports[new_name] = new_name.upper()
                del shader_imports[old_name]

        (dispatch_dir / f"{function_name}.py").write_text(
            render_model_dispatch_module(
                model_package=MODEL_PACKAGE,
                function_name=name,
                tensor_file=tensor_file,
                tensor_class=class_name,
                tensor_expr=f"model_tensors().{function_name}",
                shader_imports=shader_imports,
                function_source=bind_dispatch_function_to_tensors(function_source),
                uses_quantized_linear_dispatch="run_quantized_linear(" in function_source,
            )
        )
        print(f"  {name}: {len(used_variants)} shaders")

    original_use_gqa = sdpa_attention_mod.use_gqa_in_sdpa
    sdpa_attention_mod.use_gqa_in_sdpa = lambda *args, **kwargs: True
    try:
        export_one(
            "run_llm_forward",
            LlmForwardModule(model),
            args=(
                torch.zeros(batch, seq_len, hidden_size, device="cuda"),
                torch.zeros(batch, seq_len, head_dim, device="cuda"),
                torch.zeros(batch, seq_len, head_dim, device="cuda"),
                torch.zeros(batch, 1, seq_len, seq_len, device="cuda"),
            ),
            weight_prefix="llm.",
            layer_loop=LayerLoopHint(layer_prefix="layers", num_layers=num_layers),
            reference_input_bindings={
                "hidden_states": "llm_forward.hidden_states",
                "cos": "rope.cos",
                "sin": "rope.sin",
                "attention_mask": "attention_mask",
            },
            reference_output_bindings={"mul_365": "llm_forward.mul_365"},
            reference_tensors="model_tensors()",
            reference_name="omnivoice.step.{step:04d}.llm_forward",
            reference_policy="q4_tensor",
            export_registry=Q4_K_M_REGISTRY,
        )
    finally:
        sdpa_attention_mod.use_gqa_in_sdpa = original_use_gqa

    export_one(
        "run_audio_head",
        model.audio_heads,
        args=(torch.zeros(batch, seq_len, hidden_size, device="cuda"),),
        weight_prefix="audio_heads.",
        reference_module="audio_heads",
        reference_input_bindings={"input": "audio_head.input"},
        reference_output_bindings={"linear": "audio_head.linear"},
        reference_tensors="model_tensors()",
        reference_name="omnivoice.step.{step:04d}.audio_head",
        reference_policy="q4_tensor",
        export_registry=Q4_K_M_REGISTRY,
    )
    print(f"\n  {shader_file_count} shader files written")

    (tensors_dir / "model.py").write_text(
        _render_template(
            "model.py.j2",
            model_package=MODEL_PACKAGE,
            seq_len=shapes["seq_len"],
            batch_size=shapes["batch_size"],
            hidden_size=shapes["hidden_size"],
            head_dim=shapes["head_dim"],
            text_vocab_size=shapes["text_vocab_size"],
            num_audio_codebook=shapes["num_audio_codebook"],
            audio_vocab_size=shapes["audio_vocab_size"],
        )
    )
    print(f"  tensors/ written ({count_python_modules(tensors_dir)} files)")
    print(f"  dispatch/ written ({count_python_modules(dispatch_dir)} files)")

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package=MODEL_PACKAGE,
            model_imports=["from omnivoice.models.omnivoice import OmniVoice"],
            model_type="OmniVoice",
            reference_functions=reference_functions,
            loader_fields=loader_fields,
            loader_sources=loader_sources,
        )
    )
    print("  reference.py written")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
