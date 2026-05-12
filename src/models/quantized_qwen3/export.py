"""Export standalone Qwen3 submodules to quantized Vulkan code.

Run from project root:
    uv run python -m models.quantized_qwen3.export
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import cast

import torch
from torch import nn
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoConfig, AutoModelForCausalLM

from models.hf_cache import resolve_cached_model
from models.quantized_qwen3.export_gguf import REPO_ID
from models.quantized_qwen3.input_prep import (
    DEFAULT_PROMPT,
    load_qwen3_tokenizer,
    prepare_qwen3_inputs,
)
from models.quantized_qwen3.quantization import Q6_TENSOR_NAMES, Q8_TENSOR_NAMES
from torch2vk.export import (
    KVCacheInjectHint,
    Q4KMWeightQuantization,
    Q4_K_M_REGISTRY,
    bind_dispatch_function_to_tensors,
    cast_floating_tensors,
    clear_shader_package,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
    module_floating_dtype,
    rename_shader_variant,
    render_model_dispatch_module,
    write_shader_file,
    write_shader_init,
)
from torch2vk.export.graph import inject_kv_cache
from torch2vk.export.registry import DEFAULT_REGISTRY, ShaderRegistry
from torch2vk.export.shaders.qwen3_asr_token_select_f32 import (
    QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
)
from torch2vk.export.shaders.qwen3_asr_token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from torch2vk.export.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from torch2vk.export.tensor_codegen import render_tensor_module
from torch2vk.runtime.shader import ShaderContract, ShaderVariant


MODEL_PACKAGE = "models.quantized_qwen3"
_QUANTIZED_WEIGHTS = Q4KMWeightQuantization(
    q6_tensor_names=frozenset(Q6_TENSOR_NAMES),
    q8_tensor_names=frozenset(Q8_TENSOR_NAMES),
)
_TEMPLATE_DIR = Path(__file__).with_name("templates")
_JINJA = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    loader=FileSystemLoader(_TEMPLATE_DIR),
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _retag_shader_variant(
    variant: ShaderVariant,
    *,
    name: str,
    family: str,
    class_name: str,
) -> ShaderVariant:
    return ShaderVariant(
        name=name,
        family=family,
        contract=ShaderContract(
            class_name=class_name,
            shader_name=name,
            fields=variant.contract.fields,
            dispatch=variant.contract.dispatch,
            push_constants=variant.contract.push_constants,
            params_buffer=variant.contract.params_buffer,
        ),
        source=variant.source,
        precompiled_spv_path=variant.precompiled_spv_path,
        specialization_constants=variant.specialization_constants,
        include_dirs=variant.include_dirs,
        compile_defines=variant.compile_defines,
        execution_requirements=variant.execution_requirements,
    )


QWEN3_TOKEN_SELECT_GREEDY_F32 = _retag_shader_variant(
    QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
    name="qwen3_token_select_greedy_f32",
    family="qwen3.text",
    class_name="Qwen3TokenSelectGreedyF32Program",
)
QWEN3_TOKEN_STORE_EOS_F32 = _retag_shader_variant(
    QWEN3_ASR_TOKEN_STORE_EOS_F32,
    name="qwen3_token_store_eos_f32",
    family="qwen3.text",
    class_name="Qwen3TokenStoreEosF32Program",
)
QWEN3_SLICE_LAST_TOKEN_F16 = _retag_shader_variant(
    SLICE_LAST_TOKEN_F16,
    name="slice_last_token_f16",
    family="qwen3.text",
    class_name="Qwen3SliceLastTokenF16Program",
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


def _dispatch_tensor_expr(func_name: str) -> str:
    if func_name in ("text_layer", "decode_layer"):
        return f"model_tensors().{func_name}s[layer_idx]"
    return f"model_tensors().{func_name}"


def _dispatch_parameters_source(func_name: str) -> str:
    if func_name in ("text_layer", "decode_layer"):
        return ", layer_idx: int"
    return ""


def _load_model_and_shapes() -> tuple[torch.nn.Module, dict[str, int]]:
    model_dir = resolve_cached_model(REPO_ID)
    tokenizer = load_qwen3_tokenizer(model_dir)
    prepared = prepare_qwen3_inputs(tokenizer=tokenizer, prompt=DEFAULT_PROMPT)
    payload = json.loads((Path(model_dir) / "config.json").read_text())

    with open(os.devnull, "w") as devnull:
        stdout_fd, stderr_fd = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            config = AutoConfig.from_pretrained(model_dir)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)

    return model, {
        "prompt_length": prepared.prompt_length,
        "max_sequence_length": prepared.prompt_length + 128,
        "hidden_size": int(payload["hidden_size"]),
        "head_dim": int(payload["head_dim"]),
        "num_hidden_layers": int(payload["num_hidden_layers"]),
        "num_key_value_heads": int(payload["num_key_value_heads"]),
    }


def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    tensors_dir = output_dir / "tensors"
    dispatch_dir = output_dir / "dispatch"
    shaders_dir.mkdir(exist_ok=True)
    tensors_dir.mkdir(exist_ok=True)
    dispatch_dir.mkdir(exist_ok=True)
    clear_shader_package(shaders_dir)
    for directory in (tensors_dir, dispatch_dir):
        for source in directory.glob("*.py"):
            source.unlink()

    print("Loading Qwen3 model and computing shapes...")
    model, shapes = _load_model_and_shapes()
    text_model = cast(nn.Module, model.get_submodule("model"))
    embed_tokens = cast(nn.Module, text_model.get_submodule("embed_tokens"))
    layers = cast(nn.ModuleList, text_model.get_submodule("layers"))
    norm = cast(nn.Module, text_model.get_submodule("norm"))
    lm_head = cast(nn.Module, model.get_submodule("lm_head"))

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

    for variant in (
        QWEN3_TOKEN_SELECT_GREEDY_F32,
        QWEN3_TOKEN_STORE_EOS_F32,
        QWEN3_SLICE_LAST_TOKEN_F16,
    ):
        write_generated_shader(variant)

    def export_one(
        name: str,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        *,
        kwargs: dict[str, object] | None = None,
        weight_prefix: str = "",
        kv_inject: KVCacheInjectHint | None = None,
        export_registry: ShaderRegistry = DEFAULT_REGISTRY,
        weight_quantization: Q4KMWeightQuantization | None = None,
        shape_exprs: dict[int, str] | None = None,
    ) -> None:
        module = module.float()
        export_dtype = module_floating_dtype(module)
        if export_dtype is not None:
            args = cast_floating_tensors(args, export_dtype)
            kwargs = cast_floating_tensors(kwargs, export_dtype)
        prog = export_submodule(module, args=args, kwargs=kwargs)
        if kv_inject is not None:
            inject_kv_cache(prog, kv_inject)

        class_name = _to_class_name(name)
        function_name = name.removeprefix("run_")
        tensor_file = _tensor_file_name(class_name)
        tensor_src = generate_tensor_class_source(
            prog,
            class_name=class_name,
            function_name=f"create_{function_name}",
            weight_prefix=weight_prefix,
            registry=export_registry,
            weight_quantization=weight_quantization,
            shape_exprs=shape_exprs,
        )
        (tensors_dir / f"{tensor_file}.py").write_text(render_tensor_module([tensor_src]))

        func_src, shader_imports, used_variants = generate_dispatch_function_source(
            prog,
            class_name=class_name,
            function_name=name,
            shader_package=f"{MODEL_PACKAGE}.shaders",
            registry=export_registry,
            weight_quantization=weight_quantization,
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
            old_const = old_name.upper()
            new_const = new_name.upper()
            func_src = re.sub(rf"\b{re.escape(old_const)}\b", new_const, func_src)
            if old_name in shader_imports:
                shader_imports[new_name] = new_const
                del shader_imports[old_name]

        (dispatch_dir / f"{function_name}.py").write_text(
            render_model_dispatch_module(
                model_package=MODEL_PACKAGE,
                function_name=name,
                tensor_file=tensor_file,
                tensor_class=class_name,
                tensor_expr=_dispatch_tensor_expr(function_name),
                shader_imports=shader_imports,
                function_source=bind_dispatch_function_to_tensors(func_src),
                parameters_source=_dispatch_parameters_source(function_name),
                uses_q4_q6_dispatch="_linear_q4_or_q6" in func_src,
            )
        )
        print(f"  {name}: {len(used_variants)} shaders")

    prompt_length = shapes["prompt_length"]
    max_sequence_length = shapes["max_sequence_length"]
    hidden_size = shapes["hidden_size"]
    head_dim = shapes["head_dim"]
    text_shape_exprs = {prompt_length: "sequence_length"}
    text_layer_shape_exprs = {
        prompt_length: "sequence_length",
        max_sequence_length: "max_sequence_length",
    }
    decode_layer_shape_exprs = {max_sequence_length: "max_sequence_length"}

    export_one(
        "run_embed_tokens",
        embed_tokens.float(),
        args=(torch.zeros((1, prompt_length), dtype=torch.long, device="meta"),),
        weight_prefix="model.embed_tokens.",
        export_registry=Q4_K_M_REGISTRY,
        weight_quantization=_QUANTIZED_WEIGHTS,
        shape_exprs=text_shape_exprs,
    )
    export_one(
        "run_text_layer",
        layers[0],
        args=(torch.zeros(1, prompt_length, hidden_size, device="meta"),),
        kwargs={
            "position_embeddings": (
                torch.zeros(1, prompt_length, head_dim, device="meta"),
                torch.zeros(1, prompt_length, head_dim, device="meta"),
            ),
            "past_key_values": None,
            "attention_mask": None,
        },
        weight_prefix="model.layers.0.",
        kv_inject=KVCacheInjectHint(phase="prefill", max_seq_len=max_sequence_length),
        export_registry=Q4_K_M_REGISTRY,
        weight_quantization=_QUANTIZED_WEIGHTS,
        shape_exprs=text_layer_shape_exprs,
    )
    export_one(
        "run_text_norm",
        norm.float(),
        args=(torch.zeros(1, prompt_length, hidden_size, device="meta"),),
        weight_prefix="model.norm.",
        shape_exprs=text_shape_exprs,
    )
    export_one(
        "run_lm_head",
        lm_head.float(),
        args=(torch.zeros(1, 1, hidden_size, device="meta"),),
        weight_prefix="lm_head.",
        export_registry=Q4_K_M_REGISTRY,
        weight_quantization=_QUANTIZED_WEIGHTS,
    )
    export_one(
        "run_decode_embed",
        embed_tokens.float(),
        args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
        weight_prefix="model.embed_tokens.",
        export_registry=Q4_K_M_REGISTRY,
        weight_quantization=_QUANTIZED_WEIGHTS,
    )
    export_one(
        "run_decode_layer",
        layers[0],
        args=(torch.zeros(1, 1, hidden_size, device="meta"),),
        kwargs={
            "position_embeddings": (
                torch.zeros(1, 1, head_dim, device="meta"),
                torch.zeros(1, 1, head_dim, device="meta"),
            ),
            "past_key_values": None,
            "attention_mask": None,
        },
        weight_prefix="model.layers.0.",
        kv_inject=KVCacheInjectHint(phase="decode", max_seq_len=max_sequence_length),
        export_registry=Q4_K_M_REGISTRY,
        weight_quantization=_QUANTIZED_WEIGHTS,
        shape_exprs=decode_layer_shape_exprs,
    )
    export_one(
        "run_decode_norm",
        norm.float(),
        args=(torch.zeros(1, 1, hidden_size, device="meta"),),
        weight_prefix="model.norm.",
    )
    export_one(
        "run_decode_lm_head",
        lm_head.float(),
        args=(torch.zeros(1, 1, hidden_size, device="meta"),),
        weight_prefix="lm_head.",
        export_registry=Q4_K_M_REGISTRY,
        weight_quantization=_QUANTIZED_WEIGHTS,
    )

    write_shader_init(shaders_dir)
    print(f"\n  {shader_file_count} shader files written")
    (tensors_dir / "rope.py").write_text(_render_template("rope.py.j2"))
    (tensors_dir / "model.py").write_text(
        _render_template("model.py.j2", model_package=MODEL_PACKAGE)
    )
    (tensors_dir / "__init__.py").write_text('"""Generated tensor package."""\n')
    (dispatch_dir / "__init__.py").write_text('"""Generated dispatch package."""\n')
    print("  tensors/ and dispatch/ written")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
