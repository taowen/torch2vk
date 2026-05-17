"""Export quantized_qwen3 core Vulkan modules.

Run from project root:
    uv run python -m models.quantized_qwen3.export
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from models.hf_cache import resolve_cached_model
from models.quantized_qwen3.export_gguf import REPO_ID
from models.quantized_qwen3.input_prep import DEFAULT_PROMPT, load_qwen3_tokenizer, prepare_qwen3_inputs
from models.quantized_qwen3.quantization import qwen3_q4_k_m_config
from torch2vk.export import (
    KVCacheInjectHint,
    bind_dispatch_function_to_tensors,
    cast_floating_tensors,
    clear_python_modules,
    clear_shader_package,
    count_python_modules,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
    generate_weight_tensor_class_source,
    inject_kv_cache,
    module_floating_dtype,
    rename_shader_variant,
    render_exported_reference_function,
    render_model_dispatch_module,
    render_reference_module,
    write_shader_file,
)
from torch2vk.export.reference_codegen import render_reference_function
from torch2vk.export.shaders.lm_head_q6_k_argmax_partial_f16 import (
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16,
)
from torch2vk.export.shaders.qwen3_token_select_reduce_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
    QWEN3_TOKEN_SELECT_REDUCE_F32,
)
from torch2vk.export.shaders.qwen3_token_store import QWEN3_TOKEN_STORE_EOS
from torch2vk.export.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from torch2vk.export.tensor_codegen import layer_workspace_keep_fields, render_tensor_module
from torch2vk.quantize import Q4KMQuantizationConfig
from torch2vk.runtime.shader import ShaderVariant


MODEL_PACKAGE = "models.quantized_qwen3"
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


def _dispatch_tensor_expr(func_name: str) -> str:
    if func_name in ("text_layer", "decode_layer"):
        return f"model_tensors().{func_name}s[layer_idx]"
    return f"model_tensors().{func_name}"


def _dispatch_parameters_source(func_name: str) -> str:
    if func_name == "decode_layer":
        return ", layer_idx: int, *, cache_position: int"
    if func_name in ("text_layer", "decode_layer"):
        return ", layer_idx: int"
    return ""


def _dispatch_arguments_source(func_name: str) -> str:
    if func_name == "decode_layer":
        return ", cache_position=cache_position"
    return ""


def _load_model_and_shapes() -> tuple[Qwen3ForCausalLM, object, dict[str, int]]:
    model_dir = resolve_cached_model(REPO_ID)
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    tokenizer = load_qwen3_tokenizer(model_dir)
    prepared = prepare_qwen3_inputs(tokenizer=tokenizer, prompt=DEFAULT_PROMPT)
    prompt_length = prepared.prompt_length
    return model, config, {
        "prompt_length": prompt_length,
        "max_sequence_length": ((prompt_length + 128 + 63) // 64) * 64,
        "hidden_size": int(config.hidden_size),
        "head_dim": int(config.head_dim),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_key_value_heads": int(config.num_key_value_heads),
        "vocab_size": int(config.vocab_size),
        "lm_head_rows": int(config.vocab_size),
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
    clear_python_modules(tensors_dir)
    clear_python_modules(dispatch_dir)

    print("Loading model and computing shapes...")
    model, config, shapes = _load_model_and_shapes()
    q4_k_m_config = qwen3_q4_k_m_config(shapes["num_hidden_layers"])

    custom_shader_variants = (
        LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16,
        SLICE_LAST_TOKEN_F16,
        QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
        QWEN3_TOKEN_SELECT_REDUCE_F32,
        QWEN3_TOKEN_STORE_EOS,
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

    for variant in custom_shader_variants:
        write_generated_shader(variant)

    reference_functions: list[str] = []
    reference_functions.append(render_reference_function(
        name="token_select",
        reference_source="reference",
        tensors="model_tensors()",
        frame_name="{name}",
        policy="token",
        input_bindings={"logits": "logits", "eos_token_ids": "eos_token_ids"},
        output_bindings={"next_token": "next_token", "done": "done"},
        needs_reference=True,
    ))

    def export_one(
        name: str,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, object] | None = None,
        *,
        weight_prefix: str = "",
        kv_inject: KVCacheInjectHint | None = None,
        reference_tensors: str | None = None,
        reference_name: str | None = None,
        quantization_config: Q4KMQuantizationConfig | None = None,
        shape_exprs: dict[int, str] | None = None,
    ) -> None:
        nonlocal shader_file_count
        module = module.float()
        export_dtype = module_floating_dtype(module)
        if export_dtype is not None:
            args = cast_floating_tensors(args, export_dtype)
            kwargs = cast_floating_tensors(kwargs, export_dtype)
        prog = export_submodule(module, args=args, kwargs=kwargs)
        if kv_inject is not None:
            inject_kv_cache(prog, kv_inject)

        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        tensor_file = _tensor_file_name(cls_name)
        reference_functions.append(render_exported_reference_function(
            prog,
            name=func_name,
            reference_source="reference",
            tensors=reference_tensors if reference_tensors is not None else f"model_tensors().{func_name}",
            frame_name=reference_name if reference_name is not None else func_name,
        ))

        tensor_src = generate_tensor_class_source(
            prog,
            class_name=cls_name,
            function_name=f"create_{func_name}",
            weight_prefix=weight_prefix,
            quantization_config=quantization_config,
            shape_exprs=shape_exprs,
        )
        (tensors_dir / f"{tensor_file}.py").write_text(render_tensor_module([tensor_src]))

        func_src, shader_imports, used_variants = generate_dispatch_function_source(
            prog,
            class_name=cls_name,
            function_name=name,
            shader_package=f"{MODEL_PACKAGE}.shaders",
            weight_prefix=weight_prefix,
            quantization_config=quantization_config,
        )

        rename_map: dict[str, str] = {}
        for variant in used_variants.values():
            existing = seen_shader_variants.get(variant.name)
            if existing is None:
                write_generated_shader(variant)
                continue
            if existing.contract == variant.contract:
                continue
            renamed = rename_shader_variant(variant, f"{func_name}_{variant.name}")
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

        (dispatch_dir / f"{func_name}.py").write_text(
            render_model_dispatch_module(
                model_package=MODEL_PACKAGE,
                function_name=name,
                tensor_file=tensor_file,
                tensor_class=cls_name,
                tensor_expr=_dispatch_tensor_expr(func_name),
                shader_imports=shader_imports,
                function_source=bind_dispatch_function_to_tensors(func_src),
                parameters_source=_dispatch_parameters_source(func_name),
                arguments_source=_dispatch_arguments_source(func_name),
                uses_quantized_linear_dispatch="run_quantized_linear(" in func_src,
                workspace_keep_fields=layer_workspace_keep_fields(tensor_src),
            )
        )

        print(f"  {name}: {len(used_variants)} shaders")

    pl = shapes["prompt_length"]
    max_seq = shapes["max_sequence_length"]
    hs = shapes["hidden_size"]
    hd = shapes["head_dim"]
    text_shape_exprs = {pl: "sequence_length"}
    layer_shape_exprs = {pl: "sequence_length", max_seq: "max_sequence_length"}
    decode_shape_exprs = {max_seq: "max_sequence_length"}

    text_model = model.model
    export_one(
        "run_embed_tokens",
        text_model.embed_tokens,
        args=(torch.zeros((1, pl), dtype=torch.long, device="meta"),),
        weight_prefix="model.embed_tokens.",
        quantization_config=q4_k_m_config,
        shape_exprs=text_shape_exprs,
    )
    export_one(
        "run_text_layer",
        text_model.layers[0],
        args=(torch.zeros(1, pl, hs, device="meta"),),
        kwargs={
            "position_embeddings": (
                torch.zeros(1, pl, hd, device="meta"),
                torch.zeros(1, pl, hd, device="meta"),
            ),
            "past_key_values": None,
            "attention_mask": None,
        },
        weight_prefix="model.layers.0.",
        kv_inject=KVCacheInjectHint(phase="prefill", max_seq_len=max_seq),
        reference_tensors="model_tensors().text_layers[layer_idx]",
        reference_name="qwen3.prefill.layer.{layer_idx}",
        quantization_config=q4_k_m_config,
        shape_exprs=layer_shape_exprs,
    )
    export_one(
        "run_text_norm",
        text_model.norm,
        args=(torch.zeros(1, pl, hs, device="meta"),),
        weight_prefix="model.norm.",
        shape_exprs=text_shape_exprs,
    )
    export_one(
        "run_decode_embed",
        text_model.embed_tokens,
        args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
        weight_prefix="model.embed_tokens.",
        quantization_config=q4_k_m_config,
    )
    export_one(
        "run_decode_layer",
        text_model.layers[0],
        args=(torch.zeros(1, 1, hs, device="meta"),),
        kwargs={
            "position_embeddings": (
                torch.zeros(1, 1, hd, device="meta"),
                torch.zeros(1, 1, hd, device="meta"),
            ),
            "past_key_values": None,
            "attention_mask": None,
        },
        weight_prefix="model.layers.0.",
        kv_inject=KVCacheInjectHint(phase="decode", max_seq_len=max_seq),
        reference_tensors="model_tensors().decode_layers[layer_idx]",
        reference_name="qwen3.decode.{step:04d}.layer.{layer_idx}",
        quantization_config=q4_k_m_config,
        shape_exprs=decode_shape_exprs,
    )
    export_one(
        "run_decode_norm",
        text_model.norm,
        args=(torch.zeros(1, 1, hs, device="meta"),),
        weight_prefix="model.norm.",
    )

    lm_head_shape = tuple(int(dim) for dim in model.lm_head.weight.shape)
    (tensors_dir / "lm_head.py").write_text(render_tensor_module([
        generate_weight_tensor_class_source(
            class_name="LmHeadTensors",
            function_name="create_lm_head",
            field_name="p_weight",
            checkpoint_key="lm_head.weight",
            dtype="float32",
            shape=lm_head_shape,
        quantization_config=q4_k_m_config,
        )
    ]))
    (tensors_dir / "rope.py").write_text(_render_template("rope.py.j2"))
    (tensors_dir / "model.py").write_text(_render_template("model.py.j2", model_package=MODEL_PACKAGE))

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package=MODEL_PACKAGE,
            model_imports=["from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM"],
            model_type="Qwen3ForCausalLM",
            reference_functions=reference_functions,
            loader_fields=[],
            loader_sources=[],
        )
    )

    manifest = {
        "repo_id": REPO_ID,
        "prompt": DEFAULT_PROMPT,
        "generated_by": "models.quantized_qwen3.export",
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  {shader_file_count} shader files written")
    print(f"  tensors/ written ({count_python_modules(tensors_dir)} files)")
    print(f"  dispatch/ written ({count_python_modules(dispatch_dir)} files)")
    print("  reference.py written")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
