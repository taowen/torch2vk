"""Export OmniVoice submodules → shaders/, tensors/, dispatch/.

Generates Python source files for the TTS pipeline (embed + LLM layers + audio head).
OmniVoice uses iterative masked decoding (32 steps, full sequence, no KV cache).

Run from project root:
    .venv/bin/python -m models.exported_omnivoice.export
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import torch
import transformers.integrations.sdpa_attention as _sdpa_attn_mod
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoTokenizer

from models.exported_omnivoice.custom_shaders import (
    OMNIVOICE_CFG_SCORE_F32,
    OMNIVOICE_INPUT_EMBED_F32,
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
)
from models.exported_omnivoice.input_prep import DEFAULT_TEXT, prepare_omnivoice_inputs
from models.exported_omnivoice.pytorch_modules import LlmForwardModule
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.export import (
    LayerLoopHint,
    ReferencePolicy,
    cast_floating_tensors,
    export_submodule,
    module_floating_dtype,
    read_checkpoint_dtypes,
    set_module_checkpoint_dtypes,
)
from torch2vk.export.dispatch_codegen import (
    bind_dispatch_function_to_tensors,
    generate_dispatch_function_source,
    render_model_dispatch_module,
)
from torch2vk.export.reference_codegen import (
    render_exported_reference_function,
    render_reference_function,
    render_reference_loader,
    render_reference_module,
)
from torch2vk.export.shader_codegen import (
    clear_shader_package,
    rename_shader_variant,
    write_shader_file,
    write_shader_init,
)
from torch2vk.export.tensor_codegen import (
    generate_tensor_class_source,
    render_tensor_module,
)
from torch2vk.export.codegen_loop import (
    generate_looped_dispatch_function_source,
    generate_looped_tensor_class_sources,
)
from torch2vk.runtime.shader import ShaderVariant


_TEMPLATE_DIR = Path(__file__).with_name("templates")

_JINJA = Environment(
    autoescape=False,
    loader=FileSystemLoader(_TEMPLATE_DIR),
    keep_trailing_newline=True,
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _to_class_name(plan_name: str) -> str:
    base = plan_name.removeprefix("run_")
    return "".join(p.capitalize() for p in base.split("_")) + "Tensors"


def _tensor_file_name(class_name: str) -> str:
    if not class_name.endswith("Tensors"):
        raise ValueError(f"tensor class name must end with Tensors: {class_name}")
    stem = class_name.removesuffix("Tensors")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _render_template(name: str, **context: object) -> str:
    return _JINJA.get_template(name).render(**context)


# ==============================================================
# Model loading + shape computation
# ==============================================================

def _load_model_and_shapes():
    model_dir = resolve_cached_model(REPO_ID)
    config_data = json.loads((model_dir / "config.json").read_text())
    config = OmniVoiceConfig(**config_data)
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
    seq_len = prepared.seq_len

    shapes = {
        "batch_size": 2,
        "seq_len": seq_len,
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
    return model, config, shapes


# ==============================================================
# Main
# ==============================================================

def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    shaders_dir.mkdir(exist_ok=True)
    clear_shader_package(shaders_dir)
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir(exist_ok=True)
    dispatch_dir = output_dir / "dispatch"
    dispatch_dir.mkdir(exist_ok=True)
    for f in tensors_dir.glob("*.py"):
        f.unlink()
    for f in dispatch_dir.glob("*.py"):
        f.unlink()

    print("Loading model and computing shapes...")
    model, config, shapes = _load_model_and_shapes()
    checkpoint_dtypes = read_checkpoint_dtypes(resolve_cached_model(REPO_ID))

    B = shapes["batch_size"]
    S = shapes["seq_len"]
    H = shapes["hidden_size"]
    D = shapes["head_dim"]
    num_layers = shapes["num_hidden_layers"]

    custom_variants = (
        OMNIVOICE_INPUT_EMBED_F32,
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
        policy="tensor",
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
        name,
        module,
        args,
        kwargs=None,
        weight_prefix="",
        layer_loop=None,
        reference_input_bindings=None,
        reference_output_bindings=None,
        reference_tensors=None,
        reference_name=None,
        reference_policy: ReferencePolicy = "tensor",
        reference_module=None,
    ):
        set_module_checkpoint_dtypes(
            module,
            weight_prefix=weight_prefix,
            checkpoint_dtypes=checkpoint_dtypes,
        )
        export_dtype = module_floating_dtype(module)
        if export_dtype is not None:
            args = cast_floating_tensors(args, export_dtype)
            kwargs = cast_floating_tensors(kwargs, export_dtype)
        prog = export_submodule(module, args=args, kwargs=kwargs)
        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        tensor_file = _tensor_file_name(cls_name)
        reference_source = "reference"
        if reference_module is not None:
            loader_fields.append(func_name)
            loader_sources.append(
                render_reference_loader(
                    field=func_name,
                    module_path=reference_module,
                )
            )
            reference_source = f"_load_{func_name}()"
        reference_functions.append(render_exported_reference_function(
            prog,
            name=func_name,
            reference_source=reference_source,
            tensors=reference_tensors if reference_tensors is not None else "model_tensors()",
            frame_name=reference_name if reference_name is not None else func_name,
            policy=reference_policy,
            input_bindings=reference_input_bindings,
            output_bindings=reference_output_bindings,
        ))

        if layer_loop is not None:
            layer_cls_name = "LlmLayerTensors"
            layer_func_name = "create_llm_layer"

            parent_src, layer_src = generate_looped_tensor_class_sources(
                prog,
                parent_class_name=cls_name,
                layer_class_name=layer_cls_name,
                parent_function_name=f"create_{func_name}",
                layer_function_name=layer_func_name,
                weight_prefix=weight_prefix,
                hint=layer_loop,
            )
            (tensors_dir / f"{tensor_file}.py").write_text(
                render_tensor_module([layer_src, parent_src])
            )

            func_src, shader_imports, used_variants = generate_looped_dispatch_function_source(
                prog,
                parent_class_name=cls_name,
                layer_class_name=layer_cls_name,
                function_name=name,
                weight_prefix=weight_prefix,
                hint=layer_loop,
            )
        else:
            tensor_src = generate_tensor_class_source(
                prog,
                class_name=cls_name,
                function_name=f"create_{func_name}",
                weight_prefix=weight_prefix,
            )
            (tensors_dir / f"{tensor_file}.py").write_text(render_tensor_module([tensor_src]))

            func_src, shader_imports, used_variants = generate_dispatch_function_source(
                prog,
                class_name=cls_name,
                function_name=name,
                shader_package="models.exported_omnivoice.shaders",
            )

        rename_map: dict[str, str] = {}
        for v in used_variants.values():
            existing = seen_shader_variants.get(v.name)
            if existing is None:
                write_generated_shader(v)
                continue
            if existing.contract == v.contract:
                continue
            renamed = rename_shader_variant(v, f"{func_name}_{v.name}")
            rename_map[v.name] = renamed.name
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
                model_package="models.exported_omnivoice",
                function_name=name,
                tensor_file=tensor_file,
                tensor_class=cls_name,
                tensor_expr=f"model_tensors().{func_name}",
                shader_imports=shader_imports,
                function_source=bind_dispatch_function_to_tensors(func_src),
            )
        )

        print(f"  {name}: {len(used_variants)} shaders")

    # --- LLM forward (layers + norm) ---
    # Patch use_gqa_in_sdpa to return True so repeat_kv is skipped and K/V pass
    # directly to SDPA with NK=8. Our SDPA shader handles GQA natively.
    _orig_use_gqa = _sdpa_attn_mod.use_gqa_in_sdpa
    _sdpa_attn_mod.use_gqa_in_sdpa = lambda *a, **kw: True

    export_one(
        "run_llm_forward",
        LlmForwardModule(model).float(),
        args=(
            torch.zeros(B, S, H, device="cuda"),
            torch.zeros(B, S, D, device="cuda"),
            torch.zeros(B, S, D, device="cuda"),
            torch.zeros(B, 1, S, S, device="cuda"),
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
    )

    _sdpa_attn_mod.use_gqa_in_sdpa = _orig_use_gqa

    # --- Audio head ---
    export_one(
        "run_audio_head",
        model.audio_heads.float(),
        args=(torch.zeros(B, S, H, device="cuda"),),
        weight_prefix="audio_heads.",
        reference_module="audio_heads",
        reference_input_bindings={"input": "audio_head.input"},
        reference_output_bindings={"linear": "audio_head.linear"},
        reference_tensors="model_tensors()",
        reference_name="omnivoice.step.{step:04d}.audio_head",
    )

    write_shader_init(shaders_dir)
    print(f"\n  {shader_file_count} shader files written")

    # Write model-level tensor wiring.
    model_source = _render_template(
        "model.py.j2",
        seq_len=shapes["seq_len"],
        batch_size=shapes["batch_size"],
        hidden_size=shapes["hidden_size"],
        head_dim=shapes["head_dim"],
        text_vocab_size=shapes["text_vocab_size"],
        text_embedding_dtype=checkpoint_dtypes["llm.embed_tokens.weight"],
        num_audio_codebook=shapes["num_audio_codebook"],
        audio_vocab_size=shapes["audio_vocab_size"],
        audio_embedding_dtype=checkpoint_dtypes["audio_embeddings.weight"],
    )
    (tensors_dir / "model.py").write_text(model_source)
    (tensors_dir / "__init__.py").write_text('"""Generated tensor package."""\n')
    tensor_file_count = len([path for path in tensors_dir.glob("*.py") if path.name != "__init__.py"])
    print(f"  tensors/ written ({tensor_file_count} files)")

    (dispatch_dir / "__init__.py").write_text('"""Generated dispatch package."""\n')
    dispatch_file_count = len([path for path in dispatch_dir.glob("*.py") if path.name != "__init__.py"])
    print(f"  dispatch/ written ({dispatch_file_count} files)")

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package="models.exported_omnivoice",
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
