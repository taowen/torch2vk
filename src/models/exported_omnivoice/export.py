"""Export OmniVoice submodules → shaders/, tensors/, dispatch.py.

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

from models.exported_omnivoice.custom_shaders import (
    OMNIVOICE_CFG_SCORE_F32,
    OMNIVOICE_INPUT_EMBED_F32,
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32,
)
from models.exported_omnivoice.pytorch_modules import LlmForwardModule
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.export import (
    LayerLoopHint,
    export_submodule,
    generate_dispatch_function_source,
    generate_reference_spec,
    generate_tensor_class_source,
)
from torch2vk.export.codegen import (
    render_reference_specs_module,
    render_shader_file,
    render_simple_init,
    render_tensor_helpers,
    render_tensor_module,
)
from torch2vk.export.codegen_loop import (
    generate_looped_dispatch_function_source,
    generate_looped_tensor_class_sources,
)
from torch2vk.runtime.reference import ReferenceSpec
from torch2vk.runtime.shader import ShaderContract, ShaderVariant


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
    seq_len = 300

    shapes = {
        "batch_size": 2,
        "seq_len": seq_len,
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
# Output file assembly
# ==============================================================


def _combine_dispatch(
    dispatch_sources: list[str],
    all_shader_imports: dict[str, str],
    tensor_file_classes: dict[str, list[str]],
) -> str:
    bound_dispatch_sources = [_bind_dispatch_source(source) for source in dispatch_sources]
    dispatch_sources_source = "\n\n\n".join(bound_dispatch_sources)
    dispatch_body = _render_template(
        "dispatch.py.j2",
        shader_imports=(),
        tensor_imports=(),
        dispatch_sources_source=dispatch_sources_source,
    )
    tensor_imports = tuple(
        {"file": target_file, "classes_source": classes}
        for target_file in sorted(tensor_file_classes)
        for classes in (
            ", ".join(
                cls for cls in sorted(tensor_file_classes[target_file])
                if re.search(rf"\b{re.escape(cls)}\b", dispatch_body)
            ),
        )
        if classes
    )
    return _render_template(
        "dispatch.py.j2",
        shader_imports=tuple(
            {"shader": shader_name, "const": all_shader_imports[shader_name]}
            for shader_name in sorted(all_shader_imports)
        ),
        tensor_imports=tensor_imports,
        dispatch_sources_source=dispatch_sources_source,
    )


def _bind_dispatch_source(source: str) -> str:
    bound = re.sub(
        r"def (run_\w+)\(rt: RuntimeSession, tensors: (\w+)\) -> None:",
        r"def _\1_with_tensors(rt: RuntimeSession, tensors: \2) -> None:",
        source,
        count=1,
    )
    if bound == source:
        raise ValueError("generated dispatch source does not match tensor-bound signature")
    return bound


def _render_shader_init(shader_imports: list[str]) -> str:
    imports_source = "\n".join(shader_imports)
    return f'''"""Generated shader index."""

from __future__ import annotations

import sys

from torch2vk.runtime.shader import ShaderVariant, collect_shader_variants

{imports_source}

_MODEL_SHADERS: dict[str, ShaderVariant] | None = None


def model_shaders() -> dict[str, ShaderVariant]:
    global _MODEL_SHADERS
    if _MODEL_SHADERS is None:
        _MODEL_SHADERS = collect_shader_variants(sys.modules[__name__])
    return _MODEL_SHADERS
'''


# ==============================================================
# Main
# ==============================================================

def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    shaders_dir.mkdir(exist_ok=True)
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir(exist_ok=True)
    reference_programs_dir = output_dir / "reference_programs"
    reference_programs_dir.mkdir(exist_ok=True)
    for f in reference_programs_dir.glob("*.pt2"):
        f.unlink()

    print("Loading model and computing shapes...")
    model, config, shapes = _load_model_and_shapes()

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
    all_shader_imports: dict[str, str] = {
        variant.name: variant.name.upper() for variant in custom_variants
    }
    all_shader_variants: dict[str, ShaderVariant] = {
        variant.name: variant for variant in custom_variants
    }
    tensor_sources: dict[str, list[str]] = {}
    tensor_file_classes: dict[str, list[str]] = {}
    dispatch_sources: list[str] = []
    reference_specs = {}
    reference_specs["input_embed"] = ReferenceSpec(
        program=None,
        input_bindings={
            "input_ids": "batch_input_ids",
            "audio_mask": "batch_audio_mask",
        },
        output_bindings={"hidden_states": "llm_forward.hidden_states"},
    )
    reference_specs["token_score"] = ReferenceSpec(
        program=None,
        input_bindings={
            "logits": "audio_head.linear",
            "tokens": "tokens",
            "audio_mask_id": "audio_mask_id",
            "rng_seed": "rng_seed",
            "step_index": "step_index",
        },
        output_bindings={
            "candidate_tokens": "candidate_tokens",
            "candidate_scores": "candidate_scores",
        },
    )
    reference_specs["token_update"] = ReferenceSpec(
        program=None,
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
    )

    def export_one(
        name,
        module,
        args,
        kwargs=None,
        weight_prefix="",
        layer_loop=None,
        save_reference_program=False,
        reference_program=None,
        reference_input_bindings=None,
        reference_output_bindings=None,
    ):
        prog = export_submodule(module, args=args, kwargs=kwargs)
        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        group = func_name
        program = f"reference_programs/{func_name}.pt2" if save_reference_program else None
        reference_prog = reference_program if reference_program is not None else prog
        if program is not None:
            torch.export.save(reference_prog, output_dir / program)
        reference_specs[func_name] = generate_reference_spec(
            reference_prog,
            program=program,
            input_bindings=reference_input_bindings,
            output_bindings=reference_output_bindings,
        )

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
            tensor_sources.setdefault(group, []).append(layer_src)
            tensor_sources.setdefault(group, []).append(parent_src)
            tensor_file_classes.setdefault(group, []).append(layer_cls_name)
            tensor_file_classes.setdefault(group, []).append(cls_name)

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
            tensor_sources.setdefault(group, []).append(tensor_src)
            tensor_file_classes.setdefault(group, []).append(cls_name)

            func_src, shader_imports, used_variants = generate_dispatch_function_source(
                prog,
                class_name=cls_name,
                function_name=name,
                shader_package="models.exported_omnivoice.shaders",
            )

        rename_map: dict[str, str] = {}
        for v in used_variants.values():
            if v.name in all_shader_variants:
                if all_shader_variants[v.name].contract == v.contract:
                    continue
                new_name = f"{func_name}_{v.name}"
                rename_map[v.name] = new_name
                new_contract = ShaderContract(
                    class_name=v.contract.class_name,
                    shader_name=new_name,
                    fields=v.contract.fields,
                    dispatch=v.contract.dispatch,
                    push_constants=v.contract.push_constants,
                    params_buffer=v.contract.params_buffer,
                )
                renamed = ShaderVariant(
                    name=new_name, family=v.family, contract=new_contract,
                    source=v.source, precompiled_spv_path=v.precompiled_spv_path,
                    specialization_constants=v.specialization_constants,
                    include_dirs=v.include_dirs, compile_defines=v.compile_defines,
                    execution_requirements=v.execution_requirements,
                )
                all_shader_variants[new_name] = renamed
            else:
                all_shader_variants[v.name] = v

        for old_name in sorted(rename_map, key=len, reverse=True):
            new_name = rename_map[old_name]
            old_const = old_name.upper()
            new_const = new_name.upper()
            func_src = re.sub(rf"\b{re.escape(old_const)}\b", new_const, func_src)
            if old_name in shader_imports:
                shader_imports[new_name] = new_const
                del shader_imports[old_name]

        dispatch_sources.append(func_src)
        all_shader_imports.update(shader_imports)

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
            "attention_mask": "attention_mask",
        },
        reference_output_bindings={"mul_365": "llm_forward.mul_365"},
    )

    _sdpa_attn_mod.use_gqa_in_sdpa = _orig_use_gqa

    # --- Audio head ---
    with torch.device("meta"):
        reference_model = cast(OmniVoice, OmniVoice(config).float())  # pyright: ignore[reportCallIssue]
    audio_head_reference_program = export_submodule(
        reference_model.audio_heads.float(),
        args=(torch.zeros(B, S, H, device="meta"),),
    )
    export_one(
        "run_audio_head",
        model.audio_heads.float(),
        args=(torch.zeros(B, S, H, device="cuda"),),
        weight_prefix="audio_heads.",
        save_reference_program=True,
        reference_program=audio_head_reference_program,
        reference_input_bindings={"input": "audio_head.input"},
        reference_output_bindings={"linear": "audio_head.linear"},
    )

    # Write shaders/
    for f in shaders_dir.glob("*.py"):
        f.unlink()
    for shader_name, variant in all_shader_variants.items():
        (shaders_dir / f"{shader_name}.py").write_text(render_shader_file(variant))
    shader_init_imports = [
        f"from models.exported_omnivoice.shaders.{name} import {name.upper()}  # noqa: F401"
        for name in sorted(all_shader_variants)
    ]
    (shaders_dir / "__init__.py").write_text(_render_shader_init(shader_init_imports))
    print(f"\n  {len(all_shader_variants)} shader files written")

    # Write tensors/
    for f in tensors_dir.glob("*.py"):
        f.unlink()
    helper_source = render_tensor_helpers()
    for group, sources in tensor_sources.items():
        (tensors_dir / f"{group}.py").write_text(render_tensor_module(sources, helper_source))
    tensor_init_imports = []
    for group in sorted(tensor_file_classes):
        for cls in tensor_file_classes[group]:
            tensor_init_imports.append(
                f"from models.exported_omnivoice.tensors.{group} import {cls}  # noqa: F401"
            )
    model_source = _render_template(
        "model.py.j2",
        seq_len=shapes["seq_len"],
        batch_size=shapes["batch_size"],
        hidden_size=shapes["hidden_size"],
        head_dim=shapes["head_dim"],
        text_vocab_size=shapes["text_vocab_size"],
        num_audio_codebook=shapes["num_audio_codebook"],
        audio_vocab_size=shapes["audio_vocab_size"],
    )
    (tensors_dir / "model.py").write_text(model_source)
    tensor_init_imports.append(
        "from models.exported_omnivoice.tensors.model import create_model_tensors  # noqa: F401"
    )
    tensor_init_imports.append(
        "from models.exported_omnivoice.tensors.model import model_tensors  # noqa: F401"
    )
    (tensors_dir / "__init__.py").write_text(render_simple_init("Generated tensor declarations", tensor_init_imports))
    print(f"  tensors/ written ({len(tensor_sources)} files)")

    # Write dispatch.py
    dispatch_source = _combine_dispatch(dispatch_sources, all_shader_imports, tensor_file_classes)
    (output_dir / "dispatch.py").write_text(dispatch_source)
    print(f"  dispatch.py written ({len(dispatch_sources)} functions)")

    (output_dir / "reference_specs.py").write_text(render_reference_specs_module(reference_specs))
    print(f"  reference_specs.py written ({len(reference_specs)} specs)")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
