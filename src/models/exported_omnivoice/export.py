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

import torch
import transformers.integrations.sdpa_attention as _sdpa_attn_mod
from jinja2 import Environment, StrictUndefined

from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.export import (
    LayerLoopHint,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
)
from torch2vk.export.codegen import (
    render_shader_file,
    render_simple_init,
    render_tensor_helpers,
    render_tensor_module,
)
from torch2vk.export.codegen_loop import (
    generate_looped_dispatch_function_source,
    generate_looped_tensor_class_sources,
)
from torch2vk.runtime.shader import ShaderContract, ShaderVariant


_JINJA = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _to_class_name(plan_name: str) -> str:
    base = plan_name.removeprefix("run_")
    return "".join(p.capitalize() for p in base.split("_")) + "Tensors"


# ==============================================================
# Model loading + shape computation
# ==============================================================

def _load_model_and_shapes():
    model_dir = resolve_cached_model(REPO_ID)
    config_data = json.loads((model_dir / "config.json").read_text())
    config = OmniVoiceConfig(**config_data)
    model = OmniVoice(config).float().cuda()

    llm_config = config.llm_config
    seq_len = 300

    shapes = {
        "batch_size": 2,
        "seq_len": seq_len,
        "hidden_size": llm_config.hidden_size,
        "head_dim": llm_config.head_dim,
        "num_attention_heads": llm_config.num_attention_heads,
        "num_key_value_heads": llm_config.num_key_value_heads,
        "num_hidden_layers": llm_config.num_hidden_layers,
        "num_audio_codebook": config.num_audio_codebook,
        "audio_vocab_size": config.audio_vocab_size,
    }
    return model, config, shapes


# ==============================================================
# Output file assembly
# ==============================================================

_DISPATCH_FILE_TEMPLATE = '''"""Generated dispatch functions for OmniVoice submodules."""

from __future__ import annotations

import sys
from typing import cast

{% for item in shader_imports %}
from models.exported_omnivoice.shaders.{{ item.shader }} import {{ item.const }}
{% endfor %}
{% for item in tensor_imports %}
from models.exported_omnivoice.tensors.{{ item.file }} import {{ item.classes_source }}
{% endfor %}
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession


def shader_variant(shader_name: str) -> ShaderVariant:
    return cast(ShaderVariant, getattr(sys.modules[__name__], shader_name.upper()))


{{ dispatch_sources_source }}


def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:
    rt._materialize_read(src)
    with dst.runtime_write_scope():
        dst.buffer = src.buffer
        dst.descriptor_nbytes = src.descriptor_nbytes
        dst.version = src.version
        dst.writer = src.writer
    frame = rt._current_frame()
    frame.used_tensors.append(src)
    frame.written_tensors.append(dst)
'''


def _combine_dispatch(
    dispatch_sources: list[str],
    all_shader_imports: dict[str, str],
    tensor_file_classes: dict[str, list[str]],
) -> str:
    dispatch_body = "\n\n\n".join(dispatch_sources)
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
    return _JINJA.from_string(_DISPATCH_FILE_TEMPLATE).render(
        shader_imports=tuple(
            {"shader": shader_name, "const": all_shader_imports[shader_name]}
            for shader_name in sorted(all_shader_imports)
        ),
        tensor_imports=tensor_imports,
        dispatch_sources_source=dispatch_body,
    )


# ==============================================================
# Main
# ==============================================================

def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    shaders_dir.mkdir(exist_ok=True)
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir(exist_ok=True)

    print("Loading model and computing shapes...")
    model, config, shapes = _load_model_and_shapes()

    B = shapes["batch_size"]
    S = shapes["seq_len"]
    H = shapes["hidden_size"]
    D = shapes["head_dim"]
    num_layers = shapes["num_hidden_layers"]

    all_shader_imports: dict[str, str] = {}
    all_shader_variants: dict[str, ShaderVariant] = {}
    tensor_sources: dict[str, list[str]] = {}
    tensor_file_classes: dict[str, list[str]] = {}
    dispatch_sources: list[str] = []

    def export_one(name, module, args, kwargs=None, weight_prefix="", layer_loop=None):
        prog = export_submodule(module, args=args, kwargs=kwargs)
        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        group = func_name

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

    # --- Text embedding ---
    export_one(
        "run_text_embed",
        model.llm.get_input_embeddings().float(),
        args=(torch.zeros(B, S, dtype=torch.long, device="cuda"),),
        weight_prefix="llm.embed_tokens.",
    )

    # --- Audio embedding (per codebook, summed) ---
    # Host-side prepares shifted_ids for each codebook separately.
    # We export audio_embeddings with flat input (B, S) and call 8 times.
    export_one(
        "run_audio_embed",
        model.audio_embeddings.float(),
        args=(torch.zeros(B, S, dtype=torch.long, device="cuda"),),
        weight_prefix="audio_embeddings.",
    )

    # --- LLM forward (layers + norm) ---
    # Patch use_gqa_in_sdpa to return True so repeat_kv is skipped and K/V pass
    # directly to SDPA with NK=8. Our SDPA shader handles GQA natively.
    _orig_use_gqa = _sdpa_attn_mod.use_gqa_in_sdpa
    _sdpa_attn_mod.use_gqa_in_sdpa = lambda *a, **kw: True

    class _LLMForward(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.layers = m.llm.layers
            self.norm = m.llm.norm

        def forward(self, hidden_states, cos, sin, attention_mask):
            position_embeddings = (cos, sin)
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )
            return self.norm(hidden_states)

    export_one(
        "run_llm_forward",
        _LLMForward(model).float(),
        args=(
            torch.zeros(B, S, H, device="cuda"),
            torch.zeros(B, S, D, device="cuda"),
            torch.zeros(B, S, D, device="cuda"),
            torch.zeros(B, 1, S, S, device="cuda"),
        ),
        weight_prefix="llm.",
        layer_loop=LayerLoopHint(layer_prefix="layers", num_layers=num_layers),
    )

    _sdpa_attn_mod.use_gqa_in_sdpa = _orig_use_gqa

    # --- Audio head ---
    export_one(
        "run_audio_head",
        model.audio_heads.float(),
        args=(torch.zeros(B, S, H, device="cuda"),),
        weight_prefix="audio_heads.",
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
    (shaders_dir / "__init__.py").write_text(render_simple_init("Generated shader index", shader_init_imports))
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
    (tensors_dir / "__init__.py").write_text(render_simple_init("Generated tensor declarations", tensor_init_imports))
    print(f"  tensors/ written ({len(tensor_sources)} files)")

    # Write dispatch.py
    dispatch_source = _combine_dispatch(dispatch_sources, all_shader_imports, tensor_file_classes)
    (output_dir / "dispatch.py").write_text(dispatch_source)
    print(f"  dispatch.py written ({len(dispatch_sources)} functions)")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
