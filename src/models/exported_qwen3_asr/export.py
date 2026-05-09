"""Export all exported_qwen3_asr submodules → shaders/, tensors/, dispatch.py.

Generates Python source files for the full ASR pipeline (audio tower + text).
Shapes are computed from the test fixture (tests/fixtures/qwen3_asr_asknot.wav).

Run from project root:
    uv run python -m models.exported_qwen3_asr.export
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from models.exported_qwen3_asr.debug_audio_tower import DebugAudioTower
from models.exported_qwen3_asr.export_forwards import (
    export_audio_inject_forward,
    patched_forward,
)
from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from torch2vk.export import (
    KVCacheInjectHint,
    LayerLoopHint,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
)
from torch2vk.export.codegen import (
    generate_reference_spec,
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
    keep_trailing_newline=True,
    loader=FileSystemLoader(_TEMPLATE_DIR),
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _render_template(template_name: str, **context) -> str:
    return _JINJA.get_template(template_name).render(**context)


def _to_class_name(plan_name: str) -> str:
    base = plan_name.removeprefix("run_")
    return "".join(p.capitalize() for p in base.split("_")) + "Tensors"




# ==============================================================
# Model loading + shape computation
# ==============================================================

def _load_model_and_shapes():
    model_dir = resolve_cached_model(REPO_ID)
    payload = json.loads((Path(model_dir) / "config.json").read_text())

    with open(os.devnull, "w") as devnull:
        stdout_fd, stderr_fd = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

            config = Qwen3ASRConfig(**payload)
            with torch.device("meta"):
                model = Qwen3ASRForConditionalGeneration(config)
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)

    _, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav="tests/fixtures/qwen3_asr_asknot.wav")
    ac = config.thinker_config.audio_config
    tc = config.thinker_config.text_config

    feat_len = prepared.audio_feature_length
    n_window = 50
    chunk_num = int(np.ceil(feat_len / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feat_len % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder
    max_chunk_len = int(chunk_lengths.max())

    def conv_out_size(in_size, kernel, stride, padding):
        return (in_size + 2 * padding - kernel) // stride + 1

    h, w = ac.num_mel_bins, max_chunk_len
    h1, w1 = conv_out_size(h, 3, 2, 1), conv_out_size(w, 3, 2, 1)
    h2, w2 = conv_out_size(h1, 3, 2, 1), conv_out_size(w1, 3, 2, 1)
    h3, w3 = conv_out_size(h2, 3, 2, 1), conv_out_size(w2, 3, 2, 1)

    def get_feat_extract_output_lengths(input_lengths):
        leave = input_lengths % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    feature_lens_after_cnn = get_feat_extract_output_lengths(chunk_lengths)
    enc_seq_len = int(feature_lens_after_cnn.sum())

    n_window_infer = 800
    aftercnn_lens = get_feat_extract_output_lengths(np.array([feat_len], dtype=np.int64))
    window_aftercnn = int(feature_lens_after_cnn.max()) * (n_window_infer // (n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        rem = cnn_len % window_aftercnn
        if rem != 0:
            cu_chunk_lens += [rem]
    cu_seqlens_len = len(np.cumsum(cu_chunk_lens, dtype=np.int32))

    shapes = {
        "num_chunks": chunk_num,
        "max_chunk_len": max_chunk_len,
        "conv2d1_out": (chunk_num, 480, h1, w1),
        "conv2d2_out": (chunk_num, 480, h2, w2),
        "conv2d3_out": (chunk_num, 480, h3, w3),
        "conv_out": (chunk_num, w3, ac.d_model),
        "enc_seq_len": enc_seq_len,
        "cu_seqlens_len": cu_seqlens_len,
        "d_model": ac.d_model,
        "prompt_length": prepared.prompt_length,
        "max_sequence_length": prepared.prompt_length + 64,
        "hidden_size": tc.hidden_size,
        "head_dim": tc.head_dim,
        "num_attention_heads": tc.num_attention_heads,
        "num_key_value_heads": tc.num_key_value_heads,
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
    ac = config.thinker_config.audio_config
    at = model.thinker.audio_tower

    all_shader_imports: dict[str, str] = {}  # shader_name → CONST_NAME
    all_shader_variants: dict[str, ShaderVariant] = {}
    tensor_sources: dict[str, list[str]] = {}  # file_group → [class source, ...]
    tensor_file_classes: dict[str, list[str]] = {}  # file_group → [class names]
    dispatch_sources: list[str] = []
    reference_specs = {}
    reference_specs["token_select"] = ReferenceSpec(
        program=None,
        input_bindings={},
        output_bindings={"next_token": "next_token", "done": "done"},
    )
    reference_specs["token_store"] = ReferenceSpec(
        program=None,
        input_bindings={},
        output_bindings={
            "generated_tokens": "generated_tokens",
            "generated_length": "generated_length",
            "stopped": "stopped",
        },
    )

    def export_one(
        name,
        module,
        args,
        kwargs=None,
        weight_prefix="",
        kv_cache=None,
        kv_inject=None,
        layer_loop=None,
        save_reference_program=False,
        reference_input_bindings=None,
        reference_output_bindings=None,
    ):
        prog = export_submodule(module, args=args, kwargs=kwargs, kv_cache=kv_cache)
        if kv_inject is not None:
            from torch2vk.export.graph import inject_kv_cache
            inject_kv_cache(prog, kv_inject)
        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        group = func_name
        program = f"reference_programs/{func_name}.pt2" if save_reference_program else None
        if program is not None:
            torch.export.save(prog, output_dir / program)
        reference_specs[func_name] = generate_reference_spec(
            prog,
            program=program,
            input_bindings=reference_input_bindings,
            output_bindings=reference_output_bindings,
        )

        if layer_loop is not None:
            # Looped export: generates parent + layer tensor classes and looped dispatch
            layer_cls_name = "EncoderLayerTensors"
            layer_func_name = "create_encoder_layer"

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
            # Flat export: single tensor class + dispatch
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
                shader_package="models.exported_qwen3_asr.shaders",
            )

        # Handle cross-submodule shader name conflicts
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

        # Apply renames to dispatch source (use word boundary to avoid substring matches)
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

    # Audio encoder wrapper (conv + layers + proj as one module)
    nc = shapes["num_chunks"]
    enc_seq = shapes["enc_seq_len"]

    # Audio encoder export (single export with layer loop hint)
    num_encoder_layers = len(at.layers)
    export_one("run_audio_encoder", DebugAudioTower(at).float(),
               args=(torch.zeros(nc, 1, ac.num_mel_bins, shapes["max_chunk_len"], device="meta"),
                     torch.zeros(*shapes["conv_out"], device="meta"),
                     torch.zeros(enc_seq, dtype=torch.long, device="meta")),
               kwargs={
                   "cu_seqlens": torch.zeros(
                       shapes["cu_seqlens_len"],
                       dtype=torch.int32,
                       device="meta",
                   ),
                   "attention_mask": torch.zeros(1, 1, enc_seq, enc_seq, device="meta"),
               },
               weight_prefix="thinker.",
               layer_loop=LayerLoopHint(
                   layer_prefix="audio_tower.layers",
                   num_layers=num_encoder_layers,
               ),
               reference_input_bindings={
                   "x": "x",
                   "position_embedding": "position_embedding",
                   "compact_index": "compact_index",
                   "attention_mask": "attention_mask",
               },
               reference_output_bindings={"linear_110": "linear_110"})

    # Text pipeline exports
    pl = shapes["prompt_length"]
    max_seq = shapes["max_sequence_length"]
    hs = shapes["hidden_size"]
    hd = shapes["head_dim"]
    export_one("run_embed_tokens", model.thinker.model.embed_tokens.float(),
               args=(torch.zeros((1, pl), dtype=torch.long, device="meta"),),
               weight_prefix="thinker.model.embed_tokens.",
               save_reference_program=True)
    with patched_forward(model.thinker, export_audio_inject_forward):
        export_one("run_audio_inject", model.thinker,
                   args=(torch.zeros(1, pl, hs, device="meta"),
                         torch.zeros(enc_seq, dtype=torch.long, device="meta"),
                         torch.zeros(enc_seq, hs, device="meta")),
                   reference_input_bindings={
                       "audio_positions": "audio_positions",
                       "audio_features": "audio_features",
                   },
                   reference_output_bindings={"embedding": "index_copy"})
    export_one("run_text_layer", model.thinker.model.layers[0],
               args=(torch.zeros(1, pl, hs, device="meta"),
                     (torch.zeros(1, pl, hd, device="meta"),
                      torch.zeros(1, pl, hd, device="meta"))),
               kwargs={"past_key_values": None, "attention_mask": None},
               weight_prefix="thinker.model.layers.0.",
               kv_inject=KVCacheInjectHint(phase="prefill", max_seq_len=max_seq),
               reference_input_bindings={
                   "hidden_states": "hidden_states",
                   "position_embeddings_0": "position_embeddings_0",
                   "position_embeddings_1": "position_embeddings_1",
                   "cache_position": "cache_position",
               },
               reference_output_bindings={"add_7": "add_7"})
    export_one("run_text_norm", model.thinker.model.norm.float(),
               args=(torch.zeros(1, pl, hs, device="meta"),),
               weight_prefix="thinker.model.norm.",
               save_reference_program=True)
    export_one("run_lm_head", model.thinker.lm_head.float(),
               args=(torch.zeros(1, pl, hs, device="meta"),),
               weight_prefix="thinker.lm_head.",
               save_reference_program=True)

    # Decode-step exports (seq_len=1)
    export_one("run_decode_embed", model.thinker.model.embed_tokens.float(),
               args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
               weight_prefix="thinker.model.embed_tokens.",
               save_reference_program=True)
    export_one("run_decode_layer", model.thinker.model.layers[0],
               args=(torch.zeros(1, 1, hs, device="meta"),
                     (torch.zeros(1, 1, hd, device="meta"),
                      torch.zeros(1, 1, hd, device="meta"))),
               kwargs={"past_key_values": None, "attention_mask": None},
               weight_prefix="thinker.model.layers.0.",
               kv_inject=KVCacheInjectHint(phase="decode", max_seq_len=max_seq),
               reference_input_bindings={
                   "hidden_states": "hidden_states",
                   "position_embeddings_0": "position_embeddings_0",
                   "position_embeddings_1": "position_embeddings_1",
                   "cache_position": "cache_position",
               },
               reference_output_bindings={"add_7": "add_7"})
    export_one("run_decode_norm", model.thinker.model.norm.float(),
               args=(torch.zeros(1, 1, hs, device="meta"),),
               weight_prefix="thinker.model.norm.",
               save_reference_program=True)
    export_one("run_decode_lm_head", model.thinker.lm_head.float(),
               args=(torch.zeros(1, 1, hs, device="meta"),),
               weight_prefix="thinker.lm_head.",
               save_reference_program=True)

    # Write shaders/
    for f in shaders_dir.glob("*.py"):
        f.unlink()
    for shader_name, variant in all_shader_variants.items():
        (shaders_dir / f"{shader_name}.py").write_text(render_shader_file(variant))
    shader_init_imports = [
        f"from models.exported_qwen3_asr.shaders.{name} import {name.upper()}  # noqa: F401"
        for name in sorted(all_shader_variants)
    ]
    shader_init_imports.extend(
        [
            "from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32  # noqa: F401",
            "from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32  # noqa: F401",
        ]
    )
    (shaders_dir / "__init__.py").write_text(_render_shader_init(shader_init_imports))
    print(f"\n  {len(all_shader_variants)} shader files written")

    # Write tensors/
    for f in tensors_dir.glob("*.py"):
        f.unlink()
    tensor_file_classes.setdefault("rope", []).append("RopeTableTensors")
    helper_source = render_tensor_helpers()
    for group, sources in tensor_sources.items():
        (tensors_dir / f"{group}.py").write_text(render_tensor_module(sources, helper_source))
    (tensors_dir / "rope.py").write_text(_render_template("rope.py.j2"))
    (tensors_dir / "model.py").write_text(_render_template("model.py.j2"))
    tensor_init_imports = []
    for group in sorted(tensor_file_classes):
        for cls in tensor_file_classes[group]:
            tensor_init_imports.append(
                f"from models.exported_qwen3_asr.tensors.{group} import {cls}  # noqa: F401"
            )
    tensor_init_imports.append(
        "from models.exported_qwen3_asr.tensors.model import ExportedQwen3AsrTensors  # noqa: F401"
    )
    tensor_init_imports.append(
        "from models.exported_qwen3_asr.tensors.model import create_model_tensors  # noqa: F401"
    )
    tensor_init_imports.append(
        "from models.exported_qwen3_asr.tensors.model import model_tensors  # noqa: F401"
    )
    (tensors_dir / "__init__.py").write_text(render_simple_init("Generated tensor declarations", tensor_init_imports))
    print(f"  tensors/ written ({len(tensor_sources) + 2} files)")

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
