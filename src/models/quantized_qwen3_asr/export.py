"""Export quantized_qwen3_asr submodules → shaders/, tensors/, dispatch/.

Generates Python source files for the full ASR pipeline (audio tower + text).
Shapes are computed from the test fixture (tests/fixtures/qwen3_asr_asknot.wav).

Run from project root:
    uv run python -m models.quantized_qwen3_asr.export
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.quantized_qwen3_asr.pytorch_modules import AudioEncoderModule, AudioInjectModule
from models.quantized_qwen3_asr.quantization import (
    Q8_TENSOR_NAMES,
    Q8_TENSOR_PREFIXES,
    qwen3_asr_q4_k_m_q6_tensor_names,
)
from torch2vk.export import (
    KVCacheInjectHint,
    LayerLoopHint,
    Q4KMWeightQuantization,
    Q4_K_M_REGISTRY,
    Q8_0_REGISTRY,
    ReferencePolicy,
    cast_floating_tensors,
    export_submodule,
    module_floating_dtype,
)
from torch2vk.export.graph import inject_kv_cache
from torch2vk.export.shaders.qwen3_asr_token_select_f32 import (
    QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
)
from torch2vk.export.shaders.qwen3_asr_token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
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
from torch2vk.export.registry import DEFAULT_REGISTRY
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


MODEL_PACKAGE = "models.quantized_qwen3_asr"
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
        "num_hidden_layers": tc.num_hidden_layers,
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
    ac = config.thinker_config.audio_config
    at = model.thinker.audio_tower
    quantized_weights = Q4KMWeightQuantization(
        q6_tensor_names=frozenset(qwen3_asr_q4_k_m_q6_tensor_names(shapes["num_hidden_layers"])),
        q8_tensor_names=frozenset(Q8_TENSOR_NAMES),
        q8_tensor_prefixes=Q8_TENSOR_PREFIXES,
    )

    custom_shader_variants = (
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
        QWEN3_ASR_TOKEN_STORE_EOS_F32,
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
    loader_fields: list[str] = []
    loader_sources: list[str] = []
    reference_functions.append(render_reference_function(
        name="token_select",
        reference_source="reference",
        tensors="model_tensors()",
        frame_name="{name}",
        policy="token",
        input_bindings={
            "logits": "decode_lm_head.linear",
            "eos_token_ids": "eos_token_ids",
        },
        output_bindings={"next_token": "next_token", "done": "done"},
        needs_reference=True,
    ))
    reference_functions.append(render_reference_function(
        name="token_store",
        reference_source="reference",
        tensors="model_tensors()",
        frame_name="{name}",
        policy="token",
        input_bindings={
            "next_token": "next_token",
            "token_index": "token_index",
            "done": "done",
        },
        output_bindings={
            "generated_tokens": "generated_tokens",
            "generated_length": "generated_length",
            "stopped": "stopped",
        },
        needs_reference=True,
    ))

    def export_one(
        name,
        module,
        args,
        kwargs=None,
        weight_prefix="",
        kv_cache=None,
        kv_inject=None,
        layer_loop=None,
        reference_input_bindings=None,
        reference_output_bindings=None,
        reference_tensors=None,
        reference_name=None,
        reference_policy: ReferencePolicy = "tensor",
        reference_module=None,
        export_registry=DEFAULT_REGISTRY,
        weight_quantization: Q4KMWeightQuantization | None = None,
        shape_exprs: dict[int, str] | None = None,
    ):
        module = module.float()
        export_dtype = module_floating_dtype(module)
        if export_dtype is not None:
            args = cast_floating_tensors(args, export_dtype)
            kwargs = cast_floating_tensors(kwargs, export_dtype)
        prog = export_submodule(module, args=args, kwargs=kwargs, kv_cache=kv_cache)
        if kv_inject is not None:
            inject_kv_cache(prog, kv_inject)
        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        tensor_file = _tensor_file_name(cls_name)
        registry = export_registry
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
            tensors=reference_tensors if reference_tensors is not None else f"model_tensors().{func_name}",
            frame_name=reference_name if reference_name is not None else func_name,
            policy=reference_policy,
            input_bindings=reference_input_bindings,
            output_bindings=reference_output_bindings,
        ))

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
                registry=registry,
                weight_quantization=weight_quantization,
                shape_exprs=shape_exprs,
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
                registry=registry,
            )
        else:
            # Flat export: single tensor class + dispatch
            tensor_src = generate_tensor_class_source(
                prog,
                class_name=cls_name,
                function_name=f"create_{func_name}",
                weight_prefix=weight_prefix,
                registry=registry,
                weight_quantization=weight_quantization,
                shape_exprs=shape_exprs,
            )
            (tensors_dir / f"{tensor_file}.py").write_text(render_tensor_module([tensor_src]))

            func_src, shader_imports, used_variants = generate_dispatch_function_source(
                prog,
                class_name=cls_name,
                function_name=name,
                shader_package=f"{MODEL_PACKAGE}.shaders",
                registry=registry,
                weight_quantization=weight_quantization,
            )

        # Handle cross-submodule shader name conflicts
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

        # Apply renames to dispatch source (use word boundary to avoid substring matches)
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
                uses_q4_q6_dispatch="_linear_q4_or_q6" in func_src,
            )
        )

        print(f"  {name}: {len(used_variants)} shaders")

    # Audio encoder wrapper (conv + layers + proj as one module)
    nc = shapes["num_chunks"]
    enc_seq = shapes["enc_seq_len"]
    audio_shape_exprs = {
        nc: "audio_chunk_count",
        nc * 13: "audio_chunk_count * 13",
        enc_seq: "audio_sequence_length",
    }

    # Audio encoder export (single export with layer loop hint)
    num_encoder_layers = len(at.layers)
    export_one("run_audio_encoder", AudioEncoderModule(at).float(),
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
               reference_output_bindings={"linear_110": "linear_110"},
               reference_tensors="model_tensors().audio_encoder",
               reference_name="spike.audio.encoder",
               reference_policy="q4_tensor",
               export_registry=Q8_0_REGISTRY,
               weight_quantization=quantized_weights,
               shape_exprs=audio_shape_exprs)

    # Text pipeline exports
    pl = shapes["prompt_length"]
    max_seq = shapes["max_sequence_length"]
    hs = shapes["hidden_size"]
    hd = shapes["head_dim"]
    text_shape_exprs = {pl: "sequence_length"}
    text_layer_shape_exprs = {pl: "sequence_length", max_seq: "max_sequence_length"}
    decode_layer_shape_exprs = {max_seq: "max_sequence_length"}
    export_one("run_embed_tokens", model.thinker.model.embed_tokens.float(),
               args=(torch.zeros((1, pl), dtype=torch.long, device="meta"),),
               weight_prefix="thinker.model.embed_tokens.",
               reference_module="thinker.model.embed_tokens",
               reference_tensors="model_tensors().embed_tokens",
               reference_name="spike.text.embed",
               reference_policy="q8_tensor",
               export_registry=Q8_0_REGISTRY,
               weight_quantization=quantized_weights,
               shape_exprs=text_shape_exprs)
    export_one("run_audio_inject", AudioInjectModule(),
               args=(torch.zeros(1, pl, hs, device="meta"),
                     torch.zeros(enc_seq, dtype=torch.long, device="meta"),
                     torch.zeros(enc_seq, hs, device="meta")),
               reference_input_bindings={
                   "inputs_embeds": "index_copy",
                   "audio_positions": "audio_positions",
                   "audio_features": "audio_features",
               },
               reference_output_bindings={"embedding": "index_copy"},
               reference_tensors="model_tensors().audio_inject",
               reference_name="spike.text.audio_inject",
               shape_exprs={pl: "sequence_length", enc_seq: "audio_sequence_length"})
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
               reference_output_bindings={"add_7": "add_7"},
               reference_tensors="model_tensors().text_layers[layer_idx]",
               reference_name="spike.text.layer.{layer_idx}",
               reference_policy="q4_tensor",
               export_registry=Q4_K_M_REGISTRY,
               weight_quantization=quantized_weights,
               shape_exprs=text_layer_shape_exprs)
    export_one("run_text_norm", model.thinker.model.norm.float(),
               args=(torch.zeros(1, pl, hs, device="meta"),),
               weight_prefix="thinker.model.norm.",
               reference_module="thinker.model.norm",
               reference_tensors="model_tensors().text_norm",
               reference_name="spike.text.norm",
               shape_exprs=text_shape_exprs)
    export_one("run_lm_head", model.thinker.lm_head.float(),
               args=(torch.zeros(1, pl, hs, device="meta"),),
               weight_prefix="thinker.lm_head.",
               reference_module="thinker.lm_head",
               reference_tensors="model_tensors().lm_head",
               reference_name="spike.text.lm_head",
               reference_policy="q4_tensor",
               export_registry=Q4_K_M_REGISTRY,
               weight_quantization=quantized_weights,
               shape_exprs=text_shape_exprs)

    # Decode-step exports (seq_len=1)
    export_one("run_decode_embed", model.thinker.model.embed_tokens.float(),
               args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
               weight_prefix="thinker.model.embed_tokens.",
               reference_module="thinker.model.embed_tokens",
               reference_tensors="model_tensors().decode_embed",
               reference_name="spike.decode.{step:04d}.embed",
               reference_policy="q8_tensor",
               export_registry=Q8_0_REGISTRY,
               weight_quantization=quantized_weights)
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
               reference_output_bindings={"add_7": "add_7"},
               reference_tensors="model_tensors().decode_layers[layer_idx]",
               reference_name="spike.decode.{step:04d}.layer.{layer_idx}",
               reference_policy="q4_tensor",
               export_registry=Q4_K_M_REGISTRY,
               weight_quantization=quantized_weights,
               shape_exprs=decode_layer_shape_exprs)
    export_one("run_decode_norm", model.thinker.model.norm.float(),
               args=(torch.zeros(1, 1, hs, device="meta"),),
               weight_prefix="thinker.model.norm.",
               reference_module="thinker.model.norm",
               reference_tensors="model_tensors().decode_norm",
               reference_name="spike.decode.{step:04d}.norm")
    export_one("run_decode_lm_head", model.thinker.lm_head.float(),
               args=(torch.zeros(1, 1, hs, device="meta"),),
               weight_prefix="thinker.lm_head.",
               reference_module="thinker.lm_head",
               reference_tensors="model_tensors().decode_lm_head",
               reference_name="spike.decode.{step:04d}.lm_head",
               reference_policy="q4_tensor",
               export_registry=Q4_K_M_REGISTRY,
               weight_quantization=quantized_weights)

    write_shader_init(shaders_dir)
    print(f"\n  {shader_file_count} shader files written")

    # Write model-level tensor wiring.
    (tensors_dir / "rope.py").write_text(_render_template("rope.py.j2"))
    (tensors_dir / "model.py").write_text(
        _render_template("model.py.j2", model_package=MODEL_PACKAGE)
    )
    (tensors_dir / "__init__.py").write_text('"""Generated tensor package."""\n')
    tensor_file_count = len([path for path in tensors_dir.glob("*.py") if path.name != "__init__.py"])
    print(f"  tensors/ written ({tensor_file_count} files)")

    (dispatch_dir / "__init__.py").write_text('"""Generated dispatch package."""\n')
    dispatch_file_count = len([path for path in dispatch_dir.glob("*.py") if path.name != "__init__.py"])
    print(f"  dispatch/ written ({dispatch_file_count} files)")

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package=MODEL_PACKAGE,
            model_imports=[
                "from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (",
                "    Qwen3ASRForConditionalGeneration,",
                ")",
            ],
            model_type="Qwen3ASRForConditionalGeneration",
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
