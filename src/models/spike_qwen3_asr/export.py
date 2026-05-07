"""Export all spike_qwen3_asr submodules → shaders/, tensors/, dispatch.py.

Generates Python source files for the full ASR pipeline (audio tower + text).
Shapes are computed from the test fixture (tests/fixtures/qwen3_asr_asknot.wav).

Run from project root:
    .venv/bin/python -m models.spike_qwen3_asr.export
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.export import export_submodule
from torch2vk.export.graph import SKIP_OPS, is_alias_op, node_input_names
from torch2vk.export.registry import DEFAULT_REGISTRY
from torch2vk.runtime.shader import (
    AddExpr,
    CeilDivExpr,
    IOKind,
    MulExpr,
    ShaderContract,
    ShaderVariant,
)


# ==============================================================
# Plan extraction from ExportedProgram
# ==============================================================

def _rename_variant(variant: ShaderVariant, new_name: str) -> ShaderVariant:
    new_contract = ShaderContract(
        class_name=variant.contract.class_name,
        shader_name=new_name,
        fields=variant.contract.fields,
        dispatch=variant.contract.dispatch,
        push_constants=variant.contract.push_constants,
        params_buffer=variant.contract.params_buffer,
    )
    return ShaderVariant(
        name=new_name,
        family=variant.family,
        contract=new_contract,
        source=variant.source,
    )


def _extract_plan(prog: ExportedProgram, weight_prefix: str) -> dict:
    graph = prog.graph_module.graph
    sig = prog.graph_signature

    tensors = {}
    user_inputs = []
    param_map = {}

    for spec in sig.input_specs:
        for node in graph.nodes:
            if node.name == spec.arg.name:
                tm = node.meta.get("tensor_meta")
                if tm:
                    shape = tuple(int(d) for d in tm.shape)
                    dtype = str(tm.dtype).removeprefix("torch.")
                    is_param = spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                    tensors[spec.arg.name] = {
                        "shape": list(shape),
                        "dtype": dtype,
                        "kind": "parameter" if is_param else "user_input",
                    }
                    if is_param:
                        param_map[spec.arg.name] = f"{weight_prefix}{spec.target}"
                    else:
                        user_inputs.append(spec.arg.name)
                break

    for node in graph.nodes:
        if node.op == "call_function" and node.name not in tensors:
            if str(node.target) in SKIP_OPS:
                continue
            tm = node.meta.get("tensor_meta")
            if tm:
                shape = tuple(int(d) for d in tm.shape)
                dtype = str(tm.dtype).removeprefix("torch.")
                tensors[node.name] = {"shape": list(shape), "dtype": dtype, "kind": "intermediate"}

    ops = []
    shader_variants = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in SKIP_OPS:
            continue
        # Skip non-float32 compute (e.g. int32 cu_seqlens indexing)
        tm = node.meta.get("tensor_meta")
        if tm and str(tm.dtype) != "torch.float32":
            continue
        if is_alias_op(node):
            inputs = node_input_names(node)
            src = inputs[0] if inputs else None
            if src and src in tensors and node.name in tensors:
                ops.append({"type": "alias", "src": src, "dst": node.name})
            continue
        variant = DEFAULT_REGISTRY.resolve(node)
        if variant is None:
            ops.append({"type": "unsupported", "target": target, "name": node.name})
            continue
        shader_key = variant.name
        if shader_key in shader_variants and shader_variants[shader_key].contract != variant.contract:
            # Same op name but different shape contract within one submodule
            suffix = f"_{len(shader_variants)}"
            shader_key = f"{variant.name}{suffix}"
            variant = _rename_variant(variant, shader_key)
        if shader_key not in shader_variants:
            shader_variants[shader_key] = variant
        inputs = node_input_names(node)
        input_fields = [f for f in variant.contract.fields if f.io_kind == IOKind.INPUT]
        output_fields = [f for f in variant.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)]
        bindings = {}
        for i, field in enumerate(input_fields):
            if i < len(inputs) and inputs[i] in tensors:
                bindings[field.name] = inputs[i]
        for field in output_fields:
            bindings[field.name] = node.name
        ops.append({"type": "dispatch", "shader": shader_key, "output_tensor": node.name, "bindings": bindings})

    output_name = None
    for op in reversed(ops):
        if op["type"] == "dispatch":
            output_name = op["output_tensor"]
            break

    return {
        "tensors": tensors,
        "user_inputs": user_inputs,
        "param_map": param_map,
        "output": output_name,
        "ops": ops,
        "shader_variants": shader_variants,
    }


# ==============================================================
# Source code generation
# ==============================================================

def _expr_to_source(expr) -> str:
    if isinstance(expr, int):
        return repr(expr)
    if isinstance(expr, str):
        return repr(expr)
    if isinstance(expr, CeilDivExpr):
        return f"ceil_div({_expr_to_source(expr.lhs)}, {_expr_to_source(expr.rhs)})"
    if isinstance(expr, MulExpr):
        return f"mul({_expr_to_source(expr.lhs)}, {_expr_to_source(expr.rhs)})"
    if isinstance(expr, AddExpr):
        return f"add({_expr_to_source(expr.lhs)}, {_expr_to_source(expr.rhs)})"
    raise TypeError(f"Unknown expr type: {type(expr)}")


def _shape_to_source(shape: tuple) -> str:
    return f"({', '.join(_expr_to_source(d) for d in shape)},)"


def _generate_shader_file(variant: ShaderVariant) -> str:
    contract = variant.contract
    const_name = variant.name.upper()

    needed = {"IOKind", "ShaderContract", "ShaderVariant", "TensorContract", "TensorFieldSpec"}
    if contract.push_constants:
        needed.update({"PushConstantFieldSpec", "PushConstantSpec", "PushConstantType"})

    def _check(expr):
        if isinstance(expr, CeilDivExpr):
            needed.add("ceil_div"); _check(expr.lhs); _check(expr.rhs)
        elif isinstance(expr, MulExpr):
            needed.add("mul"); _check(expr.lhs); _check(expr.rhs)
        elif isinstance(expr, AddExpr):
            needed.add("add"); _check(expr.lhs); _check(expr.rhs)

    for d in contract.dispatch:
        _check(d)
    if contract.push_constants:
        for f in contract.push_constants.fields:
            _check(f.value)

    imports = ["from __future__ import annotations", "", "from torch2vk.runtime.shader import ("]
    for name in sorted(needed):
        imports.append(f"    {name},")
    imports.append(")")

    fields_lines = []
    for f in contract.fields:
        fields_lines.append(f"            TensorFieldSpec(")
        fields_lines.append(f"                name={f.name!r},")
        fields_lines.append(f"                io_kind=IOKind.{f.io_kind.name},")
        fields_lines.append(f"                role={f.role!r},")
        fields_lines.append(f"                contract=TensorContract(dtype={f.contract.dtype!r}, shape={_shape_to_source(f.contract.shape)}),")
        fields_lines.append(f"            ),")

    pc_source = "None"
    if contract.push_constants:
        pc_fields = []
        for pf in contract.push_constants.fields:
            val = repr(pf.value) if isinstance(pf.value, (int, float)) else _expr_to_source(pf.value)
            pc_fields.append(f"                PushConstantFieldSpec({pf.name!r}, PushConstantType.{pf.dtype.name}, {pf.offset}, {val}),")
        pc_source = (
            f"PushConstantSpec(\n"
            f"            size={contract.push_constants.size},\n"
            f"            fields=(\n" + "\n".join(pc_fields) + "\n"
            f"            ),\n"
            f"        )"
        )

    dispatch_source = f"({', '.join(_expr_to_source(d) for d in contract.dispatch)})"
    glsl = variant.source.lstrip("\n")

    lines = [f'"""Generated shader: {variant.name}."""', ""]
    lines.extend(imports)
    lines.append("")
    lines.append("")
    lines.append(f"{const_name} = ShaderVariant(")
    lines.append(f"    name={variant.name!r},")
    lines.append(f"    family={variant.family!r},")
    lines.append(f"    contract=ShaderContract(")
    lines.append(f"        class_name={contract.class_name!r},")
    lines.append(f"        shader_name={contract.shader_name!r},")
    lines.append(f"        fields=(")
    lines.extend(fields_lines)
    lines.append(f"        ),")
    lines.append(f"        push_constants={pc_source},")
    lines.append(f"        dispatch={dispatch_source},")
    lines.append(f"    ),")
    lines.append(f'    source="""\\\n{glsl}""",')
    lines.append(f")")
    lines.append("")
    return "\n".join(lines)


def _generate_dispatch_function(plan: dict, function_name: str) -> str:
    class_name = _plan_to_class_name(function_name)
    ops = plan["ops"]
    shader_imports = {}
    for op in ops:
        if op["type"] == "dispatch":
            shader_imports[op["shader"]] = op["shader"].upper()

    lines = []
    lines.append(f"def {function_name}(rt: RuntimeSession, tensors: {class_name}) -> None:")
    for op in ops:
        if op["type"] == "alias":
            lines.append(f"    _alias(rt, tensors.{op['src']}, tensors.{op['dst']})")
        elif op["type"] == "dispatch":
            const = shader_imports[op["shader"]]
            args = ", ".join(f"{k}=tensors.{v}" for k, v in op["bindings"].items())
            lines.append(f"    {const}(rt, {args})")
    lines.append("")
    return "\n".join(lines), shader_imports


def _generate_dispatch_file(plans: dict[str, dict]) -> str:
    all_shader_imports = {}
    function_sources = []

    for func_name, plan in plans.items():
        source, imports = _generate_dispatch_function(plan, func_name)
        function_sources.append(source)
        all_shader_imports.update(imports)

    # Group tensor class imports by file
    tensor_imports: dict[str, list[str]] = {}
    for func_name in plans:
        target_file = PLAN_TO_FILE.get(func_name, "misc")
        class_name = _plan_to_class_name(func_name)
        tensor_imports.setdefault(target_file, []).append(class_name)

    lines = ['"""Generated dispatch functions for all submodules."""', ""]
    lines.append("from __future__ import annotations")
    lines.append("")
    for shader_name in sorted(all_shader_imports):
        const_name = all_shader_imports[shader_name]
        lines.append(f"from models.spike_qwen3_asr.shaders.{shader_name} import {const_name}")
    for target_file in sorted(tensor_imports):
        classes = ", ".join(sorted(tensor_imports[target_file]))
        lines.append(f"from models.spike_qwen3_asr.tensors.{target_file} import {classes}")
    lines.append("from torch2vk.runtime.logical import LogicalTensor")
    lines.append("from torch2vk.runtime.session import RuntimeSession")
    lines.append("")
    lines.append("")
    for source in function_sources:
        lines.append(source)
        lines.append("")
    lines.append("def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:")
    lines.append("    rt._materialize_read(src)")
    lines.append("    with dst.runtime_write_scope():")
    lines.append("        dst.buffer = src.buffer")
    lines.append("        dst.descriptor_nbytes = src.descriptor_nbytes")
    lines.append("        dst.version = src.version")
    lines.append("")
    return "\n".join(lines)


# ==============================================================
# Tensor file generation
# ==============================================================

PLAN_TO_FILE: dict[str, str] = {
    "run_conv2d1": "audio_tower",
    "run_conv2d2": "audio_tower",
    "run_conv2d3": "audio_tower",
    "run_conv_out": "audio_tower",
    "run_ln_post": "audio_tower",
    "run_proj1": "audio_tower",
    "run_proj2": "audio_tower",
    "run_encoder_layer": "encoder_layer",
    "run_embed_tokens": "text",
    "run_text_norm": "text",
    "run_lm_head": "text",
    "run_text_layer": "text_layer",
    "run_decode_embed": "decode",
    "run_decode_norm": "decode",
    "run_decode_lm_head": "decode",
    "run_decode_layer": "decode_layer",
}


def _plan_to_class_name(plan_name: str) -> str:
    base = plan_name.removeprefix("run_")
    parts = base.split("_")
    return "".join(p.capitalize() for p in parts) + "Tensors"


def _make_weight_map_template(param_map: dict[str, str]) -> dict[str, str]:
    result = {}
    for field_name, safetensors_key in param_map.items():
        template = re.sub(r"\.layers\.(\d+)\.", ".layers.{i}.", safetensors_key)
        result[field_name] = template
    return result


def _generate_tensor_class(plan_name: str, plan: dict) -> str:
    class_name = _plan_to_class_name(plan_name)
    func_name = plan_name.removeprefix("run_")
    weight_map = _make_weight_map_template(plan["param_map"])
    output_name = plan["output"]

    # Collect fields in order: parameters, user_inputs, intermediates
    fields_lines = []
    for name, meta in plan["tensors"].items():
        fields_lines.append(f"    {name}: LogicalTensor")

    # WEIGHT_MAP constant
    wm_prefix = func_name.upper()
    wm_lines = []
    for field_name, template in weight_map.items():
        wm_lines.append(f"    {field_name!r}: {template!r},")

    # create() function body
    create_args = []
    for name, meta in plan["tensors"].items():
        shape = tuple(meta["shape"])
        dtype = "int32" if meta["dtype"] in ("int64", "int32") else "float32"
        kind = meta["kind"]

        if kind == "parameter":
            role, memory, lifetime = "TensorRole.INPUT", "MemoryClass.HOST_INPUT", "TensorLifetime.FRAME"
        elif kind == "user_input":
            role, memory, lifetime = "TensorRole.INPUT", "MemoryClass.HOST_INPUT", "TensorLifetime.FRAME"
        else:
            role, memory, lifetime = "TensorRole.ACTIVATION", "MemoryClass.FRAME_WORKSPACE", "TensorLifetime.FRAME"

        extra = ""
        if name == output_name:
            extra = (
                "\n            compare=ComparePolicy(kind=\"tensor\", rtol=3e-3, atol=3e-2),"
                "\n            pytorch_probe=PyTorchProbe(kind=\"module_output\", target=\"\", index=0),"
            )

        create_args.append(
            f"        {name}=LogicalTensor(\n"
            f"            name=f\"{{prefix}}.{name}\",\n"
            f"            spec=TensorSpec(dtype={dtype!r}, shape={shape}),\n"
            f"            role={role}, memory={memory}, lifetime={lifetime},{extra}\n"
            f"        ),"
        )

    lines = []
    lines.append(f"@dataclass(frozen=True, slots=True)")
    lines.append(f"class {class_name}:")
    lines.extend(fields_lines)
    lines.append("")
    lines.append("")
    lines.append(f"{wm_prefix}_WEIGHT_MAP: dict[str, str] = {{")
    lines.extend(wm_lines)
    lines.append("}")
    lines.append("")
    lines.append(f"{wm_prefix}_OUTPUT: str = {output_name!r}")
    lines.append("")
    lines.append("")
    lines.append(f"def create_{func_name}(prefix: str) -> {class_name}:")
    lines.append(f"    return {class_name}(")
    lines.extend(create_args)
    lines.append(f"    )")
    lines.append("")
    return "\n".join(lines)


def _generate_tensors_files(all_plans: dict[str, dict]) -> dict[str, str]:
    """Generate tensors/ file contents grouped by PLAN_TO_FILE mapping."""
    file_contents: dict[str, list[str]] = {}

    for plan_name, plan in all_plans.items():
        target_file = PLAN_TO_FILE.get(plan_name, "misc")
        if target_file not in file_contents:
            file_contents[target_file] = [
                f'"""Generated tensor declarations."""',
                "",
                "from __future__ import annotations",
                "",
                "from dataclasses import dataclass",
                "",
                "from torch2vk.runtime.logical import (",
                "    ComparePolicy,",
                "    LogicalTensor,",
                "    MemoryClass,",
                "    PyTorchProbe,",
                "    TensorLifetime,",
                "    TensorRole,",
                ")",
                "from torch2vk.vulkan.types import TensorSpec",
                "",
                "",
            ]
        file_contents[target_file].append(_generate_tensor_class(plan_name, plan))
        file_contents[target_file].append("")

    result = {}
    for filename, lines in file_contents.items():
        result[filename] = "\n".join(lines)
    return result


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
        "conv_out_in": (chunk_num * w3, 480 * h3),
        "enc_seq_len": enc_seq_len,
        "cu_seqlens_len": cu_seqlens_len,
        "d_model": ac.d_model,
        "prompt_length": prepared.prompt_length,
        "hidden_size": tc.hidden_size,
        "head_dim": tc.head_dim,
    }
    return model, config, shapes


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
    ac = config.thinker_config.audio_config
    tc = config.thinker_config.text_config
    at = model.thinker.audio_tower

    all_shader_variants = {}
    all_plans = {}

    def export_and_plan(name, module, args, kwargs=None, weight_prefix=""):
        prog = export_submodule(module, args=args, kwargs=kwargs)
        plan = _extract_plan(prog, weight_prefix)
        # Merge shader variants globally; rename on conflict to keep them unique
        for shader_key, variant in list(plan["shader_variants"].items()):
            if shader_key in all_shader_variants:
                if all_shader_variants[shader_key].contract == variant.contract:
                    continue
                # Conflict: same name, different contract. Prefix with plan name.
                new_key = f"{name}_{shader_key}".replace("run_", "")
                variant = _rename_variant(variant, new_key)
                for op in plan["ops"]:
                    if op.get("shader") == shader_key:
                        op["shader"] = new_key
                del plan["shader_variants"][shader_key]
                plan["shader_variants"][new_key] = variant
                shader_key = new_key
            all_shader_variants[shader_key] = variant
        all_plans[name] = plan
        n_ops = sum(1 for op in plan["ops"] if op["type"] == "dispatch")
        print(f"  {name}: {n_ops} ops, {len(plan['shader_variants'])} shaders")

    # Audio tower
    nc = shapes["num_chunks"]
    export_and_plan("run_conv2d1", at.conv2d1.float(),
                    args=(torch.zeros(nc, 1, ac.num_mel_bins, shapes["max_chunk_len"], device="meta"),),
                    weight_prefix="thinker.audio_tower.conv2d1.")
    export_and_plan("run_conv2d2", at.conv2d2.float(),
                    args=(torch.zeros(*shapes["conv2d1_out"], device="meta"),),
                    weight_prefix="thinker.audio_tower.conv2d2.")
    export_and_plan("run_conv2d3", at.conv2d3.float(),
                    args=(torch.zeros(*shapes["conv2d2_out"], device="meta"),),
                    weight_prefix="thinker.audio_tower.conv2d3.")
    export_and_plan("run_conv_out", at.conv_out.float(),
                    args=(torch.zeros(*shapes["conv_out_in"], device="meta"),),
                    weight_prefix="thinker.audio_tower.conv_out.")
    enc_seq = shapes["enc_seq_len"]
    export_and_plan("run_encoder_layer", at.layers[0].float(),
                    args=(torch.zeros(enc_seq, shapes["d_model"], device="meta"),
                          torch.zeros(shapes["cu_seqlens_len"], dtype=torch.int32, device="meta")),
                    kwargs={"attention_mask": torch.zeros(1, 1, enc_seq, enc_seq, device="meta")},
                    weight_prefix="thinker.audio_tower.layers.0.")
    export_and_plan("run_ln_post", at.ln_post.float(),
                    args=(torch.zeros(shapes["enc_seq_len"], shapes["d_model"], device="meta"),),
                    weight_prefix="thinker.audio_tower.ln_post.")
    export_and_plan("run_proj1", at.proj1.float(),
                    args=(torch.zeros(shapes["enc_seq_len"], shapes["d_model"], device="meta"),),
                    weight_prefix="thinker.audio_tower.proj1.")
    export_and_plan("run_proj2", at.proj2.float(),
                    args=(torch.zeros(shapes["enc_seq_len"], shapes["d_model"], device="meta"),),
                    weight_prefix="thinker.audio_tower.proj2.")

    # Text pipeline
    pl = shapes["prompt_length"]
    hs = shapes["hidden_size"]
    hd = shapes["head_dim"]
    export_and_plan("run_embed_tokens", model.thinker.model.embed_tokens.float(),
                    args=(torch.zeros((1, pl), dtype=torch.long, device="meta"),),
                    weight_prefix="thinker.model.embed_tokens.")
    export_and_plan("run_text_layer", model.thinker.model.layers[0],
                    args=(torch.zeros(1, pl, hs, device="meta"),),
                    kwargs={"position_embeddings": (
                        torch.zeros(1, pl, hd, device="meta"),
                        torch.zeros(1, pl, hd, device="meta"),
                    )},
                    weight_prefix="thinker.model.layers.0.")
    export_and_plan("run_text_norm", model.thinker.model.norm.float(),
                    args=(torch.zeros(1, pl, hs, device="meta"),),
                    weight_prefix="thinker.model.norm.")
    export_and_plan("run_lm_head", model.thinker.lm_head.float(),
                    args=(torch.zeros(1, 1, hs, device="meta"),),
                    weight_prefix="thinker.lm_head.")

    # Decode-step variants (seq_len=1)
    export_and_plan("run_decode_embed", model.thinker.model.embed_tokens.float(),
                    args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
                    weight_prefix="thinker.model.embed_tokens.")
    export_and_plan("run_decode_layer", model.thinker.model.layers[0],
                    args=(torch.zeros(1, 1, hs, device="meta"),),
                    kwargs={"position_embeddings": (
                        torch.zeros(1, 1, hd, device="meta"),
                        torch.zeros(1, 1, hd, device="meta"),
                    )},
                    weight_prefix="thinker.model.layers.0.")
    export_and_plan("run_decode_norm", model.thinker.model.norm.float(),
                    args=(torch.zeros(1, 1, hs, device="meta"),),
                    weight_prefix="thinker.model.norm.")
    export_and_plan("run_decode_lm_head", model.thinker.lm_head.float(),
                    args=(torch.zeros(1, 1, hs, device="meta"),),
                    weight_prefix="thinker.lm_head.")

    # Write shaders/
    for shader_name, variant in all_shader_variants.items():
        (shaders_dir / f"{shader_name}.py").write_text(_generate_shader_file(variant))
    init_lines = ['"""Generated shader index."""', ""]
    for shader_name in sorted(all_shader_variants):
        init_lines.append(f"from models.spike_qwen3_asr.shaders.{shader_name} import {shader_name.upper()}  # noqa: F401")
    init_lines.append("")
    (shaders_dir / "__init__.py").write_text("\n".join(init_lines))
    print(f"\n  {len(all_shader_variants)} shader files written")

    # Write dispatch.py
    dispatch_source = _generate_dispatch_file(all_plans)
    (output_dir / "dispatch.py").write_text(dispatch_source)
    print(f"  dispatch.py written ({len(all_plans)} functions)")

    # Write tensors/
    # Clean existing .py files in tensors/
    for f in tensors_dir.glob("*.py"):
        f.unlink()
    tensor_files = _generate_tensors_files(all_plans)
    for filename, content in tensor_files.items():
        (tensors_dir / f"{filename}.py").write_text(content)
    # Write tensors/__init__.py with re-exports
    init_lines = ['"""Generated tensor declarations."""', ""]
    for filename in sorted(tensor_files):
        # Collect all class names from this file's plans
        classes = [_plan_to_class_name(pn) for pn, _ in all_plans.items() if PLAN_TO_FILE.get(pn) == filename]
        for cls in classes:
            init_lines.append(f"from models.spike_qwen3_asr.tensors.{filename} import {cls}  # noqa: F401")
    init_lines.append("")
    (tensors_dir / "__init__.py").write_text("\n".join(init_lines))
    print(f"  tensors/ written ({len(tensor_files)} files)")

    # Clean up obsolete files
    for obsolete in ["plans.json", "shapes.json", "generated_text_layer.py"]:
        f = output_dir / obsolete
        if f.exists():
            f.unlink()
            print(f"  deleted {obsolete}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
