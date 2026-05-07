"""Integration test for torch2vk.export — validates Vulkan dispatch vs PyTorch on real model weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.export.graph import SKIP_OPS, export_submodule, is_alias_op, node_input_names
from torch2vk.export.registry import DEFAULT_REGISTRY
from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader import IOKind
from torch2vk.vulkan.types import TensorSpec
from torch.export.graph_signature import InputKind


def _load_model_and_config():
    model_dir = resolve_cached_model(REPO_ID)
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from torch2vk.exportv2 import instantiate_torch_module_on_meta

    payload = json.loads((Path(model_dir) / "config.json").read_text())
    source_config = Qwen3ASRConfig(**payload)
    model = instantiate_torch_module_on_meta(
        lambda: Qwen3ASRForConditionalGeneration(source_config)
    )
    return model, source_config, Path(model_dir)


def _dispatch_graph(rt, prog, tensors):
    graph = prog.graph_module.graph
    last_name = None
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in SKIP_OPS:
            continue
        if is_alias_op(node):
            inputs = node_input_names(node)
            if inputs and inputs[0] in tensors and node.name in tensors:
                src, dst = tensors[inputs[0]], tensors[node.name]
                rt._materialize_read(src)
                with dst.runtime_write_scope():
                    dst.buffer = src.buffer
                    dst.descriptor_nbytes = src.descriptor_nbytes
                    dst.version = src.version
            continue
        variant = DEFAULT_REGISTRY.resolve(node)
        if variant is None:
            raise RuntimeError(f"No shader for {target} ({node.name})")
        inputs = node_input_names(node)
        input_fields = [f for f in variant.contract.fields if f.io_kind == IOKind.INPUT]
        output_fields = [f for f in variant.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)]
        kwargs = {}
        for i, field in enumerate(input_fields):
            if i < len(inputs) and inputs[i] in tensors:
                kwargs[field.name] = tensors[inputs[i]]
        for field in output_fields:
            kwargs[field.name] = tensors[node.name]
        variant(rt, **kwargs)
        last_name = node.name
    return last_name


def _make_tensors(prog):
    graph = prog.graph_module.graph
    tensors: dict[str, LogicalTensor] = {}
    for spec in prog.graph_signature.input_specs:
        for node in graph.nodes:
            if node.name == spec.arg.name:
                tm = node.meta.get("tensor_meta")
                if tm:
                    tensors[spec.arg.name] = LogicalTensor(
                        name=spec.arg.name,
                        spec=TensorSpec(dtype="float32", shape=tuple(int(d) for d in tm.shape)),
                        role=TensorRole.INPUT,
                        memory=MemoryClass.HOST_INPUT,
                        lifetime=TensorLifetime.FRAME,
                    )
                break
    for node in graph.nodes:
        if node.op == "call_function" and node.name not in tensors:
            if str(node.target) in SKIP_OPS:
                continue
            tm = node.meta.get("tensor_meta")
            if tm:
                tensors[node.name] = LogicalTensor(
                    name=node.name,
                    spec=TensorSpec(dtype="float32", shape=tuple(int(d) for d in tm.shape)),
                    role=TensorRole.ACTIVATION,
                    memory=MemoryClass.FRAME_WORKSPACE,
                    lifetime=TensorLifetime.FRAME,
                )
    return tensors


def test_export_text_decoder_layer_matches_pytorch(tmp_path: Path) -> None:
    model, source_config, model_dir = _load_model_and_config()
    tc = source_config.thinker_config.text_config
    layer = model.thinker.model.layers[0]
    seq_len = 4

    prog = export_submodule(
        layer,
        args=(torch.zeros(1, seq_len, tc.hidden_size, device="meta"),),
        kwargs={
            "position_embeddings": (
                torch.zeros(1, seq_len, tc.head_dim, device="meta"),
                torch.zeros(1, seq_len, tc.head_dim, device="meta"),
            )
        },
    )

    tensors = _make_tensors(prog)
    weights = load_file(str(model_dir / "model.safetensors"))

    np.random.seed(42)
    h_data = np.random.randn(1, seq_len, tc.hidden_size).astype(np.float32) * 0.01
    cos_data = np.cos(
        np.arange(seq_len).reshape(1, seq_len, 1)
        * np.arange(tc.head_dim).reshape(1, 1, tc.head_dim)
        * 0.01
    ).astype(np.float32)
    sin_data = np.sin(
        np.arange(seq_len).reshape(1, seq_len, 1)
        * np.arange(tc.head_dim).reshape(1, 1, tc.head_dim)
        * 0.01
    ).astype(np.float32)

    rt = RuntimeSession(device_index=0)
    input_map: dict[LogicalTensor, np.ndarray] = {}
    for spec in prog.graph_signature.input_specs:
        t = tensors[spec.arg.name]
        if spec.kind == InputKind.PARAMETER:
            key = f"thinker.model.layers.0.{spec.target}"
            input_map[t] = weights[key].float().numpy()
        elif "hidden" in spec.arg.name:
            input_map[t] = h_data
        elif "position_embeddings_0" in spec.arg.name:
            input_map[t] = cos_data
        else:
            input_map[t] = sin_data
    rt.register_inputs(input_map)

    with rt.frame("test_text_layer"):
        last_name = _dispatch_graph(rt, prog, tensors)
        vulkan_out = rt.device.readback_tensor(
            spec=tensors[last_name].spec, slice=tensors[last_name].buffer
        )

    # PyTorch reference
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration as RealModel,
    )

    real_layer = RealModel(source_config).thinker.model.layers[0]
    state = {
        k.removeprefix("thinker.model.layers.0."): v
        for k, v in weights.items()
        if k.startswith("thinker.model.layers.0.")
    }
    real_layer.load_state_dict(state)
    real_layer = real_layer.float().eval()

    with torch.no_grad():
        out = real_layer(
            torch.from_numpy(h_data),
            position_embeddings=(torch.from_numpy(cos_data), torch.from_numpy(sin_data)),
        )
        expected = (out[0] if isinstance(out, tuple) else out).numpy()

    assert np.allclose(vulkan_out, expected, rtol=1e-4, atol=1e-4), (
        f"max diff = {np.abs(vulkan_out - expected).max():.2e}"
    )


def test_export_audio_encoder_layer_matches_pytorch(tmp_path: Path) -> None:
    model, source_config, model_dir = _load_model_and_config()
    audio_layer = model.thinker.audio_tower.layers[0]
    hidden_size = audio_layer.self_attn_layer_norm.normalized_shape[0]
    seq_len = 8

    h = torch.zeros(seq_len, hidden_size, device="meta")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="meta")
    prog = export_submodule(audio_layer, args=(h, cu))

    tensors = _make_tensors(prog)
    weights = load_file(str(model_dir / "model.safetensors"))

    np.random.seed(123)
    h_data = np.random.randn(seq_len, hidden_size).astype(np.float32) * 0.01
    cu_data = np.array([0, seq_len], dtype=np.int32)

    rt = RuntimeSession(device_index=0)
    input_map: dict[LogicalTensor, np.ndarray] = {}
    for spec in prog.graph_signature.input_specs:
        t = tensors[spec.arg.name]
        if spec.kind == InputKind.PARAMETER:
            key = f"thinker.audio_tower.layers.0.{spec.target}"
            input_map[t] = weights[key].float().numpy()
        elif "cu" in spec.arg.name:
            input_map[t] = cu_data
        else:
            input_map[t] = h_data
    rt.register_inputs(input_map)

    with rt.frame("test_audio_layer"):
        last_name = _dispatch_graph(rt, prog, tensors)
        vulkan_out = rt.device.readback_tensor(
            spec=tensors[last_name].spec, slice=tensors[last_name].buffer
        )

    # PyTorch reference
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration as RealModel,
    )

    real_layer = RealModel(source_config).thinker.audio_tower.layers[0]
    state = {
        k.removeprefix("thinker.audio_tower.layers.0."): v
        for k, v in weights.items()
        if k.startswith("thinker.audio_tower.layers.0.")
    }
    real_layer.load_state_dict(state)
    real_layer = real_layer.float().eval()

    with torch.no_grad():
        out = real_layer(
            torch.from_numpy(h_data),
            torch.from_numpy(cu_data),
        )
        expected = (out[0] if isinstance(out, tuple) else out).numpy()

    assert np.allclose(vulkan_out, expected, rtol=1e-3, atol=1e-3), (
        f"max diff = {np.abs(vulkan_out - expected).max():.2e}"
    )
