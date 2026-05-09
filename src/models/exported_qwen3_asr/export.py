"""Export all exported_qwen3_asr submodules → shaders/, tensors/, dispatch.py.

Generates Python source files for the full ASR pipeline (audio tower + text).
Shapes are computed from the test fixture (tests/fixtures/qwen3_asr_asknot.wav).

Run from project root:
    .venv/bin/python -m models.exported_qwen3_asr.export
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from jinja2 import Environment, StrictUndefined

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


def _compare_extra_lines(plan_name: str, tensor_name: str) -> tuple[str, ...]:
    if plan_name == "run_audio_proj" and tensor_name == "linear_1":
        return (
            'compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),',
            'pytorch_probe=PyTorchProbe(kind="module_output", target="", selector="last_hidden_state"),',
        )
    if plan_name in {"run_lm_head", "run_decode_lm_head"} and tensor_name == "linear":
        return (
            'compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),',
            'pytorch_probe=PyTorchProbe(kind="module_output", target="", selector="logits"),',
        )
    return ()


_DISPATCH_EXTRA_IMPORTS = """import numpy as np

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from torch2vk.runtime.rope_table import run_rope_table_f32
"""


_ROPE_TENSOR_SOURCE = '''"""Generated RoPE tensor declarations."""

from __future__ import annotations

from torch2vk.runtime.rope_table import RopeTableTensors, declare_rope_table_tensors


def create_rope_table(
    prefix: str,
    *,
    batch: int,
    sequence_length: int,
    head_dim: int,
) -> RopeTableTensors:
    return declare_rope_table_tensors(
        prefix,
        batch=batch,
        sequence_length=sequence_length,
        head_dim=head_dim,
    )
'''


_MODEL_TENSOR_SOURCE = '''"""Generated model-level tensor wiring."""

from __future__ import annotations

from dataclasses import dataclass

from models.exported_qwen3_asr.tensors.audio_encoder import (
    AUDIO_ENCODER_OUTPUT,
    AudioEncoderTensors,
    create_audio_encoder,
)
from models.exported_qwen3_asr.tensors.audio_inject import (
    AudioInjectTensors,
    create_audio_inject,
)
from models.exported_qwen3_asr.tensors.decode_embed import (
    DecodeEmbedTensors,
    create_decode_embed,
)
from models.exported_qwen3_asr.tensors.decode_layer import (
    DecodeLayerTensors,
    create_decode_layer,
)
from models.exported_qwen3_asr.tensors.decode_lm_head import (
    DECODE_LM_HEAD_OUTPUT,
    DecodeLmHeadTensors,
    create_decode_lm_head,
)
from models.exported_qwen3_asr.tensors.decode_norm import (
    DecodeNormTensors,
    create_decode_norm,
)
from models.exported_qwen3_asr.tensors.embed_tokens import (
    EmbedTokensTensors,
    create_embed_tokens,
)
from models.exported_qwen3_asr.tensors.lm_head import (
    LM_HEAD_OUTPUT,
    LmHeadTensors,
    create_lm_head,
)
from models.exported_qwen3_asr.tensors.rope import RopeTableTensors, create_rope_table
from models.exported_qwen3_asr.tensors.text_layer import TextLayerTensors, create_text_layer
from models.exported_qwen3_asr.tensors.text_norm import TextNormTensors, create_text_norm
from torch2vk.runtime.logical import (
    bind_logical_tensor_names,
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class ExportedQwen3AsrTensors:
    input_ids: LogicalTensor
    attention_mask: LogicalTensor
    input_features: LogicalTensor
    feature_attention_mask: LogicalTensor
    position_ids: LogicalTensor
    audio_encoder: AudioEncoderTensors
    embed_tokens: EmbedTokensTensors
    audio_inject: AudioInjectTensors
    key_caches: tuple[LogicalTensor, ...]
    value_caches: tuple[LogicalTensor, ...]
    prefill_rope: RopeTableTensors
    decode_rope: RopeTableTensors
    text_layers: tuple[TextLayerTensors, ...]
    text_norm: TextNormTensors
    lm_head: LmHeadTensors
    decode_embed: DecodeEmbedTensors
    decode_layers: tuple[DecodeLayerTensors, ...]
    decode_norm: DecodeNormTensors
    decode_lm_head: DecodeLmHeadTensors
    eos_token_ids: LogicalTensor
    next_token: LogicalTensor
    done: LogicalTensor
    generated_tokens: LogicalTensor
    generated_length: LogicalTensor
    stopped: LogicalTensor
    token_index: LogicalTensor


_MODEL_TENSORS: ExportedQwen3AsrTensors | None = None


def create_model_tensors(
    *,
    input_ids_shape: tuple[int, ...],
    attention_mask_shape: tuple[int, ...],
    input_features_shape: tuple[int, ...],
    feature_attention_mask_shape: tuple[int, ...],
    prompt_length: int,
    max_sequence_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    max_new_tokens: int,
    eos_token_count: int,
) -> ExportedQwen3AsrTensors:
    input_ids = _host_input_tensor("int64", input_ids_shape)
    attention_mask = _host_input_tensor("int64", attention_mask_shape)
    input_features = _host_input_tensor("float32", input_features_shape)
    feature_attention_mask = _host_input_tensor("int64", feature_attention_mask_shape)
    position_ids = _host_input_tensor("int64", (3, 1, prompt_length))

    audio_encoder = create_audio_encoder(
        "spike.audio",
        request_state_outputs={AUDIO_ENCODER_OUTPUT},
    )
    embed_tokens = create_embed_tokens(
        "spike.text.embed",
        input=input_ids,
    )
    audio_inject = create_audio_inject(
        "spike.text.audio_inject",
        audio_features=audio_encoder.linear_110,
        index_copy=embed_tokens.embedding,
    )
    key_caches = tuple(
        _request_state_tensor(
            "float32",
            (1, num_key_value_heads, max_sequence_length, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for layer_idx in range(num_hidden_layers)
    )
    value_caches = tuple(
        _request_state_tensor(
            "float32",
            (1, num_key_value_heads, max_sequence_length, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for layer_idx in range(num_hidden_layers)
    )
    prefill_rope = create_rope_table(
        "spike.text.prefill.rope",
        batch=1,
        sequence_length=prompt_length,
        head_dim=head_dim,
    )
    decode_rope = create_rope_table(
        "spike.decode.rope",
        batch=1,
        sequence_length=1,
        head_dim=head_dim,
    )

    text_layers_list: list[TextLayerTensors] = []
    text_hidden = audio_inject.index_copy
    for layer_idx in range(num_hidden_layers):
        layer_tensors = create_text_layer(
            f"spike.text.layer.{layer_idx}",
            layer_idx=layer_idx,
            hidden_states=text_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=prefill_rope.cos,
            position_embeddings_1=prefill_rope.sin,
            cache_position=text_layers_list[0].cache_position if layer_idx > 0 else None,
        )
        text_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_layers = tuple(text_layers_list)

    text_norm = create_text_norm(
        "spike.text.norm",
        hidden_states=text_layers[-1].add_7,
    )
    lm_head = create_lm_head(
        "spike.text.lm_head",
        input=text_norm.mul_1,
        request_state_outputs={LM_HEAD_OUTPUT},
    )
    decode_embed = create_decode_embed(
        "spike.decode.embed",
        p_weight=embed_tokens.p_weight,
    )

    decode_layers_list: list[DecodeLayerTensors] = []
    decode_hidden = decode_embed.embedding
    for layer_idx, prefill_layer_tensors in enumerate(text_layers):
        layer_tensors = create_decode_layer(
            f"spike.decode.layer.{layer_idx}",
            layer_idx=layer_idx,
            p_input_layernorm_weight=prefill_layer_tensors.p_input_layernorm_weight,
            p_post_attention_layernorm_weight=(
                prefill_layer_tensors.p_post_attention_layernorm_weight
            ),
            p_attn_q_proj_weight=prefill_layer_tensors.p_attn_q_proj_weight,
            p_attn_k_proj_weight=prefill_layer_tensors.p_attn_k_proj_weight,
            p_attn_v_proj_weight=prefill_layer_tensors.p_attn_v_proj_weight,
            p_attn_o_proj_weight=prefill_layer_tensors.p_attn_o_proj_weight,
            p_attn_q_norm_weight=prefill_layer_tensors.p_attn_q_norm_weight,
            p_attn_k_norm_weight=prefill_layer_tensors.p_attn_k_norm_weight,
            p_mlp_gate_proj_weight=prefill_layer_tensors.p_mlp_gate_proj_weight,
            p_mlp_up_proj_weight=prefill_layer_tensors.p_mlp_up_proj_weight,
            p_mlp_down_proj_weight=prefill_layer_tensors.p_mlp_down_proj_weight,
            hidden_states=decode_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=decode_rope.cos,
            position_embeddings_1=decode_rope.sin,
            cache_position=decode_layers_list[0].cache_position if layer_idx > 0 else None,
        )
        decode_layers_list.append(layer_tensors)
        decode_hidden = layer_tensors.add_7
    decode_layers = tuple(decode_layers_list)

    decode_norm = create_decode_norm(
        "spike.decode.norm",
        p_weight=text_norm.p_weight,
        hidden_states=decode_layers[-1].add_7,
    )
    decode_lm_head = create_decode_lm_head(
        "spike.decode.lm_head",
        p_weight=lm_head.p_weight,
        input=decode_norm.mul_1,
        request_state_outputs={DECODE_LM_HEAD_OUTPUT},
    )

    eos_token_ids = _host_input_tensor("int64", (eos_token_count,))
    next_token = _request_output_tensor("int64", (1,))
    done = _request_output_tensor("uint32", (1,))
    generated_tokens = _request_state_tensor(
        "int64",
        (1, max_new_tokens),
        semantic=TensorSemantic.TOKEN,
    )
    generated_length = _request_state_tensor(
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    stopped = _request_state_tensor(
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    token_index = _host_input_tensor("int64", (1,))

    global _MODEL_TENSORS
    _MODEL_TENSORS = ExportedQwen3AsrTensors(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        position_ids=position_ids,
        audio_encoder=audio_encoder,
        embed_tokens=embed_tokens,
        audio_inject=audio_inject,
        key_caches=key_caches,
        value_caches=value_caches,
        prefill_rope=prefill_rope,
        decode_rope=decode_rope,
        text_layers=text_layers,
        text_norm=text_norm,
        lm_head=lm_head,
        decode_embed=decode_embed,
        decode_layers=decode_layers,
        decode_norm=decode_norm,
        decode_lm_head=decode_lm_head,
        eos_token_ids=eos_token_ids,
        next_token=next_token,
        done=done,
        generated_tokens=generated_tokens,
        generated_length=generated_length,
        stopped=stopped,
        token_index=token_index,
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    return _MODEL_TENSORS


def model_tensors() -> ExportedQwen3AsrTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors must be called before generated dispatch")
    return _MODEL_TENSORS


def _host_input_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _request_output_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _request_state_tensor(
    dtype: str,
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=semantic,
    )
'''


_ROPE_DISPATCH_HELPER = '''def run_rope_table(
    rt: RuntimeSession,
    *,
    phase: str,
    frame_name: str,
) -> None:
    tensors = model_tensors()
    if phase == "prefill":
        rope_t = tensors.prefill_rope
    elif phase == "decode":
        rope_t = tensors.decode_rope
    else:
        raise ValueError(f"unknown rope phase: {phase}")
    run_rope_table_f32(
        rt,
        start_position=rope_t.start_position,
        theta=rope_t.theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )'''


_DISPATCH_WRAPPERS = '''def run_audio_encoder(rt: RuntimeSession) -> None:
    _run_audio_encoder_with_tensors(rt, model_tensors().audio_encoder)


def run_embed_tokens(rt: RuntimeSession) -> None:
    _run_embed_tokens_with_tensors(rt, model_tensors().embed_tokens)


def run_audio_inject(rt: RuntimeSession) -> None:
    _run_audio_inject_with_tensors(rt, model_tensors().audio_inject)


def run_text_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_text_layer_with_tensors(rt, model_tensors().text_layers[layer_idx])


def run_text_norm(rt: RuntimeSession) -> None:
    _run_text_norm_with_tensors(rt, model_tensors().text_norm)


def run_lm_head(rt: RuntimeSession) -> None:
    _run_lm_head_with_tensors(rt, model_tensors().lm_head)


def run_decode_embed(rt: RuntimeSession) -> None:
    _run_decode_embed_with_tensors(rt, model_tensors().decode_embed)


def run_decode_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_decode_layer_with_tensors(rt, model_tensors().decode_layers[layer_idx])


def run_decode_norm(rt: RuntimeSession) -> None:
    _run_decode_norm_with_tensors(rt, model_tensors().decode_norm)


def run_decode_lm_head(rt: RuntimeSession) -> None:
    _run_decode_lm_head_with_tensors(rt, model_tensors().decode_lm_head)'''


_DECODE_STEP_HELPERS = '''def decode_step_inputs(
    *,
    token: int,
    cache_position: int,
    eos_token_array: np.ndarray,
    token_index_value: int,
) -> dict[LogicalTensor, np.ndarray]:
    tensors = model_tensors()
    if not tensors.decode_layers:
        raise ValueError("decode_layers must not be empty")
    return {
        tensors.decode_embed.input: np.array([[token]], dtype=np.int64),
        tensors.decode_layers[0].cache_position: np.array([cache_position], dtype=np.int64),
        tensors.eos_token_ids: np.ascontiguousarray(eos_token_array, dtype=np.int64),
        tensors.token_index: np.array([token_index_value], dtype=np.int64),
    }


def run_decode_step(
    rt: RuntimeSession,
    *,
    step: int,
) -> int:
    tensors = model_tensors()
    if not tensors.decode_layers:
        raise ValueError("decode_layers must not be empty")
    with rt.frame(f"spike.decode.{step:04d}"):
        run_decode_embed(rt)
        for layer_idx in range(len(tensors.decode_layers)):
            run_decode_layer(rt, layer_idx)
        run_decode_norm(rt)
        run_decode_lm_head(rt)
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=tensors.decode_lm_head.linear,
            eos_token_ids=tensors.eos_token_ids,
            next_token=tensors.next_token,
            done=tensors.done,
        )
        QWEN3_ASR_TOKEN_STORE_EOS_F32(
            rt,
            next_token=tensors.next_token,
            token_index=tensors.token_index,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )
    return int(rt.read_request_state(tensors.next_token).reshape(-1)[0])'''


_DISPATCH_FILE_TEMPLATE = '''"""Generated dispatch functions for all submodules."""

from __future__ import annotations

import sys
from typing import cast

{{ extra_imports_source }}
{% for item in shader_imports %}
from models.exported_qwen3_asr.shaders.{{ item.shader }} import {{ item.const }}
{% endfor %}
{% for item in tensor_imports %}
from models.exported_qwen3_asr.tensors.{{ item.file }} import {{ item.classes_source }}
{% endfor %}
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession


def shader_variant(shader_name: str) -> ShaderVariant:
    return cast(ShaderVariant, getattr(sys.modules[__name__], shader_name.upper()))


{{ dispatch_sources_source }}


{{ dispatch_wrappers_source }}


{{ rope_dispatch_helper }}


{{ decode_step_helpers }}


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
    dispatch_body = (
        "\n\n\n".join(bound_dispatch_sources)
        + "\n\n\n"
        + _DISPATCH_WRAPPERS
        + "\n\n\n"
        + _ROPE_DISPATCH_HELPER
        + "\n\n\n"
        + _DECODE_STEP_HELPERS
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
    return _JINJA.from_string(_DISPATCH_FILE_TEMPLATE).render(
        extra_imports_source=_DISPATCH_EXTRA_IMPORTS.rstrip("\n"),
        shader_imports=tuple(
            {"shader": shader_name, "const": all_shader_imports[shader_name]}
            for shader_name in sorted(all_shader_imports)
        ),
        tensor_imports=tensor_imports,
        dispatch_sources_source="\n\n\n".join(bound_dispatch_sources),
        dispatch_wrappers_source=_DISPATCH_WRAPPERS,
        rope_dispatch_helper=_ROPE_DISPATCH_HELPER,
        decode_step_helpers=_DECODE_STEP_HELPERS,
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
    at = model.thinker.audio_tower

    all_shader_imports: dict[str, str] = {}  # shader_name → CONST_NAME
    all_shader_variants: dict[str, ShaderVariant] = {}
    tensor_sources: dict[str, list[str]] = {}  # file_group → [class source, ...]
    tensor_file_classes: dict[str, list[str]] = {}  # file_group → [class names]
    dispatch_sources: list[str] = []

    def export_one(name, module, args, kwargs=None, weight_prefix="", kv_cache=None, kv_inject=None, layer_loop=None):
        prog = export_submodule(module, args=args, kwargs=kwargs, kv_cache=kv_cache)
        if kv_inject is not None:
            from torch2vk.export.graph import inject_kv_cache
            inject_kv_cache(prog, kv_inject)
        cls_name = _to_class_name(name)
        func_name = name.removeprefix("run_")
        group = func_name

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
                extra_lines_fn=lambda t: _compare_extra_lines(name, t),
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
                extra_lines_fn=lambda t: _compare_extra_lines(name, t),
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
               ))

    # Text pipeline exports
    pl = shapes["prompt_length"]
    max_seq = shapes["max_sequence_length"]
    hs = shapes["hidden_size"]
    hd = shapes["head_dim"]
    export_one("run_embed_tokens", model.thinker.model.embed_tokens.float(),
               args=(torch.zeros((1, pl), dtype=torch.long, device="meta"),),
               weight_prefix="thinker.model.embed_tokens.")
    with patched_forward(model.thinker, export_audio_inject_forward):
        export_one("run_audio_inject", model.thinker,
                   args=(torch.zeros(1, pl, hs, device="meta"),
                         torch.zeros(enc_seq, dtype=torch.long, device="meta"),
                         torch.zeros(enc_seq, hs, device="meta")))
    export_one("run_text_layer", model.thinker.model.layers[0],
               args=(torch.zeros(1, pl, hs, device="meta"),
                     (torch.zeros(1, pl, hd, device="meta"),
                      torch.zeros(1, pl, hd, device="meta"))),
               kwargs={"past_key_values": None, "attention_mask": None},
               weight_prefix="thinker.model.layers.0.",
               kv_inject=KVCacheInjectHint(phase="prefill", max_seq_len=max_seq))
    export_one("run_text_norm", model.thinker.model.norm.float(),
               args=(torch.zeros(1, pl, hs, device="meta"),),
               weight_prefix="thinker.model.norm.")
    export_one("run_lm_head", model.thinker.lm_head.float(),
               args=(torch.zeros(1, pl, hs, device="meta"),),
               weight_prefix="thinker.lm_head.")

    # Decode-step exports (seq_len=1)
    export_one("run_decode_embed", model.thinker.model.embed_tokens.float(),
               args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
               weight_prefix="thinker.model.embed_tokens.")
    export_one("run_decode_layer", model.thinker.model.layers[0],
               args=(torch.zeros(1, 1, hs, device="meta"),
                     (torch.zeros(1, 1, hd, device="meta"),
                      torch.zeros(1, 1, hd, device="meta"))),
               kwargs={"past_key_values": None, "attention_mask": None},
               weight_prefix="thinker.model.layers.0.",
               kv_inject=KVCacheInjectHint(phase="decode", max_seq_len=max_seq))
    export_one("run_decode_norm", model.thinker.model.norm.float(),
               args=(torch.zeros(1, 1, hs, device="meta"),),
               weight_prefix="thinker.model.norm.")
    export_one("run_decode_lm_head", model.thinker.lm_head.float(),
               args=(torch.zeros(1, 1, hs, device="meta"),),
               weight_prefix="thinker.lm_head.")

    # Write shaders/
    for f in shaders_dir.glob("*.py"):
        f.unlink()
    for shader_name, variant in all_shader_variants.items():
        (shaders_dir / f"{shader_name}.py").write_text(render_shader_file(variant))
    shader_init_imports = [
        f"from models.exported_qwen3_asr.shaders.{name} import {name.upper()}  # noqa: F401"
        for name in sorted(all_shader_variants)
    ]
    (shaders_dir / "__init__.py").write_text(render_simple_init("Generated shader index", shader_init_imports))
    print(f"\n  {len(all_shader_variants)} shader files written")

    # Write tensors/
    for f in tensors_dir.glob("*.py"):
        f.unlink()
    tensor_file_classes.setdefault("rope", []).append("RopeTableTensors")
    helper_source = render_tensor_helpers()
    for group, sources in tensor_sources.items():
        (tensors_dir / f"{group}.py").write_text(render_tensor_module(sources, helper_source))
    (tensors_dir / "rope.py").write_text(_ROPE_TENSOR_SOURCE)
    (tensors_dir / "model.py").write_text(_JINJA.from_string(_MODEL_TENSOR_SOURCE).render())
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

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
