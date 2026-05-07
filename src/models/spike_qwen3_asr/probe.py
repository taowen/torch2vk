"""Probe which aten ops are missing from the shader registry for full ASR pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.export import export_submodule
from torch2vk.export.graph import SKIP_OPS, is_alias_op
from torch2vk.export.registry import DEFAULT_REGISTRY


def _load_model():
    model_dir = resolve_cached_model(REPO_ID)
    with open(os.devnull, "w") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )

            payload = json.loads((Path(model_dir) / "config.json").read_text())
            source_config = Qwen3ASRConfig(**payload)
            with torch.device("meta"):
                model = Qwen3ASRForConditionalGeneration(source_config)
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
    return model, source_config


def _probe_submodule(name: str, module: torch.nn.Module, args, kwargs=None):
    print(f"\n{'='*60}")
    print(f"Probing: {name}")
    print(f"{'='*60}")
    try:
        prog = export_submodule(module, args=args, kwargs=kwargs)
    except Exception as e:
        print(f"  EXPORT FAILED: {e}")
        return

    graph = prog.graph_module.graph
    total = 0
    alias_count = 0
    covered = 0
    missing: dict[str, int] = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in SKIP_OPS:
            continue
        total += 1
        if is_alias_op(node):
            alias_count += 1
            continue
        variant = DEFAULT_REGISTRY.resolve(node)
        if variant is not None:
            covered += 1
        else:
            missing[target] = missing.get(target, 0) + 1

    print(f"  Total ops: {total} ({covered} covered, {alias_count} alias, {len(missing)} missing types)")
    if missing:
        print(f"  Missing ops:")
        for op, count in sorted(missing.items()):
            print(f"    {op} ×{count}")
    else:
        print(f"  All ops covered!")


def main() -> int:
    model, config = _load_model()
    tc = config.thinker_config.text_config

    # Text decoder layer
    layer = model.thinker.model.layers[0]
    seq_len = 4
    h = torch.zeros((1, seq_len, tc.hidden_size), device="meta")
    cos = torch.zeros((1, seq_len, tc.head_dim), device="meta")
    sin = torch.zeros((1, seq_len, tc.head_dim), device="meta")
    _probe_submodule(
        "text_decoder_layer",
        layer,
        args=(h,),
        kwargs={"position_embeddings": (cos, sin)},
    )

    # Audio encoder layer
    audio_layer = model.thinker.audio_tower.layers[0]
    audio_hidden = audio_layer.self_attn_layer_norm.normalized_shape[0]
    ah = torch.zeros(seq_len, audio_hidden, device="meta")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="meta")
    _probe_submodule("audio_encoder_layer", audio_layer, args=(ah, cu))

    # Embedding
    embed = model.thinker.model.embed_tokens
    input_ids = torch.zeros((1, seq_len), dtype=torch.long, device="meta")
    _probe_submodule("embed_tokens", embed, args=(input_ids,))

    # Audio tower conv layers (Conv2d: batch, in_channels, H, W)
    at = model.thinker.audio_tower
    _probe_submodule("audio_tower.conv2d1", at.conv2d1.float(), args=(torch.zeros((1, 1, 80, 100), device="meta"),))
    _probe_submodule("audio_tower.conv2d2", at.conv2d2.float(), args=(torch.zeros((1, 480, 40, 50), device="meta"),))
    _probe_submodule("audio_tower.conv2d3", at.conv2d3.float(), args=(torch.zeros((1, 480, 20, 25), device="meta"),))

    # Audio tower post-conv: proj1 + gelu + proj2
    _probe_submodule("audio_tower.proj1", at.proj1, args=(torch.zeros((8, 896), device="meta"),))
    _probe_submodule("audio_tower.proj2", at.proj2, args=(torch.zeros((8, 896), device="meta"),))

    # Final norm + lm_head (text output projection)
    norm = model.thinker.model.norm
    norm_input = torch.zeros((1, seq_len, tc.hidden_size), device="meta")
    _probe_submodule("text_final_norm", norm, args=(norm_input,))

    lm_head = model.thinker.lm_head
    lm_input = torch.zeros((1, seq_len, tc.hidden_size), device="meta")
    _probe_submodule("lm_head", lm_head, args=(lm_input,))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
