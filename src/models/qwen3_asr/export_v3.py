"""Export qwen3_asr text decoder layer using torch2vk.export.

Demonstrates zero-config export: only the model + example inputs are needed.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import torch

from torch2vk.export import export_submodule, generate_dispatch_source


def _load_model() -> tuple[Any, Any]:
    config_path = Path.home() / (
        ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/"
        "snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/config.json"
    )
    payload = json.loads(config_path.read_text())

    with open(os.devnull, "w") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
                Qwen3ASRConfig,
            )
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )

            source_config = Qwen3ASRConfig(**payload)
            with torch.device("meta"):
                model = Qwen3ASRForConditionalGeneration(source_config)
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)

    return model, source_config


def export_text_layer() -> str:
    model, source_config = _load_model()
    text_config = source_config.thinker_config.text_config
    layer = model.thinker.model.layers[0]

    hidden_size = text_config.hidden_size
    head_dim = text_config.head_dim
    seq_len = 4

    hidden_states = torch.zeros((1, seq_len, hidden_size), device="meta")
    cos = torch.zeros((1, seq_len, head_dim), device="meta")
    sin = torch.zeros((1, seq_len, head_dim), device="meta")

    prog = export_submodule(
        layer,
        args=(hidden_states,),
        kwargs={"position_embeddings": (cos, sin)},
    )

    return generate_dispatch_source(
        prog,
        class_name="Qwen3AsrTextLayerTensors",
        function_name="run_text_layer",
    )


def main() -> int:
    source = export_text_layer()
    print(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
