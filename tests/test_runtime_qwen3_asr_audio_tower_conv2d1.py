from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import run_qwen3_asr_audio_tower_conv2d1
from models.qwen3_asr.pytorch.example import REPO_ID
from models.qwen3_asr.shaders.audio_tower_conv2d1_gelu_f32 import (
    QWEN3_ASR_AUDIO_TOWER_CONV2D1_GELU_F32,
)
from models.qwen3_asr.tensors.audio_tower import (
    CONV2D1_BIAS_KEY,
    CONV2D1_WEIGHT_KEY,
    QWEN3_ASR_CHECKPOINT,
    declare_qwen3_asr_audio_tower_conv2d1_tensors,
)
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.bootstrap import enumerate_compute_devices


class _Conv2d1GeluReference(torch.nn.Module):
    def __init__(self, *, model_dir: Path) -> None:
        super().__init__()
        with safe_open(model_dir / QWEN3_ASR_CHECKPOINT, framework="pt", device="cpu") as storage:
            self.weight = storage.get_tensor(CONV2D1_WEIGHT_KEY).float()
            self.bias = storage.get_tensor(CONV2D1_BIAS_KEY).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(F.conv2d(x.float(), self.weight, self.bias, stride=2, padding=1))


def test_runtime_compares_qwen3_asr_audio_tower_conv2d1_first_shader(tmp_path) -> None:
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    if not enumerate_compute_devices():
        pytest.skip("no Vulkan compute devices available")

    model_dir = resolve_cached_model(REPO_ID)
    input_shape = (1, 1, 8, 8)
    tensors = declare_qwen3_asr_audio_tower_conv2d1_tensors(input_shape=input_shape)
    rng = np.random.default_rng(0)
    x_np = rng.normal(0.0, 0.1, size=input_shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    with RuntimeSession.open(
        device_index=0,
        artifact_dir=tmp_path / "generated",
        model_dir=model_dir,
    ) as rt:
        if not rt.device.supports_shader_execution_requirements(
            QWEN3_ASR_AUDIO_TOWER_CONV2D1_GELU_F32.execution_requirements
        ):
            pytest.skip("Vulkan device does not support 16-bit storage required by Qwen3-ASR BF16 weights")
        rt.register_inputs({tensors.input_features: x_np})
        run_qwen3_asr_audio_tower_conv2d1(
            rt,
            tensors,
            pytorch_model=_Conv2d1GeluReference(model_dir=model_dir).eval(),
            pytorch_input=x_torch,
        )
        assert tensors.weights.conv2d1_weight.buffer is not None
        assert tensors.weights.conv2d1_bias.buffer is not None
        compare_results = rt.compare_results
        records = rt.dispatch_records

    assert len(records) == 1
    assert records[0].frame == "qwen3_asr.audio_tower.conv2d1"
    assert records[0].shader == "qwen3_asr_audio_tower_conv2d1_gelu_f32"
    assert records[0].reads == (
        ("x", tensors.input_features),
        ("weight", tensors.weights.conv2d1_weight),
        ("bias", tensors.weights.conv2d1_bias),
    )
    assert records[0].writes == (("output", tensors.conv2d1_gelu),)
    assert len(compare_results) == 1
    assert compare_results[0].tensor is tensors.conv2d1_gelu
    assert compare_results[0].passed
