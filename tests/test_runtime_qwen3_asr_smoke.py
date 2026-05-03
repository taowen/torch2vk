from __future__ import annotations

import shutil

import numpy as np

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.runtime_smoke import (
    QWEN3_ASR_SMOKE_CHECKPOINT,
    QWEN3_ASR_SMOKE_WEIGHT_KEY,
    declare_qwen3_asr_runtime_smoke_tensors,
    run_qwen3_asr_runtime_smoke_frame,
)
from torch2vk.checkpoints.safetensors import open_safetensors_mmap
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.bootstrap import enumerate_compute_devices


def test_runtime_runs_one_qwen3_asr_weight_shader_frame(tmp_path) -> None:
    if shutil.which("glslangValidator") is None:
        raise AssertionError("glslangValidator is not installed")
    devices = enumerate_compute_devices()
    if not devices:
        raise AssertionError("no Vulkan compute devices available")
    model_dir = resolve_cached_model("Qwen/Qwen3-ASR-0.6B")

    tensors = declare_qwen3_asr_runtime_smoke_tensors(model_dir)
    with RuntimeSession.open(device_index=0, artifact_dir=tmp_path / "generated") as rt:
        rt.register_model(tensors.all(), model_dir=model_dir)
        actual = run_qwen3_asr_runtime_smoke_frame(rt, tensors)
        records = rt.dispatch_records

    with open_safetensors_mmap(model_dir / QWEN3_ASR_SMOKE_CHECKPOINT) as storage:
        expected = np.frombuffer(
            storage.buffer_slice(QWEN3_ASR_SMOKE_WEIGHT_KEY),
            dtype=np.uint16,
        ).copy()

    assert actual.dtype == np.dtype("uint16")
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert len(records) == 1
    assert records[0].frame == "qwen3_asr.runtime_smoke"
    assert records[0].shader == "qwen3_asr_weight_copy_bf16"
