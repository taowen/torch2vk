from __future__ import annotations

import shutil

import pytest

from models.qwen3_asr.execution import run_qwen3_asr_audio_tower_position_embedding
from models.qwen3_asr.tensors.position_embedding import (
    declare_qwen3_asr_position_embedding_tensors,
)
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import SinusoidsPositionEmbedding
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.bootstrap import enumerate_compute_devices


def test_runtime_compares_qwen3_asr_position_embedding_frame(tmp_path) -> None:
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    if not enumerate_compute_devices():
        pytest.skip("no Vulkan compute devices available")

    seqlen = 9
    channels = 16
    tensors = declare_qwen3_asr_position_embedding_tensors(
        seqlen=seqlen,
        channels=channels,
    )
    pytorch_model = SinusoidsPositionEmbedding(length=32, channels=channels).eval()

    with RuntimeSession.open(device_index=0, artifact_dir=tmp_path / "generated") as rt:
        run_qwen3_asr_audio_tower_position_embedding(
            rt,
            tensors,
            pytorch_model=pytorch_model,
            seqlen=seqlen,
        )
        compare_results = rt.compare_results
        records = rt.dispatch_records

    assert len(records) == 1
    assert records[0].frame == "qwen3_asr.audio_tower.position_embedding"
    assert records[0].shader == "qwen3_asr_position_embedding_f32"
    assert records[0].writes[0] == ("output", tensors.output)
    assert records[0].logical_writes[0] == ("output", tensors.output.name)
    assert len(compare_results) == 1
    assert compare_results[0].tensor is tensors.output
    assert compare_results[0].passed
