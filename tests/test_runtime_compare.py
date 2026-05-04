from __future__ import annotations

import json

import numpy as np
import pytest

from torch2vk.runtime.compare import CompareAssertionError, compare_arrays
from torch2vk.runtime.frame import FrameContext
from torch2vk.runtime.logical import (
    ComparePolicy,
    DispatchWriter,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
)
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.types import TensorSpec


def _tensor(name: str = "frame.output") -> LogicalTensor:
    tensor = LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="float32", shape=(2, 3)),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
        compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
    )
    with tensor.runtime_write_scope():
        tensor.writer = DispatchWriter(frame="frame", shader="shader", dispatch_index=7)
    return tensor


def _bare_session(tmp_path) -> RuntimeSession:
    session = RuntimeSession.__new__(RuntimeSession)
    session.artifact_dir = tmp_path
    session.model_dir = None
    session._inputs = {}
    session._compare_results = []
    session._dispatch_records = []
    return session


def test_compare_arrays_reports_first_mismatch_and_dumps_artifacts(tmp_path) -> None:
    tensor = _tensor()
    candidate = np.array([[1.0, 2.0, 3.0], [4.0, 50.0, 6.0]], dtype=np.float32)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    with pytest.raises(CompareAssertionError) as raised:
        compare_arrays(
            tensor=tensor,
            frame="frame",
            candidate=candidate,
            expected=expected,
            artifact_dir=tmp_path,
            nearest_upstream_artifact_key="frame/previous",
        )

    result = raised.value.result
    assert result.first_mismatch_index == (1, 1)
    assert result.candidate_value == 50.0
    assert result.expected_value == 5.0
    assert result.candidate_shape == (2, 3)
    assert result.expected_shape == (2, 3)
    assert result.nearest_upstream_artifact_key == "frame/previous"
    assert result.candidate_artifact_path is not None
    assert result.expected_artifact_path is not None
    assert result.summary_artifact_path is not None
    np.testing.assert_array_equal(np.load(result.candidate_artifact_path), candidate)
    np.testing.assert_array_equal(np.load(result.expected_artifact_path), expected)
    summary = json.loads(open(result.summary_artifact_path, encoding="utf-8").read())
    assert summary["first_mismatch_index"] == [1, 1]
    assert summary["writer"]["shader"] == "shader"


def test_compare_arrays_records_shape_mismatch_as_structured_result(tmp_path) -> None:
    tensor = _tensor()

    with pytest.raises(CompareAssertionError) as raised:
        compare_arrays(
            tensor=tensor,
            frame="frame",
            candidate=np.zeros((2, 3), dtype=np.float32),
            expected=np.zeros((6,), dtype=np.float32),
            artifact_dir=tmp_path,
        )

    result = raised.value.result
    assert result.failure_reason == "shape_mismatch"
    assert result.candidate_shape == (2, 3)
    assert result.expected_shape == (6,)
    assert result.candidate_artifact_path is not None
    assert "shape_mismatch" in str(raised.value)


def test_pytorch_artifact_cache_reuses_matching_input_fingerprint(tmp_path) -> None:
    session = _bare_session(tmp_path)
    input_tensor = LogicalTensor(
        name="frame.x",
        spec=TensorSpec(dtype="float32", shape=(2,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )
    output_tensor = LogicalTensor(
        name="frame.y",
        spec=TensorSpec(dtype="float32", shape=(2,)),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
        compare=ComparePolicy(kind="tensor"),
        pytorch_probe=PyTorchProbe(kind="module_output", target=""),
    )
    frame = FrameContext(frame="frame", start_dispatch_index=0)
    model = object()

    session._inputs = {input_tensor: np.array([1.0, 2.0], dtype=np.float32)}
    session._store_cached_pytorch_artifacts(
        frame,
        [output_tensor],
        model,
        {"frame.y": np.array([3.0, 4.0], dtype=np.float32)},
    )
    cached, missing = session._load_cached_pytorch_artifacts(frame, [output_tensor], model)

    assert missing == []
    np.testing.assert_array_equal(cached["frame.y"], np.array([3.0, 4.0], dtype=np.float32))

    session._inputs = {input_tensor: np.array([1.0, 5.0], dtype=np.float32)}
    cached, missing = session._load_cached_pytorch_artifacts(frame, [output_tensor], model)

    assert cached == {}
    assert missing == [output_tensor]
