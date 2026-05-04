"""Replay descriptor rebind coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
)
from torch2vk.runtime.replay import execute_replay
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    ceil_div,
)
from torch2vk.vulkan.types import TensorSpec


_ADD_VARIANT = ShaderVariant(
    name="test_replay_rebind_add_f32",
    family="test_replay_rebind",
    contract=ShaderContract(
        class_name="TestReplayRebindAddF32",
        shader_name="test_replay_rebind_add_f32",
        fields=(
            TensorFieldSpec(
                name="lhs",
                io_kind=IOKind.INPUT,
                role="lhs",
                contract=TensorContract(dtype="float32", shape=("N",)),
            ),
            TensorFieldSpec(
                name="rhs",
                io_kind=IOKind.INPUT,
                role="rhs",
                contract=TensorContract(dtype="float32", shape=("N",)),
            ),
            TensorFieldSpec(
                name="out",
                io_kind=IOKind.OUTPUT,
                role="out",
                contract=TensorContract(dtype="float32", shape=("N",)),
            ),
        ),
        dispatch=(ceil_div("N", 64), 1, 1),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec(
                    name="N",
                    dtype=PushConstantType.UINT32,
                    offset=0,
                    value="N",
                ),
            ),
        ),
    ),
    source="""
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer Lhs { float lhs[]; };
layout(set = 0, binding = 1) readonly buffer Rhs { float rhs[]; };
layout(set = 0, binding = 2) writeonly buffer Out { float out_values[]; };

layout(push_constant) uniform Push {
    uint N;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.N) {
        out_values[i] = lhs[i] + rhs[i];
    }
}
""",
)


def test_replay_plan_rebinds_descriptors_without_recording(tmp_path: Path) -> None:
    lhs1 = _request_input("lhs")
    rhs1 = _request_input("rhs")
    out1 = _request_output("out")
    lhs2 = _request_input("lhs")
    rhs2 = _request_input("rhs")
    out2 = _request_output("out")

    with RuntimeSession.open(artifact_dir=tmp_path / "artifacts") as rt:
        rt.initialize_request_state(
            {
                lhs1: np.array([1, 2, 3, 4], dtype=np.float32),
                rhs1: np.array([10, 20, 30, 40], dtype=np.float32),
            }
        )
        dispatch_start = len(rt.dispatch_records)
        with rt.frame("warmup"):
            _ADD_VARIANT(rt, lhs=lhs1, rhs=rhs1, out=out1)
        warmup_records = rt.dispatch_records[dispatch_start:]

        plan = rt.build_replay_plan(
            name="test_replay_rebind",
            frame_dispatch_records=warmup_records,
            variants=(_ADD_VARIANT,),
            tensors_by_name={lhs1.name: lhs1, rhs1.name: rhs1, out1.name: out1},
        )
        try:
            execute_replay(plan)
            np.testing.assert_allclose(
                rt.read_request_state(out1),
                np.array([11, 22, 33, 44], dtype=np.float32),
            )

            command_buffer = plan.command_buffer
            dispatch_records = len(rt.dispatch_records)
            rt.cache_replay_plan("test_replay_rebind", plan)
            cached_plan = rt.cached_replay_plans("test_replay_rebind")[0]
            rt.initialize_request_state(
                {
                    lhs2: np.array([5, 6, 7, 8], dtype=np.float32),
                    rhs2: np.array([50, 60, 70, 80], dtype=np.float32),
                }
            )
            rt.rebind_replay_plan(
                cached_plan,
                tensors_by_name={lhs2.name: lhs2, rhs2.name: rhs2, out2.name: out2},
            )
            execute_replay(cached_plan)

            assert plan.command_buffer is command_buffer
            assert len(rt.dispatch_records) == dispatch_records
            np.testing.assert_allclose(
                rt.read_request_state(out2),
                np.array([55, 66, 77, 88], dtype=np.float32),
            )
        finally:
            plan.close()


def _request_input(name: str) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="float32", shape=(4,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _request_output(name: str) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="float32", shape=(4,)),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )
