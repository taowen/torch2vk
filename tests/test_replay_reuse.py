"""Replay plan reuse across compatible tensor topologies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
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
    mul,
)
from torch2vk.vulkan.types import TensorSpec


@dataclass(slots=True)
class AddTensors:
    x: LogicalTensor
    y: LogicalTensor
    output: LogicalTensor


@dataclass(slots=True)
class StaticSymbolTensors:
    x: LogicalTensor
    temp: LogicalTensor


def test_replay_reuses_dynamic_push_constants_across_shape() -> None:
    small = _add_tensors(1, 2, 3)
    large = _add_tensors(1, 5, 7)
    small_x = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    small_y = np.full((1, 2, 3), 2.0, dtype=np.float32)
    large_x = np.arange(35, dtype=np.float32).reshape(1, 5, 7)
    large_y = np.full((1, 5, 7), -3.0, dtype=np.float32)

    with RuntimeSession.open(
        model_tensors=small,
        get_shader=_get_shader,
    ) as rt:
        with rt.request(inputs={"x": small_x, "y": small_y}):
            with rt.frame("add"):
                rt.dispatch(_ADD_DYNAMIC_PC_F32, x=small.x, y=small.y, output=small.output)
            plan = rt.build_replay_plan(name="add", frame="add")

        assert plan.dynamic_symbol_names == ("B", "H", "T")
        assert len(plan.params_entries) == 1
        assert plan.params_entries[0].params_layout is not None
        assert [field.name for field in plan.params_entries[0].params_layout.fields] == ["N"]

        rt._model_tensors = large
        with rt.request(inputs={"x": large_x, "y": large_y}):
            assert rt.replay_plan_compatible(plan)
            rt.rebind_replay_plan(plan)
            execute_replay(plan)

            np.testing.assert_allclose(
                rt.read_request_state(large.output),
                large_x + large_y,
                rtol=0,
                atol=0,
            )


def test_replay_rejects_rebinding_symbol_owned_by_static_descriptor() -> None:
    small = _static_symbol_tensors(8)
    large = _static_symbol_tensors(16)
    small_x = np.arange(8, dtype=np.float32)
    large_x = np.arange(16, dtype=np.float32)

    with RuntimeSession.open(
        model_tensors=small,
        get_shader=_get_shader,
    ) as rt:
        with rt.request(inputs={"x": small_x}):
            with rt.frame("static"):
                rt.dispatch(_STATIC_SYMBOL_F32, x=small.x, temp=small.temp)
            plan = rt.build_replay_plan(name="static", frame="static")

        assert plan.dynamic_symbol_names == ()

        rt._model_tensors = large
        with rt.request(inputs={"x": large_x}):
            assert not rt.replay_plan_compatible(plan)
            with pytest.raises(ValueError, match="cannot rebind static symbol"):
                rt.rebind_replay_plan(plan)


def _add_tensors(batch: int, tokens: int, hidden: int) -> AddTensors:
    return AddTensors(
        x=_input_tensor("x", batch, tokens, hidden),
        y=_input_tensor("y", batch, tokens, hidden),
        output=LogicalTensor(
            spec=TensorSpec("float32", (batch, tokens, hidden)),
            role=TensorRole.STATE,
            memory=MemoryClass.REQUEST_STATE,
            lifetime=TensorLifetime.REQUEST,
            name="output",
        ),
    )


def _static_symbol_tensors(hidden: int) -> StaticSymbolTensors:
    return StaticSymbolTensors(
        x=LogicalTensor(
            spec=TensorSpec("float32", (hidden,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            name="x",
        ),
        temp=LogicalTensor(
            spec=TensorSpec("float32", (hidden,)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            name="temp",
        ),
    )


def _input_tensor(name: str, batch: int, tokens: int, hidden: int) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec("float32", (batch, tokens, hidden)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
        name=name,
    )


_ADD_DYNAMIC_PC_F32 = ShaderVariant(
    name="REPLAY_TEST_ADD_DYNAMIC_PC_F32",
    family="replay_test",
    contract=ShaderContract(
        class_name="ReplayTestAddDynamicPc",
        shader_name="REPLAY_TEST_ADD_DYNAMIC_PC_F32",
        fields=(
            TensorFieldSpec(
                "x",
                IOKind.INPUT,
                "activation",
                TensorContract("float32", ("B", "T", "H")),
            ),
            TensorFieldSpec(
                "y",
                IOKind.INPUT,
                "activation",
                TensorContract("float32", ("B", "T", "H")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "activation",
                TensorContract("float32", ("B", "T", "H")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "T"), "H"), 64), 1, 1),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec(
                    "N",
                    PushConstantType.UINT32,
                    0,
                    mul(mul("B", "T"), "H"),
                ),
            ),
        ),
    ),
    source="""#version 450

layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        output_values[idx] = x[idx] + y[idx];
    }
}
""",
)

_STATIC_SYMBOL_F32 = ShaderVariant(
    name="REPLAY_TEST_STATIC_SYMBOL_F32",
    family="replay_test",
    contract=ShaderContract(
        class_name="ReplayTestStaticSymbol",
        shader_name="REPLAY_TEST_STATIC_SYMBOL_F32",
        fields=(
            TensorFieldSpec(
                "x",
                IOKind.INPUT,
                "activation",
                TensorContract("float32", ("H",)),
            ),
            TensorFieldSpec(
                "temp",
                IOKind.OUTPUT,
                "activation",
                TensorContract("float32", ("H",)),
            ),
        ),
        dispatch=(ceil_div("H", 64), 1, 1),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec("H", PushConstantType.UINT32, 0, "H"),
            ),
        ),
    ),
    source="""#version 450

layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly TempBuffer { float temp[]; };
layout(push_constant) uniform PushConstants { uint H; } pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.H) {
        temp[idx] = x[idx];
    }
}
""",
)

_SHADERS = {
    _ADD_DYNAMIC_PC_F32.name: _ADD_DYNAMIC_PC_F32,
    _STATIC_SYMBOL_F32.name: _STATIC_SYMBOL_F32,
}


def _get_shader(name: str) -> ShaderVariant:
    return _SHADERS[name]
