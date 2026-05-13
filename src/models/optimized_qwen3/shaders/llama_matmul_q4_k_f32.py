"""llama.cpp generated Q4_K prefill matmul shader."""

from __future__ import annotations

from pathlib import Path

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
from torch2vk.vulkan.shader_execution_requirements import (
    CooperativeMatrixRequirements,
    ShaderExecutionRequirements,
    SubgroupRequirements,
)
from torch2vk.vulkan.types import q4_k_words_layout


_SOURCE = (Path(__file__).with_name("llama_matmul_q4_k_f32.glsl")).read_text(
    encoding="utf-8"
)


def _variant(*, name: str, tile: int, block_size: int, bn: int, wn: int) -> ShaderVariant:
    return ShaderVariant(
        name=name,
        family="optimized_qwen3",
        contract=ShaderContract(
            class_name="LlamaMatmulQ4KF32Program",
            shader_name=name,
            fields=(
                TensorFieldSpec(
                    name="weight",
                    io_kind=IOKind.INPUT,
                    role="weight",
                    contract=TensorContract(
                        dtype="uint32",
                        shape=("N", "W"),
                        layout=q4_k_words_layout(logical_k="K"),
                    ),
                ),
                TensorFieldSpec(
                    name="x",
                    io_kind=IOKind.INPUT,
                    role="input",
                    contract=TensorContract(dtype="float32", shape=("X0", "X1", "K")),
                ),
                TensorFieldSpec(
                    name="output",
                    io_kind=IOKind.OUTPUT,
                    role="output",
                    contract=TensorContract(dtype="float32", shape=("X0", "X1", "N")),
                ),
            ),
            push_constants=PushConstantSpec(
                size=68,
                fields=(
                    PushConstantFieldSpec("M", PushConstantType.UINT32, 0, "N"),
                    PushConstantFieldSpec("P_N", PushConstantType.UINT32, 4, mul("X0", "X1")),
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 8, "K"),
                    PushConstantFieldSpec("stride_a", PushConstantType.UINT32, 12, "K"),
                    PushConstantFieldSpec("stride_b", PushConstantType.UINT32, 16, "K"),
                    PushConstantFieldSpec("stride_d", PushConstantType.UINT32, 20, "N"),
                    PushConstantFieldSpec("batch_stride_a", PushConstantType.UINT32, 24, mul("N", "K")),
                    PushConstantFieldSpec(
                        "batch_stride_b",
                        PushConstantType.UINT32,
                        28,
                        mul(mul("X0", "X1"), "K"),
                    ),
                    PushConstantFieldSpec(
                        "batch_stride_d",
                        PushConstantType.UINT32,
                        32,
                        mul(mul("X0", "X1"), "N"),
                    ),
                    PushConstantFieldSpec("base_work_group_z", PushConstantType.UINT32, 36, 0),
                    PushConstantFieldSpec("num_batches", PushConstantType.UINT32, 40, 1),
                    PushConstantFieldSpec("k_split", PushConstantType.UINT32, 44, "K"),
                    PushConstantFieldSpec("ne02", PushConstantType.UINT32, 48, 1),
                    PushConstantFieldSpec("ne12", PushConstantType.UINT32, 52, 1),
                    PushConstantFieldSpec("broadcast2", PushConstantType.UINT32, 56, 1),
                    PushConstantFieldSpec("broadcast3", PushConstantType.UINT32, 60, 1),
                    PushConstantFieldSpec("ne1", PushConstantType.UINT32, 64, mul("X0", "X1")),
                ),
            ),
            dispatch=(ceil_div("N", tile), ceil_div(mul("X0", "X1"), tile), 1),
        ),
        specialization_constants=(
            (0, block_size),
            (1, tile),
            (2, bn),
            (3, 32),
            (4, 64),
            (5, wn),
            (6, 2),
            (7, 16),
            (8, 16),
            (9, 16),
            (10, 64),
        ),
        execution_requirements=ShaderExecutionRequirements(
            subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
            cooperative_matrix=CooperativeMatrixRequirements(
                scope="subgroup",
                m_size=16,
                n_size=16,
                k_size=16,
                a_type="float16",
                b_type="float16",
                c_type="float16",
                result_type="float16",
            ),
            require_storage_buffer_16bit_access=True,
        ),
        source=_SOURCE,
    )


LLAMA_MATMUL_Q4_K_F32_M = _variant(
    name="llama_matmul_q4_k_f32_m",
    tile=64,
    block_size=128,
    bn=64,
    wn=32,
)
LLAMA_MATMUL_Q4_K_F32_L = _variant(
    name="llama_matmul_q4_k_f32_l",
    tile=128,
    block_size=256,
    bn=128,
    wn=64,
)
