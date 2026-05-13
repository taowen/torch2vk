"""Q4_K matvec fused with residual add for Qwen3 decode."""

from __future__ import annotations

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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements
from torch2vk.vulkan.types import q4_k_words_layout

from models.quantized_qwen3.shaders.linear_nobias_q4_k_matvec_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32


def _source_with_residual_add() -> str:
    source = LINEAR_NOBIAS_Q4_K_MATVEC_F32.source
    source = source.replace(
        "layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };",
        "\n".join(
            (
                "layout(set = 0, binding = 2) buffer restrict readonly ResidualBuffer { float16_t residual_values[]; };",
                "layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };",
            )
        ),
    )
    source = source.replace(
        "if (col0 < pc.N) { output_values[row * pc.N + col0] = float16_t(acc0); }",
        "if (col0 < pc.N) { output_values[row * pc.N + col0] = float16_t(acc0 + float(residual_values[row * pc.N + col0])); }",
    )
    source = source.replace(
        "if (col1 < pc.N) { output_values[row * pc.N + col1] = float16_t(acc1); }",
        "if (col1 < pc.N) { output_values[row * pc.N + col1] = float16_t(acc1 + float(residual_values[row * pc.N + col1])); }",
    )
    return source


LINEAR_NOBIAS_Q4_K_MATVEC_ADD_F32 = ShaderVariant(
    name="linear_nobias_q4_k_matvec_add_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="OptimizedLinearNobiasQ4KMatvecAddProgram",
        shader_name="linear_nobias_q4_k_matvec_add_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("X0", "X1", "K")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(
                    dtype="uint32",
                    shape=("N", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K", block_size=256, words_per_block=36),
                ),
            ),
            TensorFieldSpec(
                name="residual",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("X0", "X1", "N")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float16", shape=("X0", "X1", "N")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, mul("X0", "X1"), dynamic=False),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K", dynamic=False),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 8, "N", dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div("N", 2), mul("X0", "X1"), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source=_source_with_residual_add(),
)
