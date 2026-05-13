"""llama.cpp generated prefill flash attention shader for f16 K/V."""

from __future__ import annotations

from collections.abc import Mapping
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
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


_SOURCE = (Path(__file__).with_name("llama_flash_attn_f32_f16_f16.glsl")).read_text(
    encoding="utf-8"
)


def _scale(_tensors: Mapping[str, object], symbols: Mapping[str, int]) -> float:
    return 1.0 / (float(symbols["D"]) ** 0.5)


LLAMA_FLASH_ATTN_F32_F16_F16 = ShaderVariant(
    name="llama_flash_attn_f32_f16_f16",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="LlamaFlashAttnF32F16F16Program",
        shader_name="llama_flash_attn_f32_f16_f16",
        fields=(
            TensorFieldSpec(
                "q",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "T", "NH", "D")),
            ),
            TensorFieldSpec(
                "k",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "NK", "D")),
            ),
            TensorFieldSpec(
                "v",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "NK", "D")),
            ),
            TensorFieldSpec(
                "mask",
                IOKind.INPUT,
                "mask",
                TensorContract(dtype="float16", shape=("T", "S")),
            ),
            TensorFieldSpec(
                "sink",
                IOKind.INPUT,
                "unused",
                TensorContract(dtype="float32", shape=("B", "T", "NH", "D")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", "T", "NH", "D")),
            ),
            TensorFieldSpec(
                "mask_opt",
                IOKind.INPUT,
                "mask_opt",
                TensorContract(dtype="uint32", shape=("W", "R")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=128,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("KV", PushConstantType.UINT32, 4, "S"),
                PushConstantFieldSpec("ne1", PushConstantType.UINT32, 8, "NH"),
                PushConstantFieldSpec("ne2", PushConstantType.UINT32, 12, "T"),
                PushConstantFieldSpec("ne3", PushConstantType.UINT32, 16, "B"),
                PushConstantFieldSpec("neq2", PushConstantType.UINT32, 20, "NH"),
                PushConstantFieldSpec("neq3", PushConstantType.UINT32, 24, "B"),
                PushConstantFieldSpec("nek2", PushConstantType.UINT32, 28, "NK"),
                PushConstantFieldSpec("nek3", PushConstantType.UINT32, 32, "B"),
                PushConstantFieldSpec("nev2", PushConstantType.UINT32, 36, "NK"),
                PushConstantFieldSpec("nev3", PushConstantType.UINT32, 40, "B"),
                PushConstantFieldSpec("nem1", PushConstantType.UINT32, 44, "T"),
                PushConstantFieldSpec("nem2", PushConstantType.UINT32, 48, 1),
                PushConstantFieldSpec("nem3", PushConstantType.UINT32, 52, "B"),
                PushConstantFieldSpec("nb01", PushConstantType.UINT32, 56, mul("NH", "D")),
                PushConstantFieldSpec("nb02", PushConstantType.UINT32, 60, mul("D", 4)),
                PushConstantFieldSpec(
                    "nb03",
                    PushConstantType.UINT32,
                    64,
                    mul(mul(mul("T", "NH"), "D"), 4),
                ),
                PushConstantFieldSpec("nb11", PushConstantType.UINT32, 68, mul("NK", "D")),
                PushConstantFieldSpec("nb12", PushConstantType.UINT32, 72, mul("D", 2)),
                PushConstantFieldSpec(
                    "nb13",
                    PushConstantType.UINT32,
                    76,
                    mul(mul(mul("S", "NK"), "D"), 2),
                ),
                PushConstantFieldSpec("nb21", PushConstantType.UINT32, 80, mul("NK", "D")),
                PushConstantFieldSpec("nb22", PushConstantType.UINT32, 84, mul("D", 2)),
                PushConstantFieldSpec(
                    "nb23",
                    PushConstantType.UINT32,
                    88,
                    mul(mul(mul("S", "NK"), "D"), 2),
                ),
                PushConstantFieldSpec("scale", PushConstantType.FLOAT32, 92, _scale),
                PushConstantFieldSpec("max_bias", PushConstantType.FLOAT32, 96, 0.0),
                PushConstantFieldSpec("logit_softcap", PushConstantType.FLOAT32, 100, 0.0),
                PushConstantFieldSpec("mask_n_head_log2", PushConstantType.UINT32, 104, "NH"),
                PushConstantFieldSpec("m0", PushConstantType.FLOAT32, 108, 1.0),
                PushConstantFieldSpec("m1", PushConstantType.FLOAT32, 112, 1.0),
                PushConstantFieldSpec("gqa_ratio", PushConstantType.UINT32, 116, 1),
                PushConstantFieldSpec("split_kv", PushConstantType.UINT32, 120, "S"),
                PushConstantFieldSpec("k_num", PushConstantType.UINT32, 124, 1),
            ),
        ),
        dispatch=(ceil_div("T", 16), "NH", "B"),
    ),
    specialization_constants=(
        (0, 256),
        (1, 16),
        (2, 64),
        (3, 128),
        (4, 128),
        (5, 0),
        (6, 8),
        (7, 4),
        (8, 64),
        (9, 0),
        (10, 3),
        (11, 0),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source=_SOURCE,
)
