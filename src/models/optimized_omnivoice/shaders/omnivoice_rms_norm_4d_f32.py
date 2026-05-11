"""Hand-written fused 4D RMSNorm shader for optimized OmniVoice."""

from __future__ import annotations

from models.optimized_omnivoice.shaders.omnivoice_rms_norm_3d_f32 import RMS_NORM_SOURCE
from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    mul,
)


OMNIVOICE_RMS_NORM_4D_F32 = ShaderVariant(
    name="omnivoice_rms_norm_4d_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceRmsNorm4DF32Program",
        shader_name="omnivoice_rms_norm_4d_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "D2", "H")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="float32", shape=("H",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "D2", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("rows", PushConstantType.UINT32, 0, mul(mul("D0", "D1"), "D2")),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(mul(mul("D0", "D1"), "D2"), 1, 1),
    ),
    source=RMS_NORM_SOURCE,
)
