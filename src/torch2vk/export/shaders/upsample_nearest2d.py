from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    node_input_shape,
    node_output_shape,
)
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

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint C; uint H; uint W; uint OH; uint OW; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.C * pc.OH * pc.OW;
    if (idx >= total) { return; }
    const uint ow = idx % pc.OW;
    const uint oh = (idx / pc.OW) % pc.OH;
    const uint c = (idx / (pc.OW * pc.OH)) % pc.C;
    const uint b = idx / (pc.OW * pc.OH * pc.C);
    const uint ih = min(oh * pc.H / pc.OH, pc.H - 1u);
    const uint iw = min(ow * pc.W / pc.OW, pc.W - 1u);
    const uint input_idx = ((b * pc.C + c) * pc.H + ih) * pc.W + iw;
    output_values[idx] = {{STORE}};
}
"""


def make_upsample_nearest2d_variant(
    node: Node,
    activation_dtype: str = "float32",
) -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if len(in_shape) != 4 or len(out_shape) != 4:
        return None
    in_contract = (1, "C", "H", "W")
    out_contract = (1, "C", "OH", "OW")
    total = mul(mul(mul(out_contract[0], out_contract[1]), out_contract[2]), out_contract[3])
    shader_name = "upsample_nearest2d_f32"
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportUpsampleNearest2dProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=in_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
            ),
            push_constants=PushConstantSpec(
                size=24,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, in_contract[0]),
                    PushConstantFieldSpec("C", PushConstantType.UINT32, 4, in_contract[1]),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 8, in_contract[2]),
                    PushConstantFieldSpec("W", PushConstantType.UINT32, 12, in_contract[3]),
                    PushConstantFieldSpec("OH", PushConstantType.UINT32, 16, out_contract[2]),
                    PushConstantFieldSpec("OW", PushConstantType.UINT32, 20, out_contract[3]),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(activation_dtype: str) -> str:
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE}}", activation_store("float(x[input_idx])", activation_dtype))
    )
