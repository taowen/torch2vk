"""Generated shader: klein9b_cat_txt_img_f32."""

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
    add,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


KLEIN9B_CAT_TXT_IMG_F32 = ShaderVariant(
    name='klein9b_cat_txt_img_f32',
    family='klein9b',
    contract=ShaderContract(
        class_name='Klein9BCatTxtImgF32Program',
        shader_name='klein9b_cat_txt_img_f32',
        fields=(
            TensorFieldSpec(
                name='txt',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='img',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'S', 'H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('B', add('T', 'S'), 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('B', add('T', 'S')), 'H'), dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 4, 'T', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 12, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('B', add('T', 'S')), 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly TxtBuffer { float txt[]; };
layout(set = 0, binding = 1) buffer restrict readonly ImgBuffer { float img[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint T; uint S; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint h = idx % pc.H;
    const uint token = (idx / pc.H) % (pc.T + pc.S);
    const uint batch = idx / (pc.H * (pc.T + pc.S));
    if (token < pc.T) {
        output_values[idx] = txt[(batch * pc.T + token) * pc.H + h];
    } else {
        const uint img_token = token - pc.T;
        output_values[idx] = img[(batch * pc.S + img_token) * pc.H + h];
    }
}
""",
)
