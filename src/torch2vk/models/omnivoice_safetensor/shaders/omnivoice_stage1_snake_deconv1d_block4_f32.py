"""Omnivoice Stage1 Snake Deconv1D Block4 F32."""

from __future__ import annotations

from torch2vk.shader import (
    Binding,
    BindingAccess,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    UniformBlock,
)

_SOURCE = """#version 460
layout(std430) buffer;
layout(set=0,binding=0) buffer restrict writeonly O{float t_output[];};
layout(set=0,binding=1) buffer restrict readonly X{float t_x[];};
layout(set=0,binding=2) buffer restrict readonly A{float t_alpha[];};
layout(set=0,binding=3) buffer restrict readonly W{float t_weight[];};
layout(set=0,binding=4) buffer restrict readonly B{float t_bias[];};
layout(set=0,binding=5) uniform restrict readonly U{ivec4 sizes;};
layout(local_size_x=128,local_size_y=1,local_size_z=1) in;
float snake(float v,float a){float aa=a+1.0e-9;float s=sin(a*v);return v+(s*s)/aa;}
void main(){uint oc=gl_GlobalInvocationID.x,to=gl_GlobalInvocationID.y,b=gl_GlobalInvocationID.z;uint si=uint(sizes.x),so=uint(sizes.y),ic=uint(sizes.z),ocn=uint(sizes.w);if(oc>=ocn||to>=so)return;uint al=uint(t_alpha.length());float acc=t_bias[oc];for(uint ii=0u;ii<ic;++ii){float a=t_alpha[ii%al];for(uint k=0u;k<6u;++k){int numer=int(to)+2-int(k);if((numer%3)!=0)continue;int ti=numer/3;if(ti<0||ti>=int(si))continue;acc+=snake(t_x[(b*si+uint(ti))*ic+ii],a)*t_weight[(ii*ocn+oc)*6u+k];}}t_output[(b*so+to)*ocn+oc]=acc;}
"""


OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK4_F32 = ShaderVariant(
    name="omnivoice_stage1_snake_deconv1d_block4_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_snake_deconv1d_block4_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "SI", "IC")),
            "alpha": TensorContract(dtype="float32", shape=(1, "A", 1)),
            "weight": TensorContract(dtype="float32", shape=("IC", "OC", 6)),
            "bias": TensorContract(dtype="float32", shape=("OC",)),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "SO", "OC")),
        },
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("x", 1, BindingAccess.READ),
            Binding("alpha", 2, BindingAccess.READ),
            Binding("weight", 3, BindingAccess.READ),
            Binding("bias", 4, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 5, ("SI", "SO", "IC", "OC")),),
        dispatch=("((OC) + (128) - 1)//(128)", "SO", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
