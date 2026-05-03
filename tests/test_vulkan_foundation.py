from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

from torch2vk.vulkan.allocation import BufferSlice
from torch2vk.vulkan.bootstrap import enumerate_compute_devices
from torch2vk.vulkan.compute_pipeline import ComputePipeline, DescriptorBufferBinding
from torch2vk.vulkan.device import VulkanDevice


def test_elementwise_mul_compute_smoke(tmp_path) -> None:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        raise AssertionError("glslangValidator is not installed")
    devices = enumerate_compute_devices()
    if not devices:
        raise AssertionError("no Vulkan compute devices available")

    glsl_path = tmp_path / "elementwise_mul.comp"
    spv_path = tmp_path / "elementwise_mul.spv"
    glsl_path.write_text(
        """
#version 450

layout(set = 0, binding = 0) buffer restrict readonly XBuffer {
    float x[];
};

layout(set = 0, binding = 1) buffer restrict readonly WBuffer {
    float w[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OBuffer {
    float y[];
};

layout(push_constant) uniform Params {
    uint N;
} pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) {
        return;
    }
    y[i] = x[i] * w[i];
}
""".lstrip(),
        encoding="utf-8",
    )
    subprocess.run([compiler, "-V", str(glsl_path), "-o", str(spv_path)], check=True)

    values = [1.0, 2.0, 3.0, 4.0]
    weights = [10.0, 20.0, 30.0, 40.0]
    nbytes = len(values) * 4
    with VulkanDevice(physical_device_index=0) as device:
        x = device.allocate_host_visible_allocation(nbytes)
        w = device.allocate_host_visible_allocation(nbytes)
        out = device.allocate_host_visible_allocation(nbytes)
        try:
            x.buffer.write_bytes_at(x.offset, struct.pack(f"{len(values)}f", *values))
            w.buffer.write_bytes_at(w.offset, struct.pack(f"{len(weights)}f", *weights))
            device.memory_manager.host_upload_ring.flush(allocation=x)
            device.memory_manager.host_upload_ring.flush(allocation=w)
            with ComputePipeline(
                device,
                shader_spv_path=spv_path,
                storage_buffer_count=3,
                push_constant_size=4,
            ) as pipeline:
                pipeline.dispatch(
                    buffers=[
                        DescriptorBufferBinding(BufferSlice(x, x.offset, nbytes)),
                        DescriptorBufferBinding(BufferSlice(w, w.offset, nbytes)),
                        DescriptorBufferBinding(BufferSlice(out, out.offset, nbytes)),
                    ],
                    group_count_x=1,
                    push_constants=struct.pack("I", len(values)),
                )
            device.memory_manager.host_upload_ring.invalidate(allocation=out)
            result = struct.unpack(f"{len(values)}f", out.buffer.read_bytes_at(out.offset, nbytes))
        finally:
            out.close()
            w.close()
            x.close()

    assert result == pytest.approx([10.0, 40.0, 90.0, 160.0])
