from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.bootstrap import enumerate_compute_devices
from torch2vk.vulkan.compute_pipeline import (
    ComputePipeline,
    DescriptorBufferBinding,
    normalize_descriptor_types,
    normalize_specialization_constants,
)
from torch2vk.vulkan.device import VulkanDevice


def test_pipeline_normalizers() -> None:
    assert normalize_descriptor_types(descriptor_types=None, storage_buffer_count=3) == (7, 7, 7)
    assert normalize_specialization_constants({2: 7, 1: 5}) == ((1, 5), (2, 7))
    assert normalize_specialization_constants([9, 8]) == ((0, 9), (1, 8))


def test_descriptor_binding_uses_absolute_slice_offset() -> None:
    with pytest.raises(ValueError):
        BufferSlice(allocation=_fake_allocation(size=8), offset=4, nbytes=8)

    allocation = _fake_allocation(size=128, offset=32)
    binding = DescriptorBufferBinding(BufferSlice(allocation=allocation, offset=48, nbytes=64))
    assert binding.offset == 48
    assert binding.range == 64


def test_enumerate_compute_devices_smoke() -> None:
    try:
        devices = enumerate_compute_devices()
    except Exception as exc:
        pytest.skip(f"Vulkan device enumeration is unavailable: {exc}")
    assert all(device.queue_family_index >= 0 for device in devices)


def test_elementwise_mul_compute_smoke(tmp_path) -> None:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        pytest.skip("glslangValidator is not installed")
    try:
        devices = enumerate_compute_devices()
    except Exception as exc:
        pytest.skip(f"Vulkan device enumeration is unavailable: {exc}")
    if not devices:
        pytest.skip("no Vulkan compute devices available")

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


def _fake_allocation(*, size: int, offset: int = 0) -> BufferAllocation:
    from torch2vk.vulkan.buffer import VulkanBuffer

    buffer = VulkanBuffer(
        device_handle=object(),
        require_device_open=lambda: None,
        is_device_closed=lambda: True,
        handle=object(),
        memory=object(),
        size=256,
    )
    return BufferAllocation(buffer=buffer, offset=offset, size_bytes=size, pool="test")
