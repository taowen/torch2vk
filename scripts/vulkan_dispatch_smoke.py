#!/usr/bin/env python3
"""Run a minimal Vulkan compute dispatch and read back the output buffer."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract
from torch2vk.vulkan_backend import create_compute_context


SOURCE = """#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutputBuffer { uint values[]; };
void main() {
    values[0] = 0x12345678u;
}
"""


SMOKE_VARIANT = ShaderVariant(
    name="dispatch_smoke",
    family="smoke",
    contract=ShaderContract(
        name="dispatch_smoke",
        inputs={},
        outputs={"output": TensorContract(dtype="int32", shape=(1,))},
        bindings=(Binding("output", 0, BindingAccess.WRITE),),
        dispatch=(1, 1, 1),
    ),
    source=SOURCE,
)


def main() -> int:
    spirv = _compile_smoke_shader()
    context = create_compute_context()
    try:
        output = context.create_host_buffer(nbytes=4)
        module = context.create_shader_module(spirv)
        descriptor_layout = context.create_descriptor_set_layout(SMOKE_VARIANT.contract)
        descriptor_pool = context.create_descriptor_pool(SMOKE_VARIANT.contract)
        pipeline_layout = context.create_pipeline_layout(SMOKE_VARIANT.contract, descriptor_layout)
        pipeline = context.create_compute_pipeline(
            shader_module=module,
            pipeline_layout=pipeline_layout,
        )
        fence = context.create_fence()
        command_pool = context.create_command_pool()
        try:
            descriptor_set = context.allocate_descriptor_set(
                descriptor_pool=descriptor_pool,
                descriptor_set_layout=descriptor_layout,
            )
            context.update_descriptor_set(
                descriptor_set,
                {0: output},
                descriptor_types={0: "storage_buffer"},
            )
            command_buffer = command_pool.allocate_command_buffer()
            command_buffer.begin()
            command_buffer.bind_compute_pipeline(pipeline)
            command_buffer.bind_descriptor_set(
                pipeline_layout=pipeline_layout,
                descriptor_set=descriptor_set,
            )
            command_buffer.dispatch(1)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)
            result = int.from_bytes(output.read(nbytes=4), byteorder="little", signed=False)
            if result != 0x12345678:
                raise RuntimeError(f"dispatch readback mismatch: {result:#x}")
            print(f"dispatch=ok result={result:#x}")
        finally:
            command_pool.close()
            fence.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            output.close()
    finally:
        context.close()
    return 0


def _compile_smoke_shader() -> bytes:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        raise FileNotFoundError("glslangValidator is required")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "dispatch_smoke.comp"
        output_path = Path(tmpdir) / "dispatch_smoke.spv"
        source_path.write_text(SOURCE, encoding="utf-8")
        subprocess.run(
            [
                compiler,
                "-V",
                "--target-env",
                "vulkan1.2",
                "-S",
                "comp",
                "-o",
                str(output_path),
                str(source_path),
            ],
            check=True,
        )
        return output_path.read_bytes()


if __name__ == "__main__":
    raise SystemExit(main())
