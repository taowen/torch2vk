#!/usr/bin/env python3
"""Submit an empty Vulkan command buffer on the compute queue."""

from __future__ import annotations

from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    context = create_compute_context()
    try:
        pool = context.create_command_pool()
        fence = context.create_fence()
        try:
            command_buffer = pool.allocate_command_buffer()
            command_buffer.begin()
            command_buffer.end()
            command_buffer.submit_and_wait(fence)
            print(
                "command_submit=ok "
                f"device={context.physical_device_name} queue_family={context.compute_queue_family}"
            )
        finally:
            fence.close()
            pool.close()
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
